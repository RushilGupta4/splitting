from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diffusion import probabilistic_split_with_run_ids
from ks import (
    coerce_samples_np,
    compute_reference_ks_distance,
    compute_target_ks_distance,
    prepare_reference_cdf_state,
    warm_ks_kernel_for_mode,
    warm_reference_ks_kernel,
)
from runners.base import BaseRunner, ComparisonModeSpec, SamplingConfig
from runners.edm_gmm2d import target as edm_target
from runners.edm_gmm2d.model import EDMDenoiser
from runners.edm_gmm2d.sampling import (
    EDMSchedule,
    edm_stochastic_sample_segment,
    heun_sample_segment,
    resolve_split_percentages as _resolve_split_percentages,
    sde_euler_maruyama_sample_segment,
)

SUPPORTED_SOLVERS = ("heun",)
SUPPORTED_SAMPLERS = ("heun", "edm_stochastic", "sde_euler_maruyama")

COMPARISON_MODES = (
    ComparisonModeSpec(
        name="true_dist",
        requires_reference_cache=False,
        reference_uses_sampling_config=False,
        description="Exact lower-orthant KS against analytic 2D GMM CDF.",
    ),
    ComparisonModeSpec(
        name="true_samples",
        requires_reference_cache=True,
        reference_uses_sampling_config=False,
        description="Two-sample lower-orthant KS against target samples.",
    ),
    ComparisonModeSpec(
        name="edm_samples",
        requires_reference_cache=True,
        reference_uses_sampling_config=True,
        description="Two-sample lower-orthant KS against generated EDM samples.",
    ),
)
_COMPARISON_MODE_NAMES = {spec.name for spec in COMPARISON_MODES}


def _rectangle_indicator_grid(values: torch.Tensor, x_grid: Sequence[float]) -> torch.Tensor:
    thresholds = torch.as_tensor(x_grid, device=values.device, dtype=values.dtype)
    x1_below = values[:, 0:1] <= thresholds.unsqueeze(0)
    x2_below = values[:, 1:2] <= thresholds.unsqueeze(0)
    return (
        (x1_below.unsqueeze(2) & x2_below.unsqueeze(1))
        .to(values.dtype)
        .reshape(values.shape[0], -1)
    )


class EDMGMM2DRunner(BaseRunner):
    runner_name = "edm_gmm2d"
    DEFAULT_SOLVER = "heun"

    def __init__(
        self,
        *,
        model,
        target_spec: Mapping[str, Any],
        data_mean: torch.Tensor,
        data_std: torch.Tensor,
        sampling_steps: int,
        sampler: str = "heun",
        sigma_min: float,
        sigma_max: float,
        rho: float,
        S_churn: float,
        S_min: float = 0.0,
        S_max: float = 80.0,
        S_noise: float = 1.0,
        sigma_data: float,
        device: str,
        checkpoint_path: str | None = None,
    ):
        sampler = str(sampler)
        if sampler not in SUPPORTED_SAMPLERS:
            raise ValueError(f"Unknown EDM sampler {sampler!r}. Available: {SUPPORTED_SAMPLERS}")
        if float(S_churn) < 0.0:
            raise ValueError("S_churn must be >= 0")
        if float(S_noise) < 0.0:
            raise ValueError("S_noise must be >= 0")
        if float(S_min) > float(S_max):
            raise ValueError("S_min must be <= S_max")
        self._model = model
        self._target_spec = dict(target_spec)
        self._data_mean = data_mean
        self._data_std = data_std
        self._sampling_steps = int(sampling_steps)
        self._sampler = sampler
        self._sigma_min = float(sigma_min)
        self._sigma_max = float(sigma_max)
        self._rho = float(rho)
        self._S_churn = float(S_churn)
        self._S_min = float(S_min)
        self._S_max = float(S_max)
        self._S_noise = float(S_noise)
        self._sigma_data = float(sigma_data)
        self._device = str(device)
        self._checkpoint_path = checkpoint_path or type(self).default_checkpoint_path()
        self._schedule = EDMSchedule(
            sampling_steps=self._sampling_steps,
            sigma_min=self._sigma_min,
            sigma_max=self._sigma_max,
            rho=self._rho,
            device=self._device,
        )

    @classmethod
    def add_train_args(cls, parser) -> None:
        parser.add_argument("--sigma_min", type=float, default=0.002)
        parser.add_argument("--sigma_max", type=float, default=80.0)
        parser.add_argument("--rho", type=float, default=7.0)
        parser.add_argument("--sampling_steps", type=int, default=18)
        parser.add_argument("--sigma_data", type=float, default=1.0)
        parser.add_argument("--P_mean", type=float, default=-1.2)
        parser.add_argument("--P_std", type=float, default=1.2)
        parser.add_argument("--num_samples", type=int, default=100_000)
        parser.add_argument("--batch_size", type=int, default=10_000)
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--num_blocks", type=int, default=4)
        parser.add_argument("--num_plot_samples", type=int, default=20_000)
        parser.add_argument("--num_plot_bins", type=int, default=100)
        parser.add_argument("--marginal_plot_path", type=str, default=None)
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0" if torch.cuda.is_available() else "cpu",
        )

    @classmethod
    def train_from_args(cls, args) -> None:
        from runners.edm_gmm2d.train import train as _train

        _train(args)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | None = None,
        *,
        device: str,
        no_compile: bool = False,
        **kwargs,
    ) -> "EDMGMM2DRunner":
        path = checkpoint_path or cls.default_checkpoint_path()
        checkpoint = torch.load(path, map_location=device)
        model_args = checkpoint.get("args", {})
        sigma_data = float(checkpoint.get("sigma_data", model_args.get("sigma_data", 1.0)))
        model = EDMDenoiser(
            input_dim=len(checkpoint.get("data_mean", [0.0, 0.0])),
            hidden_dim=int(model_args.get("hidden_dim", 128)),
            num_blocks=int(model_args.get("num_blocks", 4)),
            sigma_data=sigma_data,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        if hasattr(torch, "compile") and not no_compile:
            model = torch.compile(model, dynamic=True)

        defaults = {
            "sampling_steps": 18,
            "sampler": "heun",
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": 80.0,
            "S_noise": 1.0,
        }
        defaults.update(checkpoint.get("sampling_defaults", {}))
        defaults.update(kwargs)
        return cls(
            model=model,
            target_spec=checkpoint.get("target_spec", edm_target.get_target_distribution_spec()),
            data_mean=torch.as_tensor(checkpoint["data_mean"], device=device, dtype=torch.float32),
            data_std=torch.as_tensor(checkpoint["data_std"], device=device, dtype=torch.float32),
            sigma_data=sigma_data,
            device=device,
            checkpoint_path=path,
            **defaults,
        )

    def with_sampling_config(self, **kwargs) -> "EDMGMM2DRunner":
        allowed = {
            "sampler",
            "sampling_steps",
            "sigma_min",
            "sigma_max",
            "rho",
            "S_churn",
            "S_min",
            "S_max",
            "S_noise",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.with_sampling_config got unknown keys: {sorted(unknown)}"
            )
        return EDMGMM2DRunner(
            model=self._model,
            target_spec=self._target_spec,
            data_mean=self._data_mean,
            data_std=self._data_std,
            sampling_steps=int(kwargs.get("sampling_steps", self._sampling_steps)),
            sampler=str(kwargs.get("sampler", self._sampler)),
            sigma_min=float(kwargs.get("sigma_min", self._sigma_min)),
            sigma_max=float(kwargs.get("sigma_max", self._sigma_max)),
            rho=float(kwargs.get("rho", self._rho)),
            S_churn=float(kwargs.get("S_churn", self._S_churn)),
            S_min=float(kwargs.get("S_min", self._S_min)),
            S_max=float(kwargs.get("S_max", self._S_max)),
            S_noise=float(kwargs.get("S_noise", self._S_noise)),
            sigma_data=self._sigma_data,
            device=self._device,
            checkpoint_path=self._checkpoint_path,
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def input_dim(self) -> int:
        return int(getattr(self._model, "input_dim", getattr(getattr(self._model, "_orig_mod", None), "input_dim", 2)))

    @property
    def target_spec(self) -> Mapping[str, Any]:
        return self._target_spec

    @property
    def sampling_config(self) -> SamplingConfig:
        return SamplingConfig(
            values={
                "sampler": str(self._sampler),
                "sampling_steps": int(self._sampling_steps),
                "sigma_min": float(self._sigma_min),
                "sigma_max": float(self._sigma_max),
                "rho": float(self._rho),
                "S_churn": float(self._S_churn),
                "S_min": float(self._S_min),
                "S_max": float(self._S_max),
                "S_noise": float(self._S_noise),
            }
        )

    @property
    def start_time(self) -> Any:
        return float(self._sigma_max)

    @property
    def end_time(self) -> Any:
        return 0.0

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    def comparison_modes(self) -> Sequence[ComparisonModeSpec]:
        return COMPARISON_MODES

    def _validate_mode(self, comparison_mode: str) -> None:
        if comparison_mode not in _COMPARISON_MODE_NAMES:
            raise ValueError(f"Unknown comparison_mode {comparison_mode!r}. Available: {sorted(_COMPARISON_MODE_NAMES)}")

    def sample_prior(self, num_samples: int, *, generator=None) -> torch.Tensor:
        return torch.randn(
            int(num_samples), self.input_dim, device=self._device, generator=generator
        ) * float(self._sigma_max)

    def sample_segment(self, x, start_time, end_time, *, generator=None):
        if self._sampler == "heun":
            return heun_sample_segment(self._model, x, float(start_time), float(end_time), self._schedule)
        if self._sampler == "edm_stochastic":
            return edm_stochastic_sample_segment(
                self._model,
                x,
                float(start_time),
                float(end_time),
                self._schedule,
                S_churn=self._S_churn,
                S_min=self._S_min,
                S_max=self._S_max,
                S_noise=self._S_noise,
                generator=generator,
            )
        if self._sampler == "sde_euler_maruyama":
            return sde_euler_maruyama_sample_segment(
                self._model, x, float(start_time), float(end_time), self._schedule, generator=generator
            )
        raise ValueError(f"Unknown EDM sampler {self._sampler!r}")

    def postprocess_samples(self, native_samples: torch.Tensor) -> torch.Tensor:
        return edm_target.denormalize(native_samples, self._data_mean, self._data_std)

    def resolve_split_percentages(self, split_percentages: Sequence[float]):
        return _resolve_split_percentages(self._schedule, split_percentages)

    def segment_cost(self, start_time: Any, end_time: Any) -> float:
        start_idx = self._schedule.index_for_sigma(float(start_time))
        end_idx = self._schedule.index_for_sigma(float(end_time))
        if start_idx > end_idx:
            raise ValueError(f"EDM segment must move toward lower sigma, got {start_time} -> {end_time}")
        total = 0
        for idx in range(start_idx, end_idx):
            sigma_next = float(self._schedule.sigmas[idx + 1].item())
            total += self._transition_cost(sigma_next)
        return float(total)

    def segment_costs(self, split_points: Sequence[Any]) -> list:
        starts = [self.start_time] + [float(p) for p in split_points]
        ends = [float(p) for p in split_points] + [self.end_time]
        return [self.segment_cost(s, e) for s, e in zip(starts, ends)]

    def expected_cost_per_root(self, split_points, split_factors) -> float:
        if not split_points:
            return self.segment_cost(self.start_time, self.end_time)
        cost = self.segment_cost(self.start_time, split_points[0])
        cumulative_split = 1.0
        for idx, split_factor in enumerate(split_factors):
            cumulative_split *= float(split_factor)
            end_t = split_points[idx + 1] if idx + 1 < len(split_points) else self.end_time
            cost += cumulative_split * self.segment_cost(split_points[idx], end_t)
        return float(cost)

    def run_split_batch(self, *, n0_by_run, split_points, split_factors_by_run, generator=None):
        if len(n0_by_run) == 0:
            return [], [], 0.0
        if len(n0_by_run) != len(split_factors_by_run):
            raise ValueError("n0_by_run and split_factors_by_run must have the same length")
        n0_tensor = torch.as_tensor(n0_by_run, device=self._device, dtype=torch.long)
        if torch.any(n0_tensor < 1):
            raise ValueError("all n0 values must be at least 1")
        num_runs = int(n0_tensor.numel())
        run_ids = torch.repeat_interleave(torch.arange(num_runs, device=self._device), n0_tensor)
        x = self.sample_prior(int(n0_tensor.sum().item()), generator=generator)
        realized_costs = torch.zeros(num_runs, device=self._device, dtype=torch.long)
        sampling_start = time.perf_counter()

        split_points = [float(p) for p in split_points]
        if not split_points:
            realized_costs += n0_tensor * int(self.segment_cost(self.start_time, self.end_time))
            x = self.sample_segment(x, self.start_time, self.end_time, generator=generator)
        else:
            realized_costs += n0_tensor * int(self.segment_cost(self.start_time, split_points[0]))
            x = self.sample_segment(x, self.start_time, split_points[0], generator=generator)
            split_factors_tensor = torch.as_tensor(
                np.asarray(split_factors_by_run, dtype=float), device=self._device, dtype=x.dtype
            )
            if split_factors_tensor.shape != (num_runs, len(split_points)):
                raise ValueError("split_factors_by_run has incompatible shape")
            for idx, split_point in enumerate(split_points):
                x, run_ids = probabilistic_split_with_run_ids(
                    x, run_ids, split_factors_tensor[:, idx], generator=generator
                )
                end_t = split_points[idx + 1] if idx + 1 < len(split_points) else self.end_time
                counts = torch.bincount(run_ids, minlength=num_runs)
                realized_costs += counts * int(self.segment_cost(split_point, end_t))
                x = self.sample_segment(x, split_point, end_t, generator=generator)

        x = self.postprocess_samples(x)
        sampling_time = time.perf_counter() - sampling_start
        samples_by_run = [x[run_ids == run_idx] for run_idx in range(num_runs)]
        return samples_by_run, realized_costs.detach().cpu().tolist(), sampling_time

    def solver_names(self) -> Sequence[str]:
        return SUPPORTED_SOLVERS

    def solver_cost(self, solver: str, **solver_kwargs) -> float:
        if solver != "heun":
            raise ValueError(f"Unknown solver {solver!r}")
        sampling_steps = int(solver_kwargs.get("sampling_steps", self._sampling_steps))
        return float(2 * sampling_steps - 1)

    def solver_cache_key(self, solver: str, **solver_kwargs) -> Mapping[str, Any]:
        if solver != "heun":
            raise ValueError(f"Unknown solver {solver!r}")
        key: dict[str, Any] = {}
        if "sampling_steps" in solver_kwargs:
            key["sampling_steps"] = int(solver_kwargs["sampling_steps"])
        for name in sorted(set(solver_kwargs) - set(key) - {"eta"}):
            key[name] = solver_kwargs[name]
        return key

    def _transition_cost(self, sigma_next) -> int:
        if self._sampler in {"heun", "edm_stochastic"}:
            return 1 if float(sigma_next) <= 0.0 else 2
        if self._sampler == "sde_euler_maruyama":
            return 1
        raise ValueError(f"Unknown EDM sampler {self._sampler!r}")

    def schedule_cost(self) -> float:
        if self._sampler in {"heun", "edm_stochastic"}:
            return float(2 * int(self._sampling_steps) - 1)
        if self._sampler == "sde_euler_maruyama":
            return float(int(self._sampling_steps))
        raise ValueError(f"Unknown EDM sampler {self._sampler!r}")

    def format_sampling_label(self) -> str:
        base = (
            f"sampler={self._sampler},N={self._sampling_steps},"
            f"rho={self._rho:g},sigma={self._sigma_min:g}..{self._sigma_max:g}"
        )
        if self._sampler == "edm_stochastic":
            base += f",churn={self._S_churn:g}"
        return base

    def run_solver_baseline_batch(self, *, solver: str, chunk_size: int, n0: int, generator=None, **solver_kwargs):
        if solver != "heun":
            raise ValueError(f"Unknown solver {solver!r}")
        runner = self.with_sampling_config(
            sampler="heun",
            sampling_steps=int(solver_kwargs.get("sampling_steps", self._sampling_steps)),
            sigma_min=float(solver_kwargs.get("sigma_min", self._sigma_min)),
            sigma_max=float(solver_kwargs.get("sigma_max", self._sigma_max)),
            rho=float(solver_kwargs.get("rho", self._rho)),
            S_churn=0.0,
            S_min=self._S_min,
            S_max=self._S_max,
            S_noise=self._S_noise,
        )
        sampling_start = time.perf_counter()
        x = runner.sample_prior(int(chunk_size) * int(n0), generator=generator)
        x = runner.sample_segment(x, runner.start_time, runner.end_time, generator=generator)
        x = runner.postprocess_samples(x)
        sampling_time = time.perf_counter() - sampling_start
        x = x.to(dtype=torch.float32).reshape(int(chunk_size), int(n0), -1).contiguous()
        return [x[i] for i in range(int(chunk_size))], sampling_time

    def observable_values(self, samples, *, comparison_mode: str, observable_config: Any) -> torch.Tensor:
        self._validate_mode(comparison_mode)
        return _rectangle_indicator_grid(samples, observable_config)

    def reference_cache_key(self, comparison_mode: str) -> Mapping[str, Any]:
        self._validate_mode(comparison_mode)
        base = {
            "runner": self.runner_name,
            "comparison_mode": comparison_mode,
            "target_spec": self._target_spec,
        }
        if comparison_mode == "edm_samples":
            base["sampling_config"] = dict(self.sampling_config.values)
        return base

    def reference_legacy_paths(self, comparison_mode: str) -> tuple:
        return ()

    def generate_reference_samples(self, *, comparison_mode: str, num_samples: int, batch_size: int, generator=None):
        self._validate_mode(comparison_mode)
        if comparison_mode == "true_dist":
            raise ValueError("true_dist comparison does not require reference samples")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        batches = []
        remaining = int(num_samples)
        with torch.inference_mode():
            while remaining > 0:
                current = min(int(batch_size), remaining)
                if comparison_mode == "true_samples":
                    generated = edm_target.sample_target_spec(self._target_spec, current, self._device)
                else:
                    x = self.sample_prior(current, generator=generator)
                    generated = self.sample_segment(x, self.start_time, self.end_time, generator=generator)
                    generated = self.postprocess_samples(generated)
                batches.append(generated.cpu())
                remaining -= current
        return torch.cat(batches, dim=0)

    def prepare_comparison_state(self, *, comparison_mode: str, reference_samples=None):
        self._validate_mode(comparison_mode)
        if comparison_mode == "true_dist":
            warm_ks_kernel_for_mode(comparison_mode, self._target_spec)
            return None
        if reference_samples is None:
            raise ValueError(f"comparison_mode={comparison_mode!r} requires reference_samples")
        state = prepare_reference_cdf_state(reference_samples)
        warm_reference_ks_kernel(state)
        return state

    def compute_ks_distance(self, samples, *, comparison_mode: str, comparison_state=None, extra_samples=None) -> float:
        self._validate_mode(comparison_mode)
        samples_np = coerce_samples_np(samples)
        if extra_samples is not None:
            extra_np = coerce_samples_np(extra_samples)
            if extra_np.shape[0] > 0:
                samples_np = np.concatenate([extra_np, samples_np], axis=0)
        if samples_np.shape[0] == 0:
            return float("nan")
        if comparison_mode == "true_dist":
            value, _, _ = compute_target_ks_distance(samples_np, self._target_spec)
        else:
            if comparison_state is None:
                raise ValueError(f"comparison_state is required for comparison_mode={comparison_mode!r}")
            value, _, _ = compute_reference_ks_distance(samples_np, comparison_state)
        return float(value)
