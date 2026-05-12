"""DDPM/DDIM runner for the 2D Gaussian-mixture target."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diffusion import (
    DDIM,
    expected_cost_per_root as _diffusion_expected_cost_per_root,
    resolve_split_percentages as _diffusion_resolve_split_percentages,
    run_probabilistic_inference_batch,
    run_solver_baseline_batch,
    sample_segment as _diffusion_sample_segment,
    segment_costs as _diffusion_segment_costs,
)
from ks import (
    coerce_samples_np,
    compute_reference_ks_distance,
    compute_target_ks_distance,
    prepare_reference_cdf_state,
    warm_ks_kernel_for_mode,
    warm_reference_ks_kernel,
)
from model_io import load_model_and_stats, model_input_dim
from reference_cache import (
    reference_samples_path as _legacy_ddpm_samples_path,
    true_reference_samples_path as _legacy_true_samples_path,
)
from runners.base import (
    BaseRunner,
    ComparisonModeSpec,
    SamplingConfig,
)
from runners.ddpm_gmm2d import target as ddpm_target

SUPPORTED_SOLVERS = ("ddim", "dpmpp_2m")

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
        name="ddpm_samples",
        requires_reference_cache=True,
        reference_uses_sampling_config=True,
        description="Two-sample lower-orthant KS against generated DDPM/DDIM samples.",
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


class DDPMGMM2DRunner(BaseRunner):
    runner_name = "ddpm_gmm2d"
    DEFAULT_SOLVER = "ddim"

    def __init__(
        self,
        *,
        model,
        target_spec: Mapping[str, Any],
        data_mean: torch.Tensor,
        data_std: torch.Tensor,
        T: int,
        sampling_steps: int,
        eta: float,
        sampler: str = "ddim",
        device: str,
        checkpoint_path: str | None = None,
    ):
        sampler = str(sampler)
        if sampler != "ddim":
            raise ValueError("DDPMGMM2DRunner only supports sampler='ddim'")
        self._model = model
        self._target_spec = dict(target_spec)
        self._data_mean = data_mean
        self._data_std = data_std
        self._T = int(T)
        self._sampling_steps = int(sampling_steps)
        self._eta = float(eta)
        self._sampler = sampler
        self._device = str(device)
        self._checkpoint_path = checkpoint_path or type(self).default_checkpoint_path()
        self._ddim = DDIM(
            T=self._T,
            device=self._device,
            eta=self._eta,
            sampling_steps=self._sampling_steps,
        )

    # ------------------------------------------------------------------
    # CLI hooks
    # ------------------------------------------------------------------

    @classmethod
    def add_train_args(cls, parser) -> None:
        parser.add_argument("--T", type=int, default=1000)
        parser.add_argument("--num_samples", type=int, default=100_000)
        parser.add_argument("--batch_size", type=int, default=10_000)
        parser.add_argument("--epochs", type=int, default=100)
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
        from runners.ddpm_gmm2d.train import train as _train

        _train(args)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | None = None,
        *,
        device: str,
        no_compile: bool = False,
        T: int | None = None,
        sampling_steps: int | None = None,
        eta: float = 1.0,
        sampler: str = "ddim",
        **kwargs,
    ) -> "DDPMGMM2DRunner":
        path = checkpoint_path or cls.default_checkpoint_path()
        model, target_spec, data_mean, data_std = load_model_and_stats(
            path, device, no_compile=no_compile
        )
        if T is None:
            T = 1000
        if sampling_steps is None:
            sampling_steps = int(T)
        return cls(
            model=model,
            target_spec=target_spec,
            data_mean=data_mean,
            data_std=data_std,
            T=int(T),
            sampling_steps=int(sampling_steps),
            eta=float(eta),
            sampler=sampler,
            device=device,
            checkpoint_path=path,
        )

    def with_sampling_config(self, **kwargs) -> "DDPMGMM2DRunner":
        allowed = {"sampler", "T", "sampling_steps", "eta"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.with_sampling_config got unknown keys: "
                f"{sorted(unknown)}"
            )
        T = int(kwargs.get("T", self._T))
        sampling_steps = int(kwargs.get("sampling_steps", self._sampling_steps))
        eta = float(kwargs.get("eta", self._eta))
        sampler = str(kwargs.get("sampler", self._sampler))
        return DDPMGMM2DRunner(
            model=self._model,
            target_spec=self._target_spec,
            data_mean=self._data_mean,
            data_std=self._data_std,
            T=T,
            sampling_steps=sampling_steps,
            eta=eta,
            sampler=sampler,
            device=self._device,
            checkpoint_path=self._checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def input_dim(self) -> int:
        return int(model_input_dim(self._model))

    @property
    def target_spec(self) -> Mapping[str, Any]:
        return self._target_spec

    @property
    def sampling_config(self) -> SamplingConfig:
        return SamplingConfig(
            values={
                "sampler": str(self._sampler),
                "T": int(self._T),
                "sampling_steps": int(self._sampling_steps),
                "eta": float(self._eta),
            }
        )

    @property
    def start_time(self) -> Any:
        return int(self._T)

    @property
    def end_time(self) -> Any:
        return 0

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @property
    def model(self):
        return self._model

    @property
    def data_mean(self):
        return self._data_mean

    @property
    def data_std(self):
        return self._data_std

    @property
    def ddim(self) -> DDIM:
        return self._ddim

    # ------------------------------------------------------------------
    # Modes and KS distance
    # ------------------------------------------------------------------

    def comparison_modes(self) -> Sequence[ComparisonModeSpec]:
        return COMPARISON_MODES

    def _validate_mode(self, comparison_mode: str) -> None:
        if comparison_mode not in _COMPARISON_MODE_NAMES:
            raise ValueError(
                f"Unknown comparison_mode '{comparison_mode}'. "
                f"Available: {sorted(_COMPARISON_MODE_NAMES)}"
            )

    # ------------------------------------------------------------------
    # Sampling primitives
    # ------------------------------------------------------------------

    def sample_prior(self, num_samples: int, *, generator=None) -> torch.Tensor:
        return torch.randn(
            int(num_samples),
            self.input_dim,
            device=self._device,
            generator=generator,
        )

    def sample_segment(
        self,
        x: torch.Tensor,
        start_time: Any,
        end_time: Any,
        *,
        generator=None,
    ) -> torch.Tensor:
        return _diffusion_sample_segment(
            self._ddim, self._model, x, int(start_time), int(end_time), generator=generator
        )

    def postprocess_samples(self, native_samples: torch.Tensor) -> torch.Tensor:
        return ddpm_target.denormalize(native_samples, self._data_mean, self._data_std)

    # ------------------------------------------------------------------
    # Splits and costs
    # ------------------------------------------------------------------

    def resolve_split_percentages(self, split_percentages: Sequence[float]):
        return _diffusion_resolve_split_percentages(self._ddim, split_percentages)

    def segment_cost(self, start_time: Any, end_time: Any) -> float:
        return float(self._ddim.segment_cost(int(start_time), int(end_time)))

    def segment_costs(self, split_points: Sequence[Any]) -> list:
        return _diffusion_segment_costs(self._ddim, [int(p) for p in split_points])

    def expected_cost_per_root(
        self,
        split_points: Sequence[Any],
        split_factors: Sequence[float],
    ) -> float:
        return float(
            _diffusion_expected_cost_per_root(
                self._ddim, [int(p) for p in split_points], [float(f) for f in split_factors]
            )
        )

    def run_split_batch(
        self,
        *,
        n0_by_run: Sequence[int],
        split_points: Sequence[Any],
        split_factors_by_run: Sequence[Sequence[float]],
        generator=None,
    ):
        return run_probabilistic_inference_batch(
            model=self._model,
            ddim=self._ddim,
            n0_by_run=n0_by_run,
            split_points=[int(p) for p in split_points],
            split_factors_by_run=split_factors_by_run,
            data_mean=self._data_mean,
            data_std=self._data_std,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # Baseline solvers
    # ------------------------------------------------------------------

    def solver_names(self) -> Sequence[str]:
        return SUPPORTED_SOLVERS

    def solver_cost(self, solver: str, **solver_kwargs) -> float:
        if solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Unknown solver '{solver}'")
        sampling_steps = solver_kwargs.get("sampling_steps", self._sampling_steps)
        return float(int(sampling_steps))

    def solver_cache_key(self, solver: str, **solver_kwargs) -> Mapping[str, Any]:
        if solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Unknown solver '{solver}'")
        key: dict[str, Any] = {}
        if "sampling_steps" in solver_kwargs:
            key["sampling_steps"] = int(solver_kwargs["sampling_steps"])
        if solver == "ddim" and float(solver_kwargs.get("eta", 0.0)) != 0.0:
            key["eta"] = float(solver_kwargs["eta"])
        for name in sorted(set(solver_kwargs) - set(key) - {"eta"}):
            key[name] = solver_kwargs[name]
        return key

    def schedule_cost(self) -> float:
        return float(int(self._sampling_steps))

    def format_sampling_label(self) -> str:
        return f"sampler=ddim,steps={self._sampling_steps},eta={self._eta:g}"

    def run_solver_baseline_batch(
        self,
        *,
        solver: str,
        chunk_size: int,
        n0: int,
        generator=None,
        **solver_kwargs,
    ):
        T = int(solver_kwargs.get("T", self._T))
        sampling_steps = int(solver_kwargs.get("sampling_steps", self._sampling_steps))
        eta = float(solver_kwargs.get("eta", 0.0))
        return run_solver_baseline_batch(
            model=self._model,
            data_mean=self._data_mean,
            data_std=self._data_std,
            solver=solver,
            chunk_size=int(chunk_size),
            n0=int(n0),
            T=T,
            sampling_steps=sampling_steps,
            eta=eta,
            device=self._device,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def observable_values(
        self,
        samples: torch.Tensor,
        *,
        comparison_mode: str,
        observable_config: Any,
    ) -> torch.Tensor:
        self._validate_mode(comparison_mode)
        return _rectangle_indicator_grid(samples, observable_config)

    # ------------------------------------------------------------------
    # Reference cache & comparison state
    # ------------------------------------------------------------------

    def reference_cache_key(self, comparison_mode: str) -> Mapping[str, Any]:
        self._validate_mode(comparison_mode)
        if comparison_mode == "true_dist":
            return {
                "runner": self.runner_name,
                "comparison_mode": comparison_mode,
                "target_spec": self._target_spec,
            }
        if comparison_mode == "true_samples":
            return {
                "runner": self.runner_name,
                "comparison_mode": comparison_mode,
                "target_spec": self._target_spec,
            }
        return {
            "runner": self.runner_name,
            "comparison_mode": comparison_mode,
            "target_spec": self._target_spec,
            "sampling_config": dict(self.sampling_config.values),
        }

    def reference_legacy_paths(self, comparison_mode: str) -> tuple:
        """Optional legacy paths to check when loading reference samples."""
        if self._checkpoint_path is None:
            return ()
        if comparison_mode == "true_samples":
            return (_legacy_true_samples_path(self._checkpoint_path),)
        if comparison_mode == "ddpm_samples":
            return (
                _legacy_ddpm_samples_path(
                    self._checkpoint_path, self._sampling_steps, self._eta
                ),
            )
        return ()

    def generate_reference_samples(
        self,
        *,
        comparison_mode: str,
        num_samples: int,
        batch_size: int,
        generator=None,
    ) -> torch.Tensor:
        self._validate_mode(comparison_mode)
        if comparison_mode == "true_dist":
            raise ValueError("true_dist comparison does not require reference samples")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        batches: list[torch.Tensor] = []
        remaining = int(num_samples)

        if comparison_mode == "true_samples":
            with torch.inference_mode():
                while remaining > 0:
                    current = min(batch_size, remaining)
                    batches.append(
                        ddpm_target.sample_target_spec(
                            self._target_spec, current, self._device
                        ).cpu()
                    )
                    remaining -= current
            return torch.cat(batches, dim=0)

        # ddpm_samples
        with torch.inference_mode():
            while remaining > 0:
                current = min(batch_size, remaining)
                x_T = torch.randn(
                    current, self.input_dim, device=self._device, generator=generator
                )
                generated = self._ddim.sample_loop(self._model, x_T, self._ddim.T, 0)
                generated = self.postprocess_samples(generated)
                batches.append(generated.cpu())
                remaining -= current
        return torch.cat(batches, dim=0)

    def prepare_comparison_state(
        self,
        *,
        comparison_mode: str,
        reference_samples=None,
    ):
        self._validate_mode(comparison_mode)
        if comparison_mode == "true_dist":
            warm_ks_kernel_for_mode(comparison_mode, self._target_spec)
            return None
        if reference_samples is None:
            raise ValueError(
                f"comparison_mode='{comparison_mode}' requires reference_samples"
            )
        state = prepare_reference_cdf_state(reference_samples)
        warm_reference_ks_kernel(state)
        return state

    def compute_ks_distance(
        self,
        samples,
        *,
        comparison_mode: str,
        comparison_state=None,
        extra_samples=None,
    ) -> float:
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
                raise ValueError(
                    f"comparison_state is required for comparison_mode='{comparison_mode}'"
                )
            value, _, _ = compute_reference_ks_distance(samples_np, comparison_state)
        return float(value)
