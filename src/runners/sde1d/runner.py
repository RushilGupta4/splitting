from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ks import (
    coerce_samples_np,
    compute_reference_ks_distance,
    prepare_reference_cdf_state,
    warm_reference_ks_kernel,
)
from runners.base import BaseRunner, ComparisonModeSpec, SamplingConfig
from runners.sde1d.cases import SDE_CASES
from runners.sde1d.sampling import SDE1D_SAMPLERS, sample_sde_segment
from runners.splitting import balanced_split_with_run_ids
from utils import validate_split_percentages

SUPPORTED_SAMPLERS = SDE1D_SAMPLERS


class SDE1DRunner(BaseRunner):
    runner_name = "sde_1d"
    case_name = ""
    DEFAULT_SAMPLER = "euler"

    def __init__(
        self,
        *,
        sampler: str = DEFAULT_SAMPLER,
        sampling_steps: int = 128,
        terminal_time: float = 1.0,
        reference_sampler: str = "euler",
        reference_steps: int = 5000,
        device: str = "cpu",
        checkpoint_path: str | None = None,
    ):
        if self.case_name not in SDE_CASES:
            raise ValueError(f"Unknown SDE case {self.case_name!r}")
        if sampler not in SUPPORTED_SAMPLERS:
            raise ValueError(
                f"Unknown SDE sampler {sampler!r}. Available: {SUPPORTED_SAMPLERS}"
            )
        if reference_sampler not in SUPPORTED_SAMPLERS:
            raise ValueError(
                f"Unknown reference sampler {reference_sampler!r}. Available: {SUPPORTED_SAMPLERS}"
            )
        if int(sampling_steps) < 1:
            raise ValueError("sampling_steps must be at least 1")
        if int(reference_steps) < 1:
            raise ValueError("reference_steps must be at least 1")
        if float(terminal_time) <= 0.0:
            raise ValueError("terminal_time must be positive")
        if float(SDE_CASES[self.case_name].initial_variance) < 0.0:
            raise ValueError("initial_variance must be nonnegative")

        self._case = SDE_CASES[self.case_name]
        self._sampler = str(sampler)
        self._sampling_steps = int(sampling_steps)
        self._terminal_time = float(terminal_time)
        self._reference_sampler = str(reference_sampler)
        self._reference_steps = int(reference_steps)
        self._device = str(device)
        self._dtype = torch.float64
        self._checkpoint_path = checkpoint_path or self.default_checkpoint_path()
        self._target_spec = self._case.target_spec_factory(self._terminal_time)

    @classmethod
    def add_train_args(cls, parser) -> None:
        parser.add_argument("--sampling_steps", type=int, default=128)
        parser.add_argument("--terminal_time", type=float, default=1.0)
        parser.add_argument(
            "--sampler", choices=SUPPORTED_SAMPLERS, default=cls.DEFAULT_SAMPLER
        )
        parser.add_argument("--device", type=str, default="cpu")

    @classmethod
    def train_from_args(cls, args) -> None:
        del args
        raise RuntimeError(
            f"{cls.runner_name} is analytic/simulation-only and has no training step"
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | None = None,
        *,
        device: str,
        no_compile: bool = False,
        **kwargs,
    ) -> "SDE1DRunner":
        del no_compile
        return cls(device=device, checkpoint_path=checkpoint_path, **kwargs)

    def with_sampling_config(self, **kwargs) -> "SDE1DRunner":
        allowed = {
            "sampler",
            "sampling_steps",
            "terminal_time",
            "reference_sampler",
            "reference_steps",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.with_sampling_config got unknown keys: {sorted(unknown)}"
            )
        return type(self)(
            sampler=str(kwargs.get("sampler", self._sampler)),
            sampling_steps=int(kwargs.get("sampling_steps", self._sampling_steps)),
            terminal_time=float(kwargs.get("terminal_time", self._terminal_time)),
            reference_sampler=str(
                kwargs.get("reference_sampler", self._reference_sampler)
            ),
            reference_steps=int(kwargs.get("reference_steps", self._reference_steps)),
            device=self._device,
            checkpoint_path=self._checkpoint_path,
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def input_dim(self) -> int:
        return 1

    @property
    def target_spec(self) -> Mapping[str, Any]:
        return self._target_spec

    @property
    def sampling_config(self) -> SamplingConfig:
        return SamplingConfig(
            values={
                "sampler": self._sampler,
                "sampling_steps": int(self._sampling_steps),
                "terminal_time": float(self._terminal_time),
            }
        )

    @property
    def start_time(self) -> Any:
        return 0

    @property
    def end_time(self) -> Any:
        return int(self._sampling_steps)

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    def comparison_modes(self) -> Sequence[ComparisonModeSpec]:
        return (
            ComparisonModeSpec(
                name="true_samples",
                requires_reference_cache=True,
                reference_uses_sampling_config=True,
                description="Two-sample 1D KS against fine-SDE terminal reference samples.",
            ),
        )

    def _validate_mode(self, comparison_mode: str) -> None:
        names = {spec.name for spec in self.comparison_modes()}
        if comparison_mode not in names:
            raise ValueError(
                f"Unknown comparison_mode {comparison_mode!r}. Available: {sorted(names)}"
            )

    def sample_prior(self, num_samples: int, *, generator=None) -> torch.Tensor:
        noise = torch.randn(
            (int(num_samples), 1),
            device=self._device,
            dtype=self._dtype,
            generator=generator,
        )
        return (
            float(self._case.initial_mean)
            + math.sqrt(float(self._case.initial_variance)) * noise
        )

    def _coerce_index(self, value: Any, *, name: str) -> int:
        index = int(round(float(value)))
        if not math.isclose(float(value), float(index), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name} must be an integer grid index, got {value!r}")
        if index < 0 or index > self._sampling_steps:
            raise ValueError(f"{name}={index} outside [0, {self._sampling_steps}]")
        return index

    def sample_segment(
        self,
        x: torch.Tensor,
        start_time: Any,
        end_time: Any,
        *,
        generator=None,
    ) -> torch.Tensor:
        start_idx = self._coerce_index(start_time, name="start_time")
        end_idx = self._coerce_index(end_time, name="end_time")
        if end_idx < start_idx:
            raise ValueError(
                f"SDE segment must move forward, got {start_idx} -> {end_idx}"
            )
        return sample_sde_segment(
            case=self._case,
            sampler=self._sampler,
            x=x,
            start_idx=start_idx,
            end_idx=end_idx,
            sampling_steps=self._sampling_steps,
            terminal_time=self._terminal_time,
            device=self._device,
            dtype=self._dtype,
            generator=generator,
        )

    def postprocess_samples(self, native_samples: torch.Tensor) -> torch.Tensor:
        return native_samples.reshape(-1, 1)

    def resolve_split_percentages(self, split_percentages: Sequence[float]):
        validate_split_percentages(split_percentages)
        remaining_steps = [
            int(round(self._sampling_steps * float(pct))) for pct in split_percentages
        ]
        for idx, steps_left in enumerate(remaining_steps):
            if steps_left <= 0 or steps_left >= self._sampling_steps:
                raise ValueError(
                    "Each split percentage must map to an interior split. "
                    f"Got round({self._sampling_steps} * {split_percentages[idx]}) = {steps_left}."
                )
        split_points = [
            self._sampling_steps - steps_left for steps_left in remaining_steps
        ]
        for idx in range(len(split_points) - 1):
            if split_points[idx] >= split_points[idx + 1]:
                raise ValueError(
                    "Rounded SDE split points must be strictly increasing. "
                    f"Got {split_points[idx]} >= {split_points[idx + 1]}."
                )
        return remaining_steps, split_points

    def segment_cost(self, start_time: Any, end_time: Any) -> float:
        start_idx = self._coerce_index(start_time, name="start_time")
        end_idx = self._coerce_index(end_time, name="end_time")
        if end_idx < start_idx:
            raise ValueError(
                f"SDE segment must move forward, got {start_idx} -> {end_idx}"
            )
        return float(end_idx - start_idx)

    def segment_costs(self, split_points: Sequence[Any]) -> list:
        points = [self._coerce_index(p, name="split_point") for p in split_points]
        starts = [self.start_time] + points
        ends = points + [self.end_time]
        return [self.segment_cost(start, end) for start, end in zip(starts, ends)]

    def expected_cost_per_root(
        self,
        split_points: Sequence[Any],
        split_factors: Sequence[float],
    ) -> float:
        if not split_points:
            return self.segment_cost(self.start_time, self.end_time)
        if len(split_factors) != len(split_points):
            raise ValueError("split_factors must match split_points length")
        starts = [self.start_time] + list(split_points)
        ends = list(split_points) + [self.end_time]
        cost = self.segment_cost(starts[0], ends[0])
        cumulative_split = 1.0
        for idx, split_factor in enumerate(split_factors):
            cumulative_split *= float(split_factor)
            cost += cumulative_split * self.segment_cost(starts[idx + 1], ends[idx + 1])
        return float(cost)

    def run_split_batch(
        self,
        *,
        n0_by_run: Sequence[int],
        split_points: Sequence[Any],
        split_factors_by_run: Sequence[Sequence[float]],
        generator=None,
    ):
        if len(n0_by_run) == 0:
            return [], [], 0.0
        if len(n0_by_run) != len(split_factors_by_run):
            raise ValueError(
                "n0_by_run and split_factors_by_run must have the same length"
            )
        n0_tensor = torch.as_tensor(n0_by_run, device=self._device, dtype=torch.long)
        if torch.any(n0_tensor < 1):
            raise ValueError("all n0 values must be at least 1")

        split_points = [self._coerce_index(p, name="split_point") for p in split_points]
        num_runs = int(n0_tensor.numel())
        run_ids = torch.repeat_interleave(
            torch.arange(num_runs, device=self._device, dtype=torch.long), n0_tensor
        )
        x = self.sample_prior(int(n0_tensor.sum().item()), generator=generator)
        realized_costs = torch.zeros(num_runs, device=self._device, dtype=torch.long)
        sampling_start = time.perf_counter()

        if not split_points:
            realized_costs += n0_tensor * int(
                self.segment_cost(self.start_time, self.end_time)
            )
            x = self.sample_segment(
                x, self.start_time, self.end_time, generator=generator
            )
        else:
            realized_costs += n0_tensor * int(
                self.segment_cost(self.start_time, split_points[0])
            )
            x = self.sample_segment(
                x, self.start_time, split_points[0], generator=generator
            )
            split_factors_tensor = torch.as_tensor(
                np.asarray(split_factors_by_run, dtype=float),
                device=self._device,
                dtype=x.dtype,
            )
            if split_factors_tensor.shape != (num_runs, len(split_points)):
                raise ValueError("split_factors_by_run has incompatible shape")
            for idx, split_point in enumerate(split_points):
                x, run_ids = balanced_split_with_run_ids(
                    x, run_ids, split_factors_tensor[:, idx], generator=generator
                )
                end_t = (
                    split_points[idx + 1]
                    if idx + 1 < len(split_points)
                    else self.end_time
                )
                counts = torch.bincount(run_ids, minlength=num_runs)
                realized_costs += counts * int(self.segment_cost(split_point, end_t))
                x = self.sample_segment(x, split_point, end_t, generator=generator)

        x = self.postprocess_samples(x)
        sampling_time = time.perf_counter() - sampling_start
        samples_by_run = [x[run_ids == run_idx] for run_idx in range(num_runs)]
        return samples_by_run, realized_costs.detach().cpu().tolist(), sampling_time

    def solver_names(self) -> Sequence[str]:
        return SUPPORTED_SAMPLERS

    def solver_cost(self, solver: str, **solver_kwargs) -> float:
        if solver not in SUPPORTED_SAMPLERS:
            raise ValueError(f"Unknown solver {solver!r}")
        return float(int(solver_kwargs.get("sampling_steps", self._sampling_steps)))

    def solver_cache_key(self, solver: str, **solver_kwargs) -> Mapping[str, Any]:
        if solver not in SUPPORTED_SAMPLERS:
            raise ValueError(f"Unknown solver {solver!r}")
        key: dict[str, Any] = {}
        if "sampling_steps" in solver_kwargs:
            key["sampling_steps"] = int(solver_kwargs["sampling_steps"])
        for name in sorted(set(solver_kwargs) - set(key)):
            key[name] = solver_kwargs[name]
        return key

    def schedule_cost(self) -> float:
        return float(self._sampling_steps)

    def run_solver_baseline_batch(
        self,
        *,
        solver: str,
        chunk_size: int,
        n0: int,
        generator=None,
        **solver_kwargs,
    ):
        if solver not in SUPPORTED_SAMPLERS:
            raise ValueError(f"Unknown solver {solver!r}")
        runner = self.with_sampling_config(
            sampler=solver,
            sampling_steps=int(
                solver_kwargs.get("sampling_steps", self._sampling_steps)
            ),
            terminal_time=float(
                solver_kwargs.get("terminal_time", self._terminal_time)
            ),
        )
        sampling_start = time.perf_counter()
        x = runner.sample_prior(int(chunk_size) * int(n0), generator=generator)
        x = runner.sample_segment(
            x, runner.start_time, runner.end_time, generator=generator
        )
        x = runner.postprocess_samples(x)
        sampling_time = time.perf_counter() - sampling_start
        x = x.to(dtype=torch.float32).reshape(int(chunk_size), int(n0), 1).contiguous()
        return [x[idx] for idx in range(int(chunk_size))], sampling_time

    def observable_values(
        self,
        samples,
        *,
        comparison_mode: str,
        x_grid: Any,
    ) -> torch.Tensor:
        self._validate_mode(comparison_mode)
        values = (
            samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
        )
        values = values.to(device=self._device, dtype=self._dtype).reshape(-1, 1)
        thresholds = torch.as_tensor(
            x_grid,
            device=values.device,
            dtype=values.dtype,
        ).reshape(1, -1)
        return (values[:, 0:1] <= thresholds).to(values.dtype)

    def reference_cache_key(self, comparison_mode: str) -> Mapping[str, Any]:
        self._validate_mode(comparison_mode)
        key: dict[str, Any] = {
            "runner": self.runner_name,
            "comparison_mode": comparison_mode,
            "target_spec": self._target_spec,
            "reference_sampler": self._reference_sampler,
            "reference_steps": int(self._reference_steps),
            "terminal_time": float(self._terminal_time),
            "path_clipping": None,
        }
        return key

    def generate_reference_samples(
        self,
        *,
        comparison_mode: str,
        num_samples: int,
        batch_size: int,
        generator=None,
    ) -> torch.Tensor:
        self._validate_mode(comparison_mode)
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")

        batches: list[torch.Tensor] = []
        remaining = int(num_samples)
        runner = self.with_sampling_config(
            sampler=self._reference_sampler,
            sampling_steps=self._reference_steps,
        )
        with torch.inference_mode():
            while remaining > 0:
                current = min(int(batch_size), remaining)
                x = runner.sample_prior(current, generator=generator)
                samples = runner.sample_segment(
                    x, runner.start_time, runner.end_time, generator=generator
                )
                samples = runner.postprocess_samples(samples)
                batches.append(samples.cpu().to(dtype=torch.float32))
                remaining -= current
        return torch.cat(batches, dim=0)

    def prepare_comparison_state(
        self,
        *,
        comparison_mode: str,
        reference_samples=None,
    ):
        self._validate_mode(comparison_mode)
        if reference_samples is None:
            raise ValueError(
                f"comparison_mode={comparison_mode!r} requires reference_samples"
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
        if comparison_state is None:
            raise ValueError(
                f"comparison_state is required for comparison_mode={comparison_mode!r}"
            )
        value, _, _ = compute_reference_ks_distance(samples_np, comparison_state)
        return float(value)


class SimpleOURunner(SDE1DRunner):
    """1D SDE runner for Simple OU."""

    runner_name = "simple_ou"
    case_name = "simple_ou"
    config_module = "runners.sde1d.simple_ou.configs"


class CEVSecurityPriceRunner(SDE1DRunner):
    """1D SDE runner for the Duffie-Glynn CEV security-price model."""

    runner_name = "cev_security_price"
    case_name = "cev_security_price"
    config_module = "runners.sde1d.cev_security_price.configs"


class SmoothThresholdAutoregressionRunner(SDE1DRunner):
    """1D SDE runner for a smooth threshold autoregression model."""

    runner_name = "smooth_threshold_autoregression"
    case_name = "smooth_threshold_autoregression"
    config_module = "runners.sde1d.smooth_threshold_autoregression.configs"


SDE1D_RUNNER_CLASSES = {
    SimpleOURunner.runner_name: SimpleOURunner,
    CEVSecurityPriceRunner.runner_name: CEVSecurityPriceRunner,
    SmoothThresholdAutoregressionRunner.runner_name: SmoothThresholdAutoregressionRunner,
}
