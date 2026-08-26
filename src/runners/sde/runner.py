from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from runners.base import BaseRunner, ComparisonModeSpec, SamplingConfig
from runners.sde.cases import SDE_CASES
from runners.sde.sampling import SDE_SAMPLERS, sample_sde_segment
from runners.splitting import (
    append_by_counts as _append_by_counts,
    cat_parts_by_run as _cat_parts_by_run,
    run_full_trajectory_batch,
    run_split_trajectory_batch,
    trajectory_expected_cost_per_root,
    trajectory_segment_costs,
)
from utils import validate_split_percentages

SUPPORTED_SAMPLERS = SDE_SAMPLERS


class SDERunner(BaseRunner):
    runner_name = "sde"
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
        dimension: int = 1,
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
        dimension = int(dimension)
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        case = SDE_CASES[self.case_name]
        if dimension > 1 and (sampler == "milstein" or reference_sampler == "milstein"):
            raise ValueError("Milstein sampler is only supported for dimension=1 for now")
        if str(case.diffusion_structure) != "diagonal" and (
            sampler == "milstein" or reference_sampler == "milstein"
        ):
            raise ValueError("Milstein sampler requires diagonal diffusion")

        self._case = case
        self._sampler = str(sampler)
        self._sampling_steps = int(sampling_steps)
        self._terminal_time = float(terminal_time)
        self._reference_sampler = str(reference_sampler)
        self._reference_steps = int(reference_steps)
        self._dimension = int(dimension)
        self._device = str(device)
        self._dtype = torch.float32
        self._checkpoint_path = checkpoint_path or self.default_checkpoint_path()
        self._target_spec = self._case.target_spec_factory(
            self._terminal_time,
            self._dimension,
        )

    @classmethod
    def add_train_args(cls, parser) -> None:
        parser.add_argument("--sampling_steps", type=int, default=128)
        parser.add_argument("--terminal_time", type=float, default=1.0)
        parser.add_argument(
            "--sampler", choices=SUPPORTED_SAMPLERS, default=cls.DEFAULT_SAMPLER
        )
        parser.add_argument("--dimension", type=int, default=1)
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
    ) -> "SDERunner":
        del no_compile
        return cls(device=device, checkpoint_path=checkpoint_path, **kwargs)

    def with_sampling_config(self, **kwargs) -> "SDERunner":
        allowed = {
            "sampler",
            "sampling_steps",
            "terminal_time",
            "reference_sampler",
            "reference_steps",
        }
        runner_defaults = {"dimension"} & set(kwargs)
        if runner_defaults:
            raise ValueError(
                f"{type(self).__name__}.with_sampling_config cannot change runner "
                f"defaults: {sorted(runner_defaults)}"
            )
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
            dimension=self._dimension,
            device=self._device,
            checkpoint_path=self._checkpoint_path,
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def input_dim(self) -> int:
        return int(self._dimension)

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
                reference_uses_sampling_config=False,
                description="Two-sample KS against fine-SDE terminal reference samples.",
            ),
        )

    def _validate_mode(self, comparison_mode: str) -> None:
        self.comparison_mode_spec(comparison_mode)

    def sample_prior(self, num_samples: int, *, generator=None) -> torch.Tensor:
        if self._case.initial_sampler is not None:
            values = self._case.initial_sampler(
                num_samples=int(num_samples),
                dimension=self._dimension,
                device=self._device,
                dtype=self._dtype,
                generator=generator,
            )
            return self._coerce_sample_tensor(values, name="initial_samples")
        noise = torch.randn(
            (int(num_samples), self._dimension),
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
        progress_callback=None,
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
            dimension=self._dimension,
            device=self._device,
            dtype=self._dtype,
            generator=generator,
            progress_callback=progress_callback,
        )

    def _coerce_sample_tensor(self, samples, *, name: str) -> torch.Tensor:
        values = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
        if values.ndim == 1:
            if self._dimension != 1:
                raise ValueError(
                    f"{name} must have shape [N, {self._dimension}], got {tuple(values.shape)}"
                )
            values = values.reshape(-1, 1)
        elif values.ndim > 2:
            values = values.reshape(-1, values.shape[-1])
        if values.ndim != 2 or int(values.shape[1]) != self._dimension:
            raise ValueError(
                f"{name} must have shape [N, {self._dimension}], got {tuple(values.shape)}"
            )
        return values.to(device=self._device, dtype=self._dtype).contiguous()

    def postprocess_samples(self, native_samples: torch.Tensor) -> torch.Tensor:
        values = self._coerce_sample_tensor(native_samples, name="native_samples")
        if self._case.terminal_transform is not None:
            values = self._case.terminal_transform(values)
        return self._coerce_sample_tensor(values, name="terminal_samples")

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
        return trajectory_segment_costs(self, points)

    def expected_cost_per_root(
        self,
        split_points: Sequence[Any],
        split_factors: Sequence[float],
    ) -> float:
        points = [self._coerce_index(p, name="split_point") for p in split_points]
        return trajectory_expected_cost_per_root(self, points, split_factors)

    def run_split_batch(
        self,
        *,
        n0_by_run: Sequence[int],
        split_points: Sequence[Any],
        split_factors_by_run: Sequence[Sequence[float]],
        generator=None,
        max_sampling_batch_size=None,
    ):
        split_points = [self._coerce_index(p, name="split_point") for p in split_points]
        return run_split_trajectory_batch(
            self,
            n0_by_run=n0_by_run,
            split_points=split_points,
            split_factors_by_run=split_factors_by_run,
            generator=generator,
            max_sampling_batch_size=max_sampling_batch_size,
        )

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
        max_sampling_batch_size=None,
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
        return run_full_trajectory_batch(
            runner,
            chunk_size=chunk_size,
            n0=n0,
            generator=generator,
            max_sampling_batch_size=max_sampling_batch_size,
        )

    def normalize_reference_generation_config(
        self,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        self._validate_mode(comparison_mode)
        if reference_generation_config is None:
            raise ValueError(
                f"reference_generation_config is required for comparison_mode={comparison_mode!r}"
            )
        cfg = dict(reference_generation_config)
        required = {"method", "sampler", "sampling_steps", "terminal_time"}
        missing = sorted(required - set(cfg))
        if missing:
            raise ValueError(
                f"reference_generation_config missing required keys: {missing}"
            )
        unknown = set(cfg) - required
        if unknown:
            raise ValueError(
                f"Unknown reference_generation_config keys: {sorted(unknown)}"
            )
        if str(cfg["method"]) != "sde_terminal_samples":
            raise ValueError(
                "SDE reference_generation_config must set method='sde_terminal_samples'"
            )
        sampler = str(cfg["sampler"])
        if sampler not in SUPPORTED_SAMPLERS:
            raise ValueError(
                f"Unknown SDE reference sampler {sampler!r}. Available: {SUPPORTED_SAMPLERS}"
            )
        sampling_steps = int(cfg["sampling_steps"])
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be at least 1")
        terminal_time = float(cfg["terminal_time"])
        if terminal_time <= 0.0:
            raise ValueError("terminal_time must be positive")
        if terminal_time != float(self._terminal_time):
            raise ValueError(
                "reference_generation_config terminal_time must match runner terminal_time "
                f"({self._terminal_time})"
            )
        self.with_sampling_config(
            sampler=sampler,
            sampling_steps=sampling_steps,
            terminal_time=terminal_time,
        )
        return {
            "method": "sde_terminal_samples",
            "sampler": sampler,
            "sampling_steps": sampling_steps,
            "terminal_time": terminal_time,
        }

    def reference_cache_key(
        self,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self._validate_mode(comparison_mode)
        ref_cfg = self.normalize_reference_generation_config(
            comparison_mode, reference_generation_config
        )
        key: dict[str, Any] = {
            "runner": self.runner_name,
            "comparison_mode": comparison_mode,
            "target_spec": self._target_spec,
            "reference_generation_config": dict(ref_cfg),
        }
        return key

    def generate_reference_samples(
        self,
        *,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any],
        num_samples: int,
        batch_size: int,
        generator=None,
        progress=None,
    ) -> torch.Tensor:
        self._validate_mode(comparison_mode)
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        ref_cfg = self.normalize_reference_generation_config(
            comparison_mode, reference_generation_config
        )

        batches: list[torch.Tensor] = []
        remaining = int(num_samples)
        runner = self.with_sampling_config(
            sampler=ref_cfg["sampler"],
            sampling_steps=ref_cfg["sampling_steps"],
            terminal_time=ref_cfg["terminal_time"],
        )
        with torch.inference_mode():
            while remaining > 0:
                current = min(int(batch_size), remaining)
                x = runner.sample_prior(current, generator=generator)
                if progress is not None:
                    progress["start_steps"](
                        runner._coerce_index(runner.end_time, name="end_time")
                        - runner._coerce_index(runner.start_time, name="start_time")
                    )
                try:
                    samples = runner.sample_segment(
                        x,
                        runner.start_time,
                        runner.end_time,
                        generator=generator,
                        progress_callback=None if progress is None else progress["step"],
                    )
                finally:
                    if progress is not None:
                        progress["finish_steps"]()
                samples = runner.postprocess_samples(samples)
                batches.append(samples.cpu().to(dtype=torch.float32))
                remaining -= current
                if progress is not None:
                    progress["batch"](1)
        return torch.cat(batches, dim=0)

    def prepare_comparison_state(
        self,
        *,
        comparison_mode: str,
        reference_samples=None,
        metric_params=None,
    ):
        return super().prepare_comparison_state(
            comparison_mode=comparison_mode,
            reference_samples=reference_samples,
            metric_params=metric_params,
        )

    def compute_ks_distance(
        self,
        parts,
        *,
        comparison_mode: str,
        comparison_state=None,
        part_weights=None,
    ) -> float:
        return super().compute_ks_distance(
            parts,
            comparison_mode=comparison_mode,
            comparison_state=comparison_state,
            part_weights=part_weights,
        )


class SimpleOURunner(SDERunner):
    """SDE runner for Simple OU."""

    runner_name = "simple_ou"
    case_name = "simple_ou"
    config_module = "runners.sde.simple_ou.configs"


class CoupledDoubleWellLangevinRunner(SDERunner):
    """SDE runner for coupled double-well overdamped Langevin dynamics."""

    runner_name = "coupled_double_well_langevin"
    case_name = "coupled_double_well_langevin"
    config_module = "runners.sde.coupled_double_well_langevin.configs"


SDE_RUNNER_CLASSES = {
    SimpleOURunner.runner_name: SimpleOURunner,
    CoupledDoubleWellLangevinRunner.runner_name: CoupledDoubleWellLangevinRunner,
}
