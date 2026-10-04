from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from runners.base import BaseRunner, ComparisonModeSpec
from runners.sde.coupled_double_well_langevin.spec import (
    SPEC as COUPLED_DOUBLE_WELL_LANGEVIN_SPEC,
)
from runners.sde.sampling import SDE_SAMPLERS, sample_sde_segment
from runners.sde.simple_ou.spec import SPEC as SIMPLE_OU_SPEC
from runners.sde.spec import SDECase
from runners.splitting import run_full_trajectory_batch
from utils import validate_split_percentages

SUPPORTED_SAMPLERS = SDE_SAMPLERS


class SDERunner(BaseRunner):
    runner_name = "sde"
    case: SDECase
    supported_solvers = SUPPORTED_SAMPLERS
    comparison_mode_specs = (
        ComparisonModeSpec(name="true_samples", requires_reference_cache=True),
    )

    def __init__(
        self,
        *,
        sampler: str = "euler",
        sampling_steps: int = 128,
        terminal_time: float = 1.0,
        dimension: int = 1,
        device: str = "cpu",
        checkpoint_path: str | None = None,
    ):
        if sampler not in SUPPORTED_SAMPLERS:
            raise ValueError(
                f"Unknown SDE sampler {sampler!r}. Available: {SUPPORTED_SAMPLERS}"
            )
        if int(sampling_steps) < 1:
            raise ValueError("sampling_steps must be at least 1")
        if float(terminal_time) <= 0.0:
            raise ValueError("terminal_time must be positive")
        if float(self.case.initial_variance) < 0.0:
            raise ValueError("initial_variance must be nonnegative")
        if int(dimension) < 1:
            raise ValueError("dimension must be at least 1")

        self._sampler = str(sampler)
        self._sampling_steps = int(sampling_steps)
        self._terminal_time = float(terminal_time)
        self._dimension = int(dimension)
        self._device = str(device)
        self._dtype = torch.float32
        self._checkpoint_path = checkpoint_path or self.default_checkpoint_path()
        self._target_spec = self.case.target_spec_factory(
            self._terminal_time,
            self._dimension,
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
        unknown = set(kwargs) - {"sampler", "sampling_steps", "terminal_time"}
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.with_sampling_config got unknown keys: {sorted(unknown)}"
            )
        return type(self)(
            sampler=str(kwargs.get("sampler", self._sampler)),
            sampling_steps=int(kwargs.get("sampling_steps", self._sampling_steps)),
            terminal_time=float(kwargs.get("terminal_time", self._terminal_time)),
            dimension=self._dimension,
            device=self._device,
            checkpoint_path=self._checkpoint_path,
        )

    @property
    def input_dim(self) -> int:
        return int(self._dimension)

    def sampling_cache_key(self) -> Mapping[str, Any]:
        return {
            "sampler": self._sampler,
            "sampling_steps": int(self._sampling_steps),
            "terminal_time": float(self._terminal_time),
        }

    @property
    def start_time(self) -> Any:
        return 0

    @property
    def end_time(self) -> Any:
        return int(self._sampling_steps)

    def sample_prior(self, num_samples: int, *, generator=None) -> torch.Tensor:
        noise = torch.randn(
            (int(num_samples), self._dimension),
            device=self._device,
            dtype=self._dtype,
            generator=generator,
        )
        return (
            float(self.case.initial_mean)
            + math.sqrt(float(self.case.initial_variance)) * noise
        )

    def _coerce_index(self, value: Any, *, name: str) -> int:
        index = int(round(float(value)))
        if not math.isclose(float(value), float(index), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name} must be an integer grid index, got {value!r}")
        if index < 0 or index > self._sampling_steps:
            raise ValueError(f"{name}={index} outside [0, {self._sampling_steps}]")
        return index

    def _split_point(self, point):
        return self._coerce_index(point, name="split_point")

    def _segment_indices(self, start_time: Any, end_time: Any) -> tuple[int, int]:
        start_idx = self._coerce_index(start_time, name="start_time")
        end_idx = self._coerce_index(end_time, name="end_time")
        if end_idx < start_idx:
            raise ValueError(
                f"SDE segment must move forward, got {start_idx} -> {end_idx}"
            )
        return start_idx, end_idx

    def sample_segment(
        self,
        x: torch.Tensor,
        start_time: Any,
        end_time: Any,
        *,
        generator=None,
        progress_callback=None,
    ) -> torch.Tensor:
        start_idx, end_idx = self._segment_indices(start_time, end_time)
        return sample_sde_segment(
            self.case,
            self._coerce_sample_tensor(x, name="SDE state"),
            start_idx,
            end_idx,
            sampling_steps=self._sampling_steps,
            terminal_time=self._terminal_time,
            generator=generator,
            progress_callback=progress_callback,
        )

    def _coerce_sample_tensor(self, samples, *, name: str) -> torch.Tensor:
        values = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
        if values.ndim == 1 and self._dimension == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or int(values.shape[1]) != self._dimension:
            raise ValueError(
                f"{name} must have shape [N, {self._dimension}], got {tuple(values.shape)}"
            )
        return values.to(device=self._device, dtype=self._dtype).contiguous()

    def postprocess_samples(self, native_samples: torch.Tensor) -> torch.Tensor:
        return self._coerce_sample_tensor(native_samples, name="terminal_samples")

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
        start_idx, end_idx = self._segment_indices(start_time, end_time)
        return float(end_idx - start_idx)

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
        self.comparison_mode_spec(comparison_mode)
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
        normalized = {
            "method": "sde_terminal_samples",
            "sampler": str(cfg["sampler"]),
            "sampling_steps": int(cfg["sampling_steps"]),
            "terminal_time": float(cfg["terminal_time"]),
        }
        self.with_sampling_config(
            sampler=normalized["sampler"],
            sampling_steps=normalized["sampling_steps"],
            terminal_time=normalized["terminal_time"],
        )
        if normalized["terminal_time"] != float(self._terminal_time):
            raise ValueError(
                "reference_generation_config terminal_time must match runner terminal_time "
                f"({self._terminal_time})"
            )
        return normalized

    def reference_cache_key(
        self,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        ref_cfg = self.normalize_reference_generation_config(
            comparison_mode, reference_generation_config
        )
        return {
            "runner": self.runner_name,
            "comparison_mode": comparison_mode,
            "target_spec": self._target_spec,
            "reference_generation_config": dict(ref_cfg),
        }

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
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        ref_cfg = self.normalize_reference_generation_config(
            comparison_mode, reference_generation_config
        )
        runner = self.with_sampling_config(
            sampler=ref_cfg["sampler"],
            sampling_steps=ref_cfg["sampling_steps"],
            terminal_time=ref_cfg["terminal_time"],
        )
        return runner._sample_reference(
            num_samples,
            batch_size,
            sample_dim=runner._dimension,
            num_steps=runner._sampling_steps,
            generator=generator,
            progress=progress,
        )


class SimpleOURunner(SDERunner):
    runner_name = "simple_ou"
    case = SIMPLE_OU_SPEC
    config_module = "runners.sde.simple_ou.configs"


class CoupledDoubleWellLangevinRunner(SDERunner):
    runner_name = "coupled_double_well_langevin"
    case = COUPLED_DOUBLE_WELL_LANGEVIN_SPEC
    config_module = "runners.sde.coupled_double_well_langevin.configs"
