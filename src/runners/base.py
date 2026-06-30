from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping as MappingABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class SamplingConfig:
    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonModeSpec:
    name: str
    requires_reference_cache: bool
    reference_uses_sampling_config: bool = False
    description: str = ""


@dataclass(frozen=True)
class SamplingStepSchedule:
    name: str
    steps: int | Mapping[int, int] | None


@dataclass(frozen=True)
class ResolvedSamplingConfig:
    sampling_config: Mapping[str, Any]
    budget: int
    step_schedule: str


STEP_SCHEDULES = "step_schedules"
BASELINE_STEP_SCHEDULES = "baseline_step_schedules"
RESERVED_STEP_SCHEDULES = frozenset({"fixed", "default"})


def _coerce_positive_int(value, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if result < 1:
        raise ValueError(f"{name} must be >= 1, got {result}")
    return result


def _normalize_step_schedule(
    schedule_key: str,
    schedule_name: str,
    sampler_name: str,
    step_spec,
) -> SamplingStepSchedule:
    if not isinstance(step_spec, MappingABC):
        return SamplingStepSchedule(
            name=str(schedule_name),
            steps=_coerce_positive_int(
                step_spec,
                name=(
                    f"sampling steps for sampler {sampler_name!r}, "
                    f"schedule {schedule_name!r}"
                ),
            ),
        )

    normalized: dict[int, int] = {}
    for budget, steps in step_spec.items():
        budget_value = _coerce_positive_int(
            budget,
            name=f"budget for sampler {sampler_name!r}, schedule {schedule_name!r}",
        )
        normalized[budget_value] = _coerce_positive_int(
            steps,
            name=(
                f"sampling steps for sampler {sampler_name!r}, "
                f"schedule {schedule_name!r}, budget {budget!r}"
            ),
        )
    if not normalized:
        raise ValueError(
            f"{schedule_key}[{schedule_name!r}][{sampler_name!r}] must not be empty"
        )
    return SamplingStepSchedule(name=str(schedule_name), steps=normalized)


def _step_schedules_map(
    cfg: Mapping[str, Any],
    schedule_key: str = STEP_SCHEDULES,
) -> dict[str, dict[str, SamplingStepSchedule]]:
    raw = cfg.get(schedule_key)
    if raw is None:
        return {}
    if not isinstance(raw, MappingABC):
        raise ValueError(f"{schedule_key} must be a mapping")

    normalized: dict[str, dict[str, SamplingStepSchedule]] = {}
    for schedule, sampler_specs in raw.items():
        schedule_name = str(schedule)
        if not schedule_name:
            raise ValueError(f"{schedule_key} has an empty schedule name")
        if schedule_name in RESERVED_STEP_SCHEDULES:
            raise ValueError(
                f"{schedule_key} schedule name {schedule_name!r} is reserved"
            )
        if not isinstance(sampler_specs, MappingABC):
            raise ValueError(f"{schedule_key}[{schedule_name!r}] must be a mapping")
        if not sampler_specs:
            raise ValueError(f"{schedule_key}[{schedule_name!r}] must not be empty")

        by_sampler: dict[str, SamplingStepSchedule] = {}
        for sampler, step_spec in sampler_specs.items():
            sampler_name = str(sampler)
            if not sampler_name:
                raise ValueError(
                    f"{schedule_key}[{schedule_name!r}] has an empty sampler key"
                )
            by_sampler[sampler_name] = _normalize_step_schedule(
                schedule_key, schedule_name, sampler_name, step_spec
            )
        normalized[schedule_name] = by_sampler
    return normalized


def steps_for_budget(
    schedule: SamplingStepSchedule,
    sampler_name: str,
    budget: int,
    schedule_key: str = STEP_SCHEDULES,
) -> int | None:
    if schedule.steps is None:
        return None
    if isinstance(schedule.steps, int):
        return int(schedule.steps)

    budget_value = _coerce_positive_int(budget, name="budget")
    if budget_value not in schedule.steps:
        raise ValueError(
            f"{schedule_key} schedule {schedule.name!r} for sampler "
            f"{sampler_name!r} has no entry for budget {budget_value}"
        )
    return int(schedule.steps[budget_value])


def _schedule_for_name(
    schedules: Sequence[SamplingStepSchedule],
    sampler_name: str,
    step_schedule: str | None,
    schedule_key: str = STEP_SCHEDULES,
) -> SamplingStepSchedule:
    if step_schedule is None:
        if len(schedules) == 1:
            return schedules[0]
        raise ValueError(
            f"{schedule_key} has multiple schedules for sampler "
            f"{sampler_name!r}; step_schedule is required"
        )
    step_schedule_name = str(step_schedule)
    for schedule in schedules:
        if schedule.name == step_schedule_name:
            return schedule
    raise ValueError(
        f"{schedule_key} has no schedule {step_schedule_name!r} "
        f"for sampler {sampler_name!r}"
    )


def step_schedules_for_sampler(
    cfg: Mapping[str, Any],
    sampler: str,
    schedule_key: str = STEP_SCHEDULES,
    allow_default: bool = True,
) -> list[SamplingStepSchedule]:
    schedules_map = _step_schedules_map(cfg, schedule_key)
    sampler_name = str(sampler)
    if not schedules_map:
        if not allow_default:
            raise ValueError(f"{schedule_key} has no entry for sampler {sampler_name!r}")
        return [SamplingStepSchedule(name="default", steps=None)]

    schedules = [
        sampler_specs[sampler_name]
        for sampler_specs in schedules_map.values()
        if sampler_name in sampler_specs
    ]
    if not schedules:
        raise ValueError(
            f"{schedule_key} has no entry for sampler {sampler_name!r}"
        )
    return schedules


def sampling_step_schedules_for_config(
    cfg: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
) -> list[SamplingStepSchedule]:
    if "sampling_steps" in sampling_config and sampling_config["sampling_steps"] is not None:
        return [
            SamplingStepSchedule(
                name="fixed",
                steps=_coerce_positive_int(
                    sampling_config["sampling_steps"], name="sampling_steps"
                ),
            )
        ]

    schedules_map = _step_schedules_map(cfg)
    if not schedules_map:
        return [SamplingStepSchedule(name="default", steps=None)]

    sampler = _sampler_for_step_schedules(sampling_config, schedules_map)
    return step_schedules_for_sampler(cfg, sampler)


def _sampler_for_step_schedules(
    sampling_config: Mapping[str, Any],
    schedules_map: Mapping[str, Mapping[str, Any]],
) -> str:
    sampler = sampling_config.get("sampler")
    if sampler is not None:
        return str(sampler)
    samplers = {
        str(sampler)
        for sampler_specs in schedules_map.values()
        for sampler in sampler_specs
    }
    if len(samplers) == 1:
        return next(iter(samplers))
    raise ValueError(
        "sampling_config must include 'sampler' when "
        f"{STEP_SCHEDULES} is configured for multiple samplers"
    )


def sampling_steps_for_sampler_budget(
    cfg: Mapping[str, Any],
    sampler: str,
    budget: int,
    step_schedule: str | None = None,
) -> int | None:
    """Return configured steps for (sampler, budget), or None when unset."""
    schedules_map = _step_schedules_map(cfg)
    if not schedules_map:
        return None

    sampler_name = str(sampler)
    schedule = _schedule_for_name(
        step_schedules_for_sampler(cfg, sampler_name), sampler_name, step_schedule
    )
    return steps_for_budget(schedule, sampler_name, budget)


def resolve_sampling_config_for_budget(
    cfg: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
    budget: int,
    step_schedule: str | None = None,
) -> dict[str, Any]:
    """Return a concrete sampling config for a total budget."""
    resolved = dict(sampling_config)
    if "sampling_steps" in resolved and resolved["sampling_steps"] is not None:
        resolved["sampling_steps"] = _coerce_positive_int(
            resolved["sampling_steps"], name="sampling_steps"
        )
        return resolved

    schedules_map = _step_schedules_map(cfg)
    if not schedules_map:
        return resolved

    sampler = _sampler_for_step_schedules(resolved, schedules_map)
    resolved.setdefault("sampler", sampler)
    schedule = _schedule_for_name(
        step_schedules_for_sampler(cfg, sampler), sampler, step_schedule
    )
    resolved["sampling_steps"] = steps_for_budget(schedule, sampler, budget)
    return resolved


def _budgets_for_config(cfg: Mapping[str, Any]) -> list[int]:
    raw_budgets = cfg.get("B_list")
    if not raw_budgets:
        raise ValueError("B_list must be non-empty")
    return [_coerce_positive_int(budget, name="B_list entry") for budget in raw_budgets]


def iter_budget_resolved_sampling_config_specs(cfg: Mapping[str, Any]):
    """Yield resolved sampling configs with step-schedule metadata."""
    budgets = _budgets_for_config(cfg)
    for sampling_config in cfg.get("sampling_configs") or []:
        for schedule in sampling_step_schedules_for_config(cfg, sampling_config):
            for budget in budgets:
                yield ResolvedSamplingConfig(
                    sampling_config=resolve_sampling_config_for_budget(
                        cfg,
                        sampling_config,
                        int(budget),
                        step_schedule=schedule.name,
                    ),
                    budget=int(budget),
                    step_schedule=str(schedule.name),
                )


def iter_budget_resolved_sampling_configs(cfg: Mapping[str, Any]):
    """Yield sampling configs after applying budget-dependent step counts."""
    for resolved in iter_budget_resolved_sampling_config_specs(cfg):
        yield dict(resolved.sampling_config)


class BaseRunner(ABC):
    runner_name: str
    config_module: str | None = None

    @classmethod
    def runner_dir(cls) -> str:
        return os.path.join("checkpoints", cls.runner_name)

    @classmethod
    def default_checkpoint_path(cls) -> str:
        return os.path.join(cls.runner_dir(), "model_final.pt")

    @classmethod
    def load_configs(cls) -> dict[str, dict]:
        module_name = cls.config_module or f"runners.{cls.runner_name}.configs"
        module = importlib.import_module(module_name)
        return deepcopy(dict(module.CONFIGS))

    @classmethod
    def get_config(cls, name: str) -> dict:
        configs = cls.load_configs()
        if name not in configs:
            raise ValueError(
                f"Runner {cls.runner_name!r} has no config {name!r}. "
                f"Available: {sorted(configs)}"
            )
        cfg = dict(configs[name])
        sampling_defaults = cfg.pop("sampling_defaults", None) or {}
        cfg["sampling_configs"] = [
            {**sampling_defaults, **entry}
            for entry in cfg["sampling_configs"]
        ]
        return cfg

    @classmethod
    @abstractmethod
    def add_train_args(cls, parser) -> None:
        """Register runner-specific training args."""

    @classmethod
    def add_compare_args(cls, parser) -> None:
        """Register runner-specific compare/sweep args. Default: no extra args."""

    @classmethod
    def add_reference_args(cls, parser) -> None:
        """Register runner-specific reference-cache args. Default: no extra args."""

    @classmethod
    @abstractmethod
    def train_from_args(cls, args) -> None:
        """Train this runner from CLI args and save a checkpoint."""

    @classmethod
    @abstractmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        device: str,
        no_compile: bool = False,
        **kwargs,
    ) -> "BaseRunner":
        """Load model, target info, normalization, and default schedule config."""

    @abstractmethod
    def with_sampling_config(self, **kwargs) -> "BaseRunner":
        """Return a cheap copy using the same loaded model but a different sampling config."""

    @property
    @abstractmethod
    def device(self) -> str:
        ...

    @property
    @abstractmethod
    def input_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def target_spec(self) -> Mapping[str, Any]:
        """Runner-owned target description. JSON-serializable when possible."""

    @property
    @abstractmethod
    def sampling_config(self) -> SamplingConfig:
        ...

    def sampling_cache_key(self) -> Mapping[str, Any]:
        """Return JSON-safe identity for the current sampler configuration."""
        return dict(self.sampling_config.values)

    def parse_baseline_name(self, name: str) -> dict:
        """Parse a config baseline name into a generic compare spec."""
        if name == "fixed_N":
            return {"mode": "fixed_N"}
        for solver in sorted(self.solver_names(), key=len, reverse=True):
            parsed = self.parse_solver_baseline_name(name, solver)
            if parsed is not None:
                return parsed
        raise ValueError(
            f"Unknown baseline {name!r}. Expected fixed_N or one of: "
            f"{', '.join(self.solver_names())}"
        )

    def parse_solver_baseline_name(self, name: str, solver: str) -> dict | None:
        if name == solver:
            return self._solver_baseline_spec(name, solver)
        prefix = f"{solver}_"
        if not name.startswith(prefix):
            return None
        rest = name[len(prefix) :].split("_")
        if len(rest) != 1:
            return None
        try:
            steps = int(rest[0])
        except ValueError as exc:
            raise ValueError(f"baseline {name!r} must use integer steps") from exc
        if steps < 1:
            raise ValueError(f"baseline {name!r} must use steps >= 1")
        return self._solver_baseline_spec(name, solver, steps)

    def _solver_baseline_spec(
        self,
        name: str,
        solver: str,
        steps: int | None = None,
        **kwargs,
    ) -> dict:
        solver_kwargs = dict(kwargs)
        if steps is not None:
            solver_kwargs = {"sampling_steps": int(steps), **solver_kwargs}
        return {
            "mode": "solver_baseline",
            "solver": solver,
            "solver_kwargs": solver_kwargs,
        }

    @property
    @abstractmethod
    def start_time(self) -> Any:
        """Native start point for the runner trajectory."""

    @property
    @abstractmethod
    def end_time(self) -> Any:
        """Native terminal point for the runner trajectory."""

    @abstractmethod
    def comparison_modes(self) -> Sequence[ComparisonModeSpec]:
        """Return modes supported by this runner."""

    @abstractmethod
    def sample_prior(self, num_samples: int, *, generator=None) -> torch.Tensor:
        """Sample native initial noise/state at start_time."""

    @abstractmethod
    def sample_segment(
        self,
        x: torch.Tensor,
        start_time: Any,
        end_time: Any,
        *,
        generator=None,
    ) -> torch.Tensor:
        """Propagate native samples from start_time to end_time."""

    @abstractmethod
    def postprocess_samples(self, native_samples: torch.Tensor) -> torch.Tensor:
        """Convert final native samples into target/comparison space."""

    @abstractmethod
    def resolve_split_percentages(self, split_percentages: Sequence[float]):
        """Return runner-native split information."""

    @abstractmethod
    def segment_cost(self, start_time: Any, end_time: Any) -> float:
        """Return cost/NFE for one particle over a segment."""

    @abstractmethod
    def segment_costs(self, split_points: Sequence[Any]) -> list:
        """Return costs for all segments implied by split_points."""

    @abstractmethod
    def expected_cost_per_root(
        self,
        split_points: Sequence[Any],
        split_factors: Sequence[float],
    ) -> float:
        """Expected total cost from one root under split factors."""

    @abstractmethod
    def run_split_batch(
        self,
        *,
        n0_by_run: Sequence[int],
        split_points: Sequence[Any],
        split_factors_by_run: Sequence[Sequence[float]],
        generator=None,
    ):
        """Return (samples_by_run, realized_costs, sampling_time). Samples must be postprocessed."""

    @abstractmethod
    def solver_names(self) -> Sequence[str]:
        """Return supported full-trajectory baseline solvers."""

    @abstractmethod
    def solver_cost(self, solver: str, **solver_kwargs) -> float:
        """Return per-sample cost/NFE for a full baseline solver run."""

    @abstractmethod
    def solver_cache_key(self, solver: str, **solver_kwargs) -> Mapping[str, Any]:
        """Return JSON-safe identity for a full-trajectory baseline solver run."""

    @abstractmethod
    def schedule_cost(self) -> float:
        """Return per-sample cost/NFE for the current sampling configuration."""

    @abstractmethod
    def run_solver_baseline_batch(
        self,
        *,
        solver: str,
        chunk_size: int,
        n0: int,
        generator=None,
        **solver_kwargs,
    ):
        """Return (samples_by_run, sampling_time). Samples must be postprocessed."""

    @abstractmethod
    def normalize_reference_generation_config(
        self,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        """Validate and canonicalize config used to build cached references."""

    @abstractmethod
    def reference_cache_key(
        self,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return JSON-safe identity for reference samples/state for this mode."""

    @abstractmethod
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
        """Generate or load raw reference samples for modes that need cached samples."""

    @abstractmethod
    def prepare_comparison_state(
        self,
        *,
        comparison_mode: str,
        reference_samples=None,
    ) -> Any:
        """Build KS comparison state. For true_dist this may return None."""

    @abstractmethod
    def compute_ks_distance(
        self,
        samples,
        *,
        comparison_mode: str,
        comparison_state=None,
        extra_samples=None,
    ) -> float:
        """Compute KS distance. extra_samples is used for reused phase-1 samples."""
