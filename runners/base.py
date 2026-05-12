from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
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


class BaseRunner(ABC):
    runner_name: str

    @classmethod
    def runner_dir(cls) -> str:
        return os.path.join("checkpoints", cls.runner_name)

    @classmethod
    def default_checkpoint_path(cls) -> str:
        return os.path.join(cls.runner_dir(), "model_final.pt")

    @classmethod
    def load_configs(cls) -> dict[str, dict]:
        module = importlib.import_module(f"runners.{cls.runner_name}.configs")
        return dict(module.CONFIGS)

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
    def format_sampling_label(self) -> str:
        """Return a short label for the current sampling configuration."""

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
    def observable_values(
        self,
        samples: torch.Tensor,
        *,
        comparison_mode: str,
        observable_config: Any,
    ) -> torch.Tensor:
        """Return [num_samples, num_observables] values for phase-1 variance estimation."""

    @abstractmethod
    def reference_cache_key(self, comparison_mode: str) -> Mapping[str, Any]:
        """Return JSON-safe identity for reference samples/state for this mode."""

    @abstractmethod
    def generate_reference_samples(
        self,
        *,
        comparison_mode: str,
        num_samples: int,
        batch_size: int,
        generator=None,
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
