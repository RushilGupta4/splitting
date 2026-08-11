from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from metrics.ks import (
    compute_reference_ks_distance,
    prepare_reference_cdf_state,
    warm_reference_ks_kernel,
)
from metrics.utils import coerce_samples_np
from runners.base import BaseRunner, ComparisonModeSpec, SamplingConfig
from runners.sde.cases import SDE_CASES
from runners.sde.sampling import SDE_SAMPLERS, sample_sde_segment
from runners.splitting import (
    apply_to_run_batches,
    balanced_branch_counts_by_group,
    balanced_split_with_run_ids,
    child_counts_by_run,
    iter_child_parent_batches_by_run,
    normalize_max_sampling_batch_size,
    split_counts_by_run_batches,
)
from utils import validate_split_percentages

SUPPORTED_SAMPLERS = SDE_SAMPLERS


def _append_by_counts(parts_by_run, values: torch.Tensor, counts_by_run):
    offset = 0
    for run_idx, count in enumerate(counts_by_run):
        count = int(count)
        if count > 0:
            parts_by_run[run_idx].append(values[offset : offset + count])
        offset += count


def _cat_parts_by_run(parts_by_run, empty_template: torch.Tensor):
    return [
        torch.cat(parts, dim=0) if parts else empty_template[:0]
        for parts in parts_by_run
    ]


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
        names = {spec.name for spec in self.comparison_modes()}
        if comparison_mode not in names:
            raise ValueError(
                f"Unknown comparison_mode {comparison_mode!r}. Available: {sorted(names)}"
            )

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
        points = list(split_points)
        starts = [self.start_time] + points
        ends = points + [self.end_time]
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
        max_sampling_batch_size=None,
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
        max_sampling_batch_size = normalize_max_sampling_batch_size(
            max_sampling_batch_size
        )

        split_points = [self._coerce_index(p, name="split_point") for p in split_points]
        num_runs = int(n0_tensor.numel())
        realized_costs = torch.zeros(num_runs, device=self._device, dtype=torch.long)
        sampling_start = time.perf_counter()

        if max_sampling_batch_size is not None and not split_points:
            parts_by_run = [[] for _ in range(num_runs)]
            empty_template = self.postprocess_samples(self.sample_prior(0))
            segment_cost = int(self.segment_cost(self.start_time, self.end_time))
            for counts in split_counts_by_run_batches(
                n0_by_run,
                max_sampling_batch_size,
            ):
                total = int(sum(counts))
                if total <= 0:
                    continue
                counts_tensor = torch.as_tensor(
                    counts, device=self._device, dtype=torch.long
                )
                x = self.sample_prior(total, generator=generator)
                realized_costs += counts_tensor * segment_cost
                x = self.sample_segment(
                    x, self.start_time, self.end_time, generator=generator
                )
                x = self.postprocess_samples(x)
                empty_template = x
                _append_by_counts(parts_by_run, x, counts)
            sampling_time = time.perf_counter() - sampling_start
            return (
                _cat_parts_by_run(parts_by_run, empty_template),
                realized_costs.detach().cpu().tolist(),
                sampling_time,
            )

        run_ids = torch.repeat_interleave(
            torch.arange(num_runs, device=self._device, dtype=torch.long), n0_tensor
        )
        if max_sampling_batch_size is None:
            x = self.sample_prior(int(n0_tensor.sum().item()), generator=generator)
        else:
            initial_parts_by_run = [[] for _ in range(num_runs)]
            for counts in split_counts_by_run_batches(
                n0_by_run,
                max_sampling_batch_size,
            ):
                total = int(sum(counts))
                if total <= 0:
                    continue
                batch = self.sample_prior(total, generator=generator)
                batch = self.sample_segment(
                    batch, self.start_time, split_points[0], generator=generator
                )
                _append_by_counts(initial_parts_by_run, batch, counts)
            x = torch.cat([torch.cat(parts, dim=0) for parts in initial_parts_by_run], dim=0)

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
            if max_sampling_batch_size is None:
                x = self.sample_segment(
                    x, self.start_time, split_points[0], generator=generator
                )
            split_factors_array = np.asarray(split_factors_by_run, dtype=float)
            if not np.isfinite(split_factors_array).all() or np.any(
                split_factors_array < 0.0
            ):
                raise ValueError("split_factors_by_run must contain finite nonnegative values")
            if split_factors_array.shape != (num_runs, len(split_points)):
                raise ValueError("split_factors_by_run has incompatible shape")
            split_factors_tensor = torch.as_tensor(
                split_factors_array,
                device=self._device,
                dtype=torch.float64,
            )
            for idx, split_point in enumerate(split_points):
                end_t = (
                    split_points[idx + 1]
                    if idx + 1 < len(split_points)
                    else self.end_time
                )
                if max_sampling_batch_size is not None and idx + 1 == len(split_points):
                    child_counts = balanced_branch_counts_by_group(
                        run_ids,
                        split_factors_tensor[:, idx],
                        num_groups=num_runs,
                        generator=generator,
                    )
                    counts = child_counts_by_run(
                        run_ids,
                        child_counts,
                        num_runs=num_runs,
                    )
                    realized_costs += counts * int(self.segment_cost(split_point, end_t))
                    parts_by_run = [[] for _ in range(num_runs)]
                    empty_template = self.postprocess_samples(x[:0])
                    for batch_parent_indices, batch_counts in iter_child_parent_batches_by_run(
                        run_ids,
                        child_counts,
                        num_runs=num_runs,
                        max_count_per_run=max_sampling_batch_size,
                    ):
                        if batch_parent_indices.numel() == 0:
                            continue
                        batch = x.index_select(0, batch_parent_indices)
                        batch = self.sample_segment(
                            batch, split_point, end_t, generator=generator
                        )
                        batch = self.postprocess_samples(batch)
                        empty_template = batch
                        _append_by_counts(parts_by_run, batch, batch_counts)
                    sampling_time = time.perf_counter() - sampling_start
                    return (
                        _cat_parts_by_run(parts_by_run, empty_template),
                        realized_costs.detach().cpu().tolist(),
                        sampling_time,
                    )
                x, run_ids = balanced_split_with_run_ids(
                    x, run_ids, split_factors_tensor[:, idx], generator=generator
                )
                counts = torch.bincount(run_ids, minlength=num_runs)
                realized_costs += counts * int(self.segment_cost(split_point, end_t))
                x = apply_to_run_batches(
                    x,
                    run_ids,
                    num_runs=num_runs,
                    max_count_per_run=max_sampling_batch_size,
                    fn=lambda batch, split_point=split_point, end_t=end_t: self.sample_segment(
                        batch, split_point, end_t, generator=generator
                    ),
                )

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
        max_sampling_batch_size = normalize_max_sampling_batch_size(
            max_sampling_batch_size
        )
        sampling_start = time.perf_counter()
        if max_sampling_batch_size is not None:
            counts_by_run = [int(n0)] * int(chunk_size)
            parts_by_run = [[] for _ in range(int(chunk_size))]
            empty_template = runner.postprocess_samples(runner.sample_prior(0))
            for counts in split_counts_by_run_batches(
                counts_by_run,
                max_sampling_batch_size,
            ):
                total = int(sum(counts))
                if total <= 0:
                    continue
                x = runner.sample_prior(total, generator=generator)
                x = runner.sample_segment(
                    x, runner.start_time, runner.end_time, generator=generator
                )
                x = runner.postprocess_samples(x).to(dtype=torch.float32)
                empty_template = x
                _append_by_counts(parts_by_run, x, counts)
            sampling_time = time.perf_counter() - sampling_start
            return _cat_parts_by_run(parts_by_run, empty_template), sampling_time

        x = runner.sample_prior(int(chunk_size) * int(n0), generator=generator)
        x = runner.sample_segment(
            x, runner.start_time, runner.end_time, generator=generator
        )
        x = runner.postprocess_samples(x)
        sampling_time = time.perf_counter() - sampling_start
        x = (
            x.to(dtype=torch.float32)
            .reshape(int(chunk_size), int(n0), self._dimension)
            .contiguous()
        )
        return [x[idx] for idx in range(int(chunk_size))], sampling_time

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
        self._validate_mode(comparison_mode)
        if metric_params:
            raise ValueError("ks no longer accepts metric parameters")
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
