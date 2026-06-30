"""Generic DDPM/DDIM runner family implementation."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from runners.ddpm.sampling import (
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
from runners.base import (
    BaseRunner,
    ComparisonModeSpec,
    SamplingConfig,
)


class DDPMRunner(BaseRunner):
    runner_name = "ddpm"
    target_adapter = None
    supported_samplers: Sequence[str] = ()
    supported_solvers: Sequence[str] = ()
    comparison_mode_specs: Sequence[ComparisonModeSpec] = ()
    DEFAULT_SOLVER = "ddim"

    @classmethod
    def _target_adapter(cls):
        if cls.target_adapter is None:
            raise RuntimeError(f"{cls.__name__} must define target_adapter")
        return cls.target_adapter

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
        if sampler not in type(self).supported_samplers:
            raise ValueError(
                f"Unknown DDPM sampler {sampler!r}. "
                f"Available: {tuple(type(self).supported_samplers)}"
            )
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
        raise NotImplementedError(f"{cls.__name__}.add_train_args must be implemented")

    @classmethod
    def train_from_args(cls, args) -> None:
        raise NotImplementedError(f"{cls.__name__}.train_from_args must be implemented")

    @classmethod
    def load_model_and_stats(cls, checkpoint_path: str, device: str, *, no_compile: bool):
        raise NotImplementedError(
            f"{cls.__name__}.load_model_and_stats must be implemented"
        )

    @staticmethod
    def model_input_dim(model) -> int:
        raise NotImplementedError("DDPM subclasses must implement model_input_dim")

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
    ) -> "DDPMRunner":
        path = checkpoint_path or cls.default_checkpoint_path()
        model, target_spec, data_mean, data_std = cls.load_model_and_stats(
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

    def with_sampling_config(self, **kwargs) -> "DDPMRunner":
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
        return type(self)(
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
        return int(type(self).model_input_dim(self._model))

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
        return tuple(type(self).comparison_mode_specs)

    def _validate_mode(self, comparison_mode: str) -> None:
        names = {spec.name for spec in self.comparison_modes()}
        if comparison_mode not in names:
            raise ValueError(
                f"Unknown comparison_mode '{comparison_mode}'. "
                f"Available: {sorted(names)}"
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
        return type(self)._target_adapter().denormalize(
            native_samples,
            self._data_mean,
            self._data_std,
        )

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
            postprocess_fn=self.postprocess_samples,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # Baseline solvers
    # ------------------------------------------------------------------

    def solver_names(self) -> Sequence[str]:
        return tuple(type(self).supported_solvers)

    def parse_solver_baseline_name(self, name: str, solver: str) -> dict | None:
        prefix = f"{solver}_"
        rest = name[len(prefix) :].split("_") if name.startswith(prefix) else []
        has_eta = (
            len(rest) == 1 and rest[0].startswith("eta")
        ) or (
            len(rest) == 2 and rest[1].startswith("eta")
        )
        if solver != "ddim" or not has_eta:
            parsed = super().parse_solver_baseline_name(name, solver)
            if parsed is not None or solver != "ddim":
                return parsed
        if not name.startswith(prefix):
            return None
        if len(rest) == 1 and rest[0].startswith("eta") and rest[0] != "eta":
            step_token = None
            eta_token = rest[0]
        elif len(rest) == 2 and rest[1].startswith("eta") and rest[1] != "eta":
            step_token = rest[0]
            eta_token = rest[1]
        else:
            return None
        try:
            steps = None if step_token is None else int(step_token)
            eta = float(eta_token[3:])
        except ValueError as exc:
            raise ValueError(f"baseline {name!r} has bad ddim solver syntax") from exc
        if steps is not None and steps < 1:
            raise ValueError(f"baseline {name!r} must use steps >= 1")
        return self._solver_baseline_spec(name, solver, steps, eta=eta)

    def solver_cost(self, solver: str, **solver_kwargs) -> float:
        if solver not in type(self).supported_solvers:
            raise ValueError(f"Unknown solver '{solver}'")
        sampling_steps = solver_kwargs.get("sampling_steps", self._sampling_steps)
        return float(int(sampling_steps))

    def solver_cache_key(self, solver: str, **solver_kwargs) -> Mapping[str, Any]:
        if solver not in type(self).supported_solvers:
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
            postprocess_fn=self.postprocess_samples,
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
    # Reference cache & comparison state
    # ------------------------------------------------------------------

    def normalize_reference_generation_config(
        self,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        self._validate_mode(comparison_mode)
        if comparison_mode == "true_dist":
            return {}
        if reference_generation_config is None:
            raise ValueError(
                f"reference_generation_config is required for comparison_mode={comparison_mode!r}"
            )
        cfg = dict(reference_generation_config)
        method = str(cfg.get("method", ""))
        if comparison_mode == "true_samples":
            if method != "target_samples":
                raise ValueError(
                    "true_samples reference_generation_config must set method='target_samples'"
                )
            allowed = {"method"}
            unknown = set(cfg) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown true_samples reference_generation_config keys: {sorted(unknown)}"
                )
            return {"method": "target_samples"}

        if method != "ddpm_samples":
            raise ValueError(
                f"{comparison_mode} reference_generation_config must set method='ddpm_samples'"
            )
        required = {"method", "sampler", "T", "sampling_steps", "eta"}
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
        sampler = str(cfg["sampler"])
        if sampler not in type(self).supported_samplers:
            raise ValueError(
                f"Unknown DDPM reference sampler {sampler!r}. "
                f"Available: {tuple(type(self).supported_samplers)}"
            )
        T = int(cfg["T"])
        sampling_steps = int(cfg["sampling_steps"])
        if T < 1 or sampling_steps < 1:
            raise ValueError("T and sampling_steps must be at least 1")
        return {
            "method": "ddpm_samples",
            "sampler": sampler,
            "T": T,
            "sampling_steps": sampling_steps,
            "eta": float(cfg["eta"]),
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
        if comparison_mode == "true_dist":
            return {
                "runner": self.runner_name,
                "comparison_mode": comparison_mode,
                "target_spec": self._target_spec,
            }
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
        self._validate_mode(comparison_mode)
        if comparison_mode == "true_dist":
            raise ValueError("true_dist comparison does not require reference samples")
        ref_cfg = self.normalize_reference_generation_config(
            comparison_mode, reference_generation_config
        )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        batches: list[torch.Tensor] = []
        remaining = int(num_samples)

        if comparison_mode == "true_samples":
            with torch.inference_mode():
                while remaining > 0:
                    current = min(batch_size, remaining)
                    batches.append(
                        type(self)._target_adapter().sample_target_spec(
                            self._target_spec, current, self._device
                        ).cpu()
                    )
                    remaining -= current
                    if progress is not None:
                        progress["batch"](1)
            return torch.cat(batches, dim=0)

        # ddpm_samples
        ref_runner = self.with_sampling_config(
            sampler=ref_cfg["sampler"],
            T=ref_cfg["T"],
            sampling_steps=ref_cfg["sampling_steps"],
            eta=ref_cfg["eta"],
        )
        with torch.inference_mode():
            while remaining > 0:
                current = min(batch_size, remaining)
                x_T = torch.randn(
                    current, ref_runner.input_dim, device=self._device, generator=generator
                )
                if progress is not None:
                    progress["start_steps"](
                        len(ref_runner._ddim._segment_timesteps(ref_runner._ddim.T, 0))
                    )
                try:
                    generated = ref_runner._ddim.sample_loop(
                        ref_runner._model,
                        x_T,
                        ref_runner._ddim.T,
                        0,
                        generator=generator,
                        progress_callback=None if progress is None else progress["step"],
                    )
                finally:
                    if progress is not None:
                        progress["finish_steps"]()
                generated = ref_runner.postprocess_samples(generated)
                batches.append(generated.cpu())
                remaining -= current
                if progress is not None:
                    progress["batch"](1)
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
