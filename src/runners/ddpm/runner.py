"""Generic ancestral-DDPM runner family implementation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from reference_cache import checkpoint_fingerprint
from runners.ddpm.sampling import (
    DDPM,
    expected_cost_per_root as _diffusion_expected_cost_per_root,
    resolve_split_percentages as _diffusion_resolve_split_percentages,
    run_probabilistic_inference_batch,
    run_solver_baseline_batch,
    sample_segment as _diffusion_sample_segment,
    segment_costs as _diffusion_segment_costs,
)
from runners.base import (
    BaseRunner,
    ComparisonModeSpec,
    SamplingConfig,
)

DDPM_TIMESTEP_SPACINGS = frozenset({"leading", "linspace", "trailing"})


def _ddpm_kwargs_from_scheduler_config(config: Mapping[str, Any] | None):
    if not config:
        return {}
    supported = {
        "beta_start",
        "beta_end",
        "beta_schedule",
        "trained_betas",
        "variance_type",
        "clip_sample",
        "clip_sample_range",
        "prediction_type",
        "thresholding",
        "dynamic_thresholding_ratio",
        "sample_max_value",
        "timestep_spacing",
        "steps_offset",
        "rescale_betas_zero_snr",
    }
    kwargs = {key: config[key] for key in supported if key in config and config[key] is not None}
    return kwargs


def resolved_hf_revision(pipe) -> str | None:
    """Resolve a cached Hugging Face snapshot revision without network access."""
    direct = getattr(pipe, "_commit_hash", None)
    if direct:
        return str(direct)
    unet_config = getattr(getattr(pipe, "unet", None), "config", None)
    source = getattr(unet_config, "_name_or_path", None)
    if source:
        snapshot = Path(str(source))
        if snapshot.parent.name == "snapshots" and snapshot.name:
            return snapshot.name
    return None


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
        sampler: str = "ddpm",
        timestep_spacing: str | None = None,
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
        self._sampler = sampler
        self._device = str(device)
        self._checkpoint_path = checkpoint_path or type(self).default_checkpoint_path()
        scheduler_config = dict(self._target_spec.get("scheduler_config") or {})
        configured_T = scheduler_config.get("num_train_timesteps")
        if configured_T is not None and int(configured_T) != self._T:
            raise ValueError(
                "DDPM T must match the checkpoint scheduler: "
                f"{self._T} != {int(configured_T)}"
            )
        ddpm_kwargs = _ddpm_kwargs_from_scheduler_config(scheduler_config)
        if timestep_spacing is not None:
            ddpm_kwargs["timestep_spacing"] = str(timestep_spacing)
        self._timestep_spacing = str(ddpm_kwargs.get("timestep_spacing", "leading"))
        self._ddpm = DDPM(
            T=self._T,
            device=self._device,
            sampling_steps=self._sampling_steps,
            **ddpm_kwargs,
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
        sampler: str = "ddpm",
        timestep_spacing: str | None = None,
        **kwargs,
    ) -> "DDPMRunner":
        if kwargs:
            raise ValueError(
                f"{cls.__name__}.load_from_checkpoint got unknown keys: "
                f"{sorted(kwargs)}"
            )
        path = checkpoint_path or cls.default_checkpoint_path()
        model, target_spec, data_mean, data_std = cls.load_model_and_stats(
            path, device, no_compile=no_compile
        )
        if T is None:
            scheduler_config = dict(target_spec.get("scheduler_config") or {})
            T = int(scheduler_config.get("num_train_timesteps") or 1000)
        if sampling_steps is None:
            sampling_steps = int(T)
        return cls(
            model=model,
            target_spec=target_spec,
            data_mean=data_mean,
            data_std=data_std,
            T=int(T),
            sampling_steps=int(sampling_steps),
            sampler=sampler,
            timestep_spacing=timestep_spacing,
            device=device,
            checkpoint_path=path,
        )

    def with_sampling_config(self, **kwargs) -> "DDPMRunner":
        allowed = {"sampler", "T", "sampling_steps", "timestep_spacing"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.with_sampling_config got unknown keys: "
                f"{sorted(unknown)}"
            )
        T = int(kwargs.get("T", self._T))
        sampling_steps = int(kwargs.get("sampling_steps", self._sampling_steps))
        sampler = str(kwargs.get("sampler", self._sampler))
        timestep_spacing = str(kwargs.get("timestep_spacing", self._timestep_spacing))
        return type(self)(
            model=self._model,
            target_spec=self._target_spec,
            data_mean=self._data_mean,
            data_std=self._data_std,
            T=T,
            sampling_steps=sampling_steps,
            sampler=sampler,
            timestep_spacing=timestep_spacing,
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
                "timestep_spacing": str(self._timestep_spacing),
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
    def ddpm(self) -> DDPM:
        return self._ddpm

    # ------------------------------------------------------------------
    # Modes and KS distance
    # ------------------------------------------------------------------

    def comparison_modes(self) -> Sequence[ComparisonModeSpec]:
        return tuple(type(self).comparison_mode_specs)

    def _validate_mode(self, comparison_mode: str) -> None:
        self.comparison_mode_spec(comparison_mode)

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
            self._ddpm, self._model, x, int(start_time), int(end_time), generator=generator
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
        return _diffusion_resolve_split_percentages(self._ddpm, split_percentages)

    def segment_cost(self, start_time: Any, end_time: Any) -> float:
        return float(self._ddpm.segment_cost(int(start_time), int(end_time)))

    def segment_costs(self, split_points: Sequence[Any]) -> list:
        return _diffusion_segment_costs(self._ddpm, split_points)

    def expected_cost_per_root(
        self,
        split_points: Sequence[Any],
        split_factors: Sequence[float],
    ) -> float:
        return float(
            _diffusion_expected_cost_per_root(
                self._ddpm, split_points, [float(f) for f in split_factors]
            )
        )

    def run_split_batch(
        self,
        *,
        n0_by_run: Sequence[int],
        split_points: Sequence[Any],
        split_factors_by_run: Sequence[Sequence[float]],
        generator=None,
        max_sampling_batch_size=None,
    ):
        return run_probabilistic_inference_batch(
            model=self._model,
            sampler=self._ddpm,
            n0_by_run=n0_by_run,
            split_points=split_points,
            split_factors_by_run=split_factors_by_run,
            postprocess_fn=self.postprocess_samples,
            generator=generator,
            max_sampling_batch_size=max_sampling_batch_size,
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
        max_sampling_batch_size=None,
        **solver_kwargs,
    ):
        T = int(solver_kwargs.get("T", self._T))
        sampling_steps = int(solver_kwargs.get("sampling_steps", self._sampling_steps))
        eta = float(solver_kwargs.get("eta", 0.0))
        return run_solver_baseline_batch(
            model=self._model,
            postprocess_fn=self.postprocess_samples,
            solver=solver,
            chunk_size=chunk_size,
            n0=n0,
            T=T,
            sampling_steps=sampling_steps,
            eta=eta,
            device=self._device,
            generator=generator,
            max_sampling_batch_size=max_sampling_batch_size,
        )

    # ------------------------------------------------------------------
    # Reference cache & comparison state
    # ------------------------------------------------------------------

    def _normalize_ddpm_sample_reference_config(
        self,
        reference_generation_config: Mapping[str, Any] | None,
        *,
        method: str,
    ) -> Mapping[str, Any]:
        if reference_generation_config is None:
            raise ValueError("reference_generation_config is required")
        cfg = dict(reference_generation_config)
        required = {"method", "sampler", "T", "sampling_steps"}
        missing = sorted(required - set(cfg))
        if missing:
            raise ValueError(
                f"reference_generation_config missing required keys: {missing}"
            )
        allowed = required | {"timestep_spacing", "seed"}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown reference_generation_config keys: {sorted(unknown)}"
            )
        if str(cfg["method"]) != method:
            raise ValueError(
                "reference_generation_config must set "
                f"method={method!r}, got {cfg['method']!r}"
            )
        sampler = str(cfg["sampler"])
        if sampler not in type(self).supported_samplers:
            raise ValueError(
                f"Unknown DDPM reference sampler {sampler!r}. "
                f"Available: {tuple(type(self).supported_samplers)}"
            )
        T = int(cfg["T"])
        sampling_steps = int(cfg["sampling_steps"])
        if T < 1:
            raise ValueError("DDPM reference T must be at least 1")
        if sampling_steps < 1:
            raise ValueError("DDPM reference sampling_steps must be at least 1")
        if sampling_steps > T:
            raise ValueError("DDPM reference sampling_steps cannot exceed T")
        timestep_spacing = str(
            cfg.get("timestep_spacing", self._timestep_spacing)
        )
        if timestep_spacing not in DDPM_TIMESTEP_SPACINGS:
            raise ValueError(
                f"Unknown DDPM timestep_spacing {timestep_spacing!r}. "
                f"Available: {tuple(sorted(DDPM_TIMESTEP_SPACINGS))}"
            )
        seed = int(cfg.get("seed", 0))
        if seed < 0:
            raise ValueError("reference seed must be nonnegative")
        return {
            "method": method,
            "sampler": sampler,
            "T": T,
            "sampling_steps": sampling_steps,
            "timestep_spacing": timestep_spacing,
            "seed": seed,
        }

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

        return self._normalize_ddpm_sample_reference_config(
            cfg,
            method="ddpm_samples",
        )

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
        key = {
            "runner": self.runner_name,
            "comparison_mode": comparison_mode,
            "target_spec": self._target_spec,
            "reference_generation_config": dict(ref_cfg),
        }
        if ref_cfg.get("method") in {"ddpm_samples", "hf_ddpm_scheduler"}:
            fingerprint = checkpoint_fingerprint(self._checkpoint_path)
            if fingerprint is not None:
                key["checkpoint_fingerprint"] = fingerprint
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
        return self._generate_ddpm_model_reference(
            ref_cfg,
            num_samples=num_samples,
            batch_size=batch_size,
            generator=generator,
            progress=progress,
        )

    def _generate_ddpm_model_reference(
        self,
        ref_cfg: Mapping[str, Any],
        *,
        num_samples: int,
        batch_size: int,
        generator=None,
        progress=None,
    ) -> torch.Tensor:
        """Generate a preallocated CPU reference through the splitting sampler."""
        ref_runner = self.with_sampling_config(
            sampler=ref_cfg["sampler"],
            T=ref_cfg["T"],
            sampling_steps=ref_cfg["sampling_steps"],
            timestep_spacing=ref_cfg["timestep_spacing"],
        )
        output = torch.empty(
            (int(num_samples), ref_runner.input_dim),
            dtype=torch.float32,
            device="cpu",
        )
        offset = 0
        remaining = int(num_samples)
        with torch.inference_mode():
            while remaining > 0:
                current = min(batch_size, remaining)
                x_T = torch.randn(
                    current, ref_runner.input_dim, device=self._device, generator=generator
                )
                if progress is not None:
                    progress["start_steps"](
                        len(ref_runner._ddpm._segment_timesteps(ref_runner._ddpm.T, 0))
                    )
                try:
                    generated = ref_runner._ddpm.sample_loop(
                        ref_runner._model,
                        x_T,
                        ref_runner._ddpm.T,
                        0,
                        generator=generator,
                        progress_callback=None if progress is None else progress["step"],
                    )
                finally:
                    if progress is not None:
                        progress["finish_steps"]()
                generated = ref_runner.postprocess_samples(generated).to(
                    device="cpu",
                    dtype=torch.float32,
                )
                if generated.shape != (current, ref_runner.input_dim):
                    raise ValueError(
                        "DDPM reference generator returned unexpected shape "
                        f"{tuple(generated.shape)}"
                    )
                output[offset : offset + current].copy_(generated)
                offset += current
                remaining -= current
                if progress is not None:
                    progress["batch"](1)
        return output

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
