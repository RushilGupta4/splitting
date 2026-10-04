from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from reference_cache import checkpoint_fingerprint
from runners.base import BaseRunner
from runners.edm.sampling import (
    EDMSchedule,
    _schedule_segment_indices,
    dpmpp_2s_sample_segment,
    edm_sample_segment,
    resolve_split_percentages as _resolve_split_percentages,
    sde_euler_maruyama_sample_segment,
)
from runners.splitting import run_full_trajectory_batch

EDM_STOCHASTIC_DEFAULT_PARAMS = {
    "S_churn": 0.0,
    "S_min": 0.0,
    "S_max": 80.0,
    "S_noise": 1.0,
}
DPMPP_2S_DEFAULT_PARAMS = {
    "stochastic_churn_rate": 2.5,
    "churn_min_noise_level": 0.05,
    "churn_max_noise_level": 50.0,
    "noise_level_inflation_factor": 1.0,
}
SAMPLER_PARAM_DEFAULTS = {
    "edm_stochastic": EDM_STOCHASTIC_DEFAULT_PARAMS,
    "dpmpp_2s": DPMPP_2S_DEFAULT_PARAMS,
    "sde_euler_maruyama": {},
}
SAMPLE_SEGMENT_FNS = {
    "edm_stochastic": edm_sample_segment,
    "dpmpp_2s": dpmpp_2s_sample_segment,
    "sde_euler_maruyama": sde_euler_maruyama_sample_segment,
}
FLAT_EDM_PARAM_KEYS = frozenset(EDM_STOCHASTIC_DEFAULT_PARAMS)


class EDMRunner(BaseRunner):
    """EDM sampler family; subclasses set ``target_adapter`` and the model hooks."""

    runner_name = "edm"
    target_adapter: Any = None
    supported_samplers: Sequence[str] = ()

    def __init__(
        self,
        *,
        model,
        target_spec: Mapping[str, Any],
        data_mean: torch.Tensor,
        data_std: torch.Tensor,
        sampling_steps: int,
        sampler: str = "edm_stochastic",
        sigma_min: float,
        sigma_max: float,
        rho: float,
        sigma_data: float,
        device: str,
        sampler_params: Mapping[str, Any] | None = None,
        checkpoint_path: str | None = None,
    ):
        sampler = str(sampler)
        if sampler not in type(self).supported_samplers:
            raise ValueError(
                f"Unknown EDM sampler {sampler!r}. "
                f"Available: {tuple(type(self).supported_samplers)}"
            )
        self._sampler_params = self._normalize_sampler_params(sampler, sampler_params)
        self._model = model
        self._target_spec = dict(target_spec)
        self._data_mean = data_mean
        self._data_std = data_std
        schedule_config = self._normalize_schedule_config(
            sampling_steps=sampling_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
        )
        self._sampling_steps = schedule_config["sampling_steps"]
        self._sampler = sampler
        self._sigma_min = schedule_config["sigma_min"]
        self._sigma_max = schedule_config["sigma_max"]
        self._rho = schedule_config["rho"]
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

    @staticmethod
    def _normalize_schedule_config(
        *,
        sampling_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
    ) -> dict[str, int | float]:
        normalized = {
            "sampling_steps": int(sampling_steps),
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "rho": float(rho),
        }
        if normalized["sampling_steps"] < 2:
            raise ValueError("EDM sampling_steps must be at least 2")
        schedule_values = (
            normalized["sigma_min"],
            normalized["sigma_max"],
            normalized["rho"],
        )
        if not np.isfinite(schedule_values).all():
            raise ValueError("EDM sigma_min, sigma_max, and rho must be finite")
        if normalized["sigma_min"] <= 0.0:
            raise ValueError("EDM sigma_min must be > 0")
        if normalized["sigma_max"] <= normalized["sigma_min"]:
            raise ValueError("EDM sigma_max must be > sigma_min")
        if normalized["rho"] <= 0.0:
            raise ValueError("EDM rho must be > 0")
        return normalized

    @staticmethod
    def _normalize_sampler_params(
        sampler: str,
        sampler_params: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        if sampler not in SAMPLER_PARAM_DEFAULTS:
            raise ValueError(f"Unknown EDM sampler {sampler!r}")
        params = dict(SAMPLER_PARAM_DEFAULTS[sampler])
        if sampler_params:
            params.update(dict(sampler_params))

        unknown = set(params) - set(SAMPLER_PARAM_DEFAULTS[sampler])
        if unknown:
            raise ValueError(
                f"Unknown parameters for sampler {sampler!r}: {sorted(unknown)}"
            )

        normalized = {key: float(value) for key, value in params.items()}
        if normalized and not np.isfinite(tuple(normalized.values())).all():
            raise ValueError("EDM sampler parameters must be finite")
        if sampler == "edm_stochastic":
            if normalized["S_churn"] < 0.0:
                raise ValueError("S_churn must be >= 0")
            if normalized["S_noise"] < 0.0:
                raise ValueError("S_noise must be >= 0")
            if normalized["S_min"] > normalized["S_max"]:
                raise ValueError("S_min must be <= S_max")
        elif sampler == "dpmpp_2s":
            if normalized["stochastic_churn_rate"] < 0.0:
                raise ValueError("stochastic_churn_rate must be >= 0")
            if normalized["noise_level_inflation_factor"] < 0.0:
                raise ValueError("noise_level_inflation_factor must be >= 0")
            if normalized["churn_min_noise_level"] > normalized["churn_max_noise_level"]:
                raise ValueError("churn_min_noise_level must be <= churn_max_noise_level")
        return normalized

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | None = None,
        *,
        device: str,
        no_compile: bool = False,
        **kwargs,
    ) -> "EDMRunner":
        path = checkpoint_path or cls.default_checkpoint_path()
        checkpoint = torch.load(path, map_location=device)
        model, sigma_data = cls.load_model_from_checkpoint(
            checkpoint,
            device,
            no_compile,
        )
        defaults = {
            "sampling_steps": 18,
            "sampler": "edm_stochastic",
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
        }
        defaults.update(checkpoint.get("sampling_defaults", {}))
        defaults.update(kwargs)
        flat = {key: defaults.pop(key) for key in FLAT_EDM_PARAM_KEYS & set(defaults)}
        flat = {key: value for key, value in flat.items() if value is not None}
        if flat:
            if defaults["sampler"] != "edm_stochastic":
                raise ValueError(
                    "flat EDM S_* parameters are only valid for edm_stochastic, "
                    f"got sampler={defaults['sampler']!r}"
                )
            if defaults.get("sampler_params"):
                raise ValueError("Use either sampler_params or flat S_* parameters, not both")
            defaults["sampler_params"] = flat
        return cls(
            model=model,
            target_spec=checkpoint.get(
                "target_spec",
                cls.target_adapter.get_target_distribution_spec(),
            ),
            data_mean=torch.as_tensor(
                checkpoint["data_mean"], device=device, dtype=torch.float32
            ),
            data_std=torch.as_tensor(
                checkpoint["data_std"], device=device, dtype=torch.float32
            ),
            sigma_data=sigma_data,
            device=device,
            checkpoint_path=path,
            **defaults,
        )

    def with_sampling_config(self, **kwargs) -> "EDMRunner":
        allowed = {
            "sampler",
            "sampling_steps",
            "sigma_min",
            "sigma_max",
            "rho",
            "sampler_params",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.with_sampling_config got unknown keys: {sorted(unknown)}"
            )
        sampler = str(kwargs.get("sampler", self._sampler))
        sampler_params = kwargs.get("sampler_params")
        if sampler_params is None and sampler == self._sampler:
            sampler_params = self._sampler_params
        return type(self)(
            model=self._model,
            target_spec=self._target_spec,
            data_mean=self._data_mean,
            data_std=self._data_std,
            sampling_steps=int(kwargs.get("sampling_steps", self._sampling_steps)),
            sampler=sampler,
            sigma_min=float(kwargs.get("sigma_min", self._sigma_min)),
            sigma_max=float(kwargs.get("sigma_max", self._sigma_max)),
            rho=float(kwargs.get("rho", self._rho)),
            sampler_params=sampler_params,
            sigma_data=self._sigma_data,
            device=self._device,
            checkpoint_path=self._checkpoint_path,
        )

    @property
    def input_dim(self) -> int:
        return int(type(self).model_input_dim(self._model))

    def sampling_cache_key(self) -> Mapping[str, Any]:
        return {
            "sampler": str(self._sampler),
            "sampling_steps": int(self._sampling_steps),
            "sigma_min": float(self._sigma_min),
            "sigma_max": float(self._sigma_max),
            "rho": float(self._rho),
            "sampler_params": dict(self._sampler_params),
        }

    @property
    def start_time(self) -> Any:
        return float(self._sigma_max)

    @property
    def end_time(self) -> Any:
        return 0.0

    def sample_prior(self, num_samples: int, *, generator=None) -> torch.Tensor:
        return torch.randn(
            int(num_samples), self.input_dim, device=self._device, generator=generator
        ) * float(self._sigma_max)

    def sample_segment(
        self, x, start_time, end_time, *, generator=None, progress_callback=None
    ):
        return SAMPLE_SEGMENT_FNS[self._sampler](
            self._model,
            x,
            float(start_time),
            float(end_time),
            self._schedule,
            **self._sampler_params,
            generator=generator,
            progress_callback=progress_callback,
        )

    def postprocess_samples(self, native_samples: torch.Tensor) -> torch.Tensor:
        return type(self).target_adapter.denormalize(
            native_samples,
            self._data_mean,
            self._data_std,
        )

    def resolve_split_percentages(self, split_percentages: Sequence[float]):
        return _resolve_split_percentages(self._schedule, split_percentages)

    def segment_cost(self, start_time: Any, end_time: Any) -> float:
        start_idx, end_idx = _schedule_segment_indices(
            start_time, end_time, self._schedule
        )
        if self._sampler == "sde_euler_maruyama":
            return float(end_idx - start_idx)
        return float(
            sum(
                1 if self._schedule.sigmas_cpu[idx + 1] <= 0.0 else 2
                for idx in range(start_idx, end_idx)
            )
        )

    def _split_point(self, point):
        return float(point)

    def parse_solver_baseline_name(self, name: str, solver: str) -> dict | None:
        prefix = f"{solver}_"
        rest = name[len(prefix) :].split("_") if name.startswith(prefix) else []
        has_churn = (
            len(rest) == 1 and rest[0].startswith("churn")
        ) or (
            len(rest) == 2 and rest[1].startswith("churn")
        )
        if solver == "dpmpp_2s":
            if not has_churn:
                parsed = super().parse_solver_baseline_name(name, solver)
                if parsed is not None:
                    return parsed
        elif solver == "edm_stochastic":
            if name == solver:
                return self._solver_baseline_spec(name, solver)
        else:
            return super().parse_solver_baseline_name(name, solver)

        if not name.startswith(prefix):
            return None
        if len(rest) == 1 and rest[0].startswith("churn") and rest[0] != "churn":
            step_token = None
            churn_token = rest[0]
        elif len(rest) == 2 and rest[1].startswith("churn") and rest[1] != "churn":
            step_token = rest[0]
            churn_token = rest[1]
        else:
            return None
        try:
            steps = None if step_token is None else int(step_token)
            churn = float(churn_token[5:])
        except ValueError as exc:
            raise ValueError(
                f"baseline {name!r} has bad {solver} syntax"
            ) from exc
        if steps is not None and steps < 1:
            raise ValueError(f"baseline {name!r} must use steps >= 1")
        if churn < 0.0:
            raise ValueError(f"baseline {name!r} has negative churn")
        if solver == "edm_stochastic":
            sampler_params = {**EDM_STOCHASTIC_DEFAULT_PARAMS, "S_churn": churn}
        else:
            sampler_params = {
                **DPMPP_2S_DEFAULT_PARAMS,
                "stochastic_churn_rate": churn,
            }
        return self._solver_baseline_spec(
            name,
            solver,
            steps,
            sampler_params=sampler_params,
        )

    def solver_cost(self, solver: str, **solver_kwargs) -> float:
        self._validate_solver_kwargs(solver, solver_kwargs)
        sampling_steps = int(solver_kwargs.get("sampling_steps", self._sampling_steps))
        return float(2 * sampling_steps - 1)

    def solver_cache_key(self, solver: str, **solver_kwargs) -> Mapping[str, Any]:
        self._validate_solver_kwargs(solver, solver_kwargs)
        key: dict[str, Any] = {}
        if "sampling_steps" in solver_kwargs:
            key["sampling_steps"] = int(solver_kwargs["sampling_steps"])
        sampler_params = self._solver_sampler_params(solver, solver_kwargs)
        if sampler_params:
            key["sampler_params"] = sampler_params
        for name in sorted(set(solver_kwargs) - {"sampling_steps", "sampler_params", "eta"}):
            key[name] = solver_kwargs[name]
        return key

    def _solver_sampler_params(
        self,
        solver: str,
        solver_kwargs: Mapping[str, Any],
    ) -> dict[str, float]:
        return self._normalize_sampler_params(solver, solver_kwargs.get("sampler_params"))

    def _validate_solver_kwargs(
        self, solver: str, solver_kwargs: Mapping[str, Any]
    ) -> None:
        if solver not in type(self).supported_solvers:
            raise ValueError(f"Unknown solver {solver!r}")
        allowed = {
            "sampling_steps",
            "sigma_min",
            "sigma_max",
            "rho",
            "sampler_params",
            "eta",
        }
        unknown = set(solver_kwargs) - allowed
        if unknown:
            raise ValueError(f"Unknown solver kwargs for {solver!r}: {sorted(unknown)}")
        self._solver_sampler_params(solver, solver_kwargs)

    def schedule_cost(self) -> float:
        return self.segment_cost(self.start_time, self.end_time)

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
        self._validate_solver_kwargs(solver, solver_kwargs)
        sampler_params = self._solver_sampler_params(solver, solver_kwargs)
        runner = self.with_sampling_config(
            sampler=solver,
            sampling_steps=int(
                solver_kwargs.get("sampling_steps", self._sampling_steps)
            ),
            sigma_min=float(solver_kwargs.get("sigma_min", self._sigma_min)),
            sigma_max=float(solver_kwargs.get("sigma_max", self._sigma_max)),
            rho=float(solver_kwargs.get("rho", self._rho)),
            sampler_params=sampler_params,
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

        if method != "edm_samples":
            raise ValueError(
                f"{comparison_mode} reference_generation_config must set method='edm_samples'"
            )
        required = {
            "method",
            "sampler",
            "sampling_steps",
            "sigma_min",
            "sigma_max",
            "rho",
            "sampler_params",
        }
        missing = sorted(required - set(cfg))
        if missing:
            raise ValueError(
                f"reference_generation_config missing required keys: {missing}"
            )
        allowed = required | {"seed"}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown reference_generation_config keys: {sorted(unknown)}"
            )
        sampler = str(cfg["sampler"])
        if sampler not in type(self).supported_samplers:
            raise ValueError(
                f"Unknown EDM reference sampler {sampler!r}. "
                f"Available: {tuple(type(self).supported_samplers)}"
            )
        schedule_config = self._normalize_schedule_config(
            sampling_steps=cfg["sampling_steps"],
            sigma_min=cfg["sigma_min"],
            sigma_max=cfg["sigma_max"],
            rho=cfg["rho"],
        )
        sampler_params = self._normalize_sampler_params(sampler, dict(cfg["sampler_params"]))
        seed = int(cfg.get("seed", 0))
        if seed < 0:
            raise ValueError("reference seed must be nonnegative")
        return {
            "method": "edm_samples",
            "sampler": sampler,
            **schedule_config,
            "sampler_params": sampler_params,
            "seed": seed,
        }

    def reference_cache_key(
        self,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.comparison_mode_spec(comparison_mode)
        ref_cfg = self.normalize_reference_generation_config(
            comparison_mode, reference_generation_config
        )
        base = {
            "runner": self.runner_name,
            "comparison_mode": comparison_mode,
            "target_spec": self._target_spec,
        }
        if comparison_mode != "true_dist":
            base["reference_generation_config"] = dict(ref_cfg)
        if comparison_mode == "edm_samples":
            fingerprint = checkpoint_fingerprint(self._checkpoint_path)
            if fingerprint is not None:
                base["checkpoint_fingerprint"] = fingerprint
        return base

    def generate_reference_samples(
        self,
        *,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any],
        num_samples: int,
        batch_size: int,
        generator=None,
        progress=None,
    ):
        self.comparison_mode_spec(comparison_mode)
        if comparison_mode == "true_dist":
            raise ValueError("true_dist comparison does not require reference samples")
        ref_cfg = self.normalize_reference_generation_config(
            comparison_mode, reference_generation_config
        )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if comparison_mode == "edm_samples":
            ref_runner = self.with_sampling_config(
                sampler=ref_cfg["sampler"],
                sampling_steps=ref_cfg["sampling_steps"],
                sigma_min=ref_cfg["sigma_min"],
                sigma_max=ref_cfg["sigma_max"],
                rho=ref_cfg["rho"],
                sampler_params=ref_cfg["sampler_params"],
            )
            return ref_runner._sample_reference(
                num_samples,
                batch_size,
                sample_dim=ref_runner.input_dim,
                num_steps=ref_runner._sampling_steps,
                generator=generator,
                progress=progress,
            )
        batches = []
        remaining = int(num_samples)
        with torch.inference_mode():
            while remaining > 0:
                current = min(int(batch_size), remaining)
                generated = type(self).target_adapter.sample_target_spec(
                    self._target_spec, current, self._device
                )
                batches.append(generated.to(device="cpu", dtype=torch.float32))
                remaining -= current
                if progress is not None:
                    progress["batch"](1)
        return torch.cat(batches, dim=0)
