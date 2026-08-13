import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
from decimal import Decimal, DecimalException, ROUND_FLOOR, localcontext
from numbers import Real
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from tqdm import tqdm

from adaptive import (
    CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM,
    SUPPORTED_OPTIMIZATION_MODES,
    _crossfit_q_normalize_mlp_params,
    _crossfit_q_normalize_mlp_run_parallelism,
    _normalize_query_params,
    run_estimate_and_sample,
)
from baselines import (
    run_fixed_N_sampling,
    run_solver_baseline_sampling,
    run_uniform_c_sampling,
)
from metrics import (
    aggregate_cached_metric_rows,
    metric_cache_key,
    normalize_metric_params,
    normalize_metrics,
    prepare_metric_states,
    uses_reference_samples,
    validate_metric_dimensions,
)
from reference_cache import (
    load_reference_samples_for_runner,
    reference_samples_path_for_key,
)
from runners.base import (
    BASELINE_STEP_SCHEDULES,
    SamplingStepSchedule,
    iter_budget_resolved_sampling_configs,
    iter_budget_resolved_sampling_config_specs,
    step_schedules_for_sampler,
    steps_for_budget,
)
from runners.common_configs import uniform_c_allowed
from runners.registry import get_runner_class, names
from runners.splitting import normalize_max_sampling_batch_size
from trials import mean_vector, std_vector
from utils import validate_split_percentages

log = logging.getLogger("compare")

_FILE_IDENTITY_CACHE: Dict[str, Dict[str, Any]] = {}

CSV_FIELDS = [
    "mode",
    "sampler",
    "sampling_steps",
    "sampling_params",
    "step_schedule",
    "B",
    "B1",
    "B1_spec",
    "crossfit_q_mlp_loss",
    "reuse",
    "optimizer",
    "solver",
    "solver_steps",
    "solver_params",
    "nfe_per_sample",
    "n0",
    "oracle_gap",
    "mean_mmd",
    "std_mmd",
    "n_valid_mmd",
    "mean_ks",
    "std_ks",
    "n_valid_ks",
    "N_i",
    "N_i_std",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a config-driven adaptive splitting comparison sweep"
    )
    parser.add_argument("--runner", required=True, choices=names())
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_runs", type=int, default=None)
    parser.add_argument("--n_parallel", type=int, default=None)
    parser.add_argument("--crossfit_q_mlp_run_parallelism", type=int, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _resolve_b1_entry(value, budget: int, *, config_key: str = "B1"):
    """Resolve one absolute, ratio, or power-law B1 config entry."""
    budget_value = int(budget)
    if budget_value < 1:
        raise ValueError(f"B must be positive, got {budget!r}")

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"{config_key} power entry must have format 'C,alpha', got {value!r}"
            )
        try:
            coefficient = Decimal(parts[0])
            exponent = Decimal(parts[1])
        except DecimalException as exc:
            raise ValueError(
                f"{config_key} power entry must contain decimal C and alpha, "
                f"got {value!r}"
            ) from exc
        if not coefficient.is_finite() or coefficient <= 0:
            raise ValueError(
                f"{config_key} power coefficient C must be finite and > 0, "
                f"got {parts[0]!r}"
            )
        if not exponent.is_finite() or not Decimal(0) <= exponent <= Decimal(1):
            raise ValueError(
                f"{config_key} power exponent alpha must be finite and in [0, 1], "
                f"got {parts[1]!r}"
            )
        try:
            with localcontext() as context:
                context.prec = 50
                decimal_budget = coefficient * (
                    Decimal(budget_value) ** exponent
                )
        except DecimalException as exc:
            raise ValueError(
                f"could not resolve {config_key} power entry {value!r} for "
                f"B={budget_value}"
            ) from exc
        if not decimal_budget.is_finite():
            raise ValueError(
                f"{config_key} power entry {value!r} is not finite for B={budget_value}"
            )
        resolved = int(decimal_budget.to_integral_value(rounding=ROUND_FLOOR))
        if resolved < 1:
            raise ValueError(
                f"{config_key} power entry {value!r} resolves to {resolved} for "
                f"B={budget_value}; increase C, alpha, or the budget"
            )
        spec = (
            f"power:{_canonical_decimal(coefficient)},"
            f"{_canonical_decimal(exponent)}"
        )
        return resolved, spec

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"{config_key} entry must be a real number or 'C,alpha', got {value!r}"
        )

    decimal_value = Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"{config_key} entry must be finite, got {value!r}")
    if decimal_value <= 0:
        raise ValueError(f"{config_key} entry must be positive, got {value!r}")

    if decimal_value < 1:
        resolved = int(
            (decimal_value * Decimal(budget_value)).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        if resolved < 1:
            raise ValueError(
                f"{config_key} fraction {value!r} resolves to {resolved} for "
                f"B={budget_value}; increase the fraction or budget"
            )
        return resolved, f"ratio:{_canonical_decimal(decimal_value)}"

    if decimal_value != decimal_value.to_integral_value():
        raise ValueError(
            f"absolute {config_key} entry must be an integer, got {value!r}"
        )
    resolved = int(decimal_value)
    return resolved, f"absolute:{resolved}"


def _resolve_b1_for_budget(value, budget: int, *, config_key: str = "B1") -> int:
    return _resolve_b1_entry(value, budget, config_key=config_key)[0]


def _validate_b1_lists(cfg: Mapping[str, Any]) -> None:
    for budget in cfg["B_list"]:
        for config_key in ("B1_list",):
            for value in cfg.get(config_key, []):
                _resolve_b1_for_budget(value, int(budget), config_key=config_key)


def _resolve_unique_b1_entries(values, budget: int, *, config_key: str):
    entries = []
    seen_specs = set()
    for value in values:
        resolved, spec = _resolve_b1_entry(value, budget, config_key=config_key)
        if spec in seen_specs:
            log.info(
                "Skipping duplicate %s entry %s for B=%s",
                config_key,
                value,
                budget,
            )
            continue
        seen_specs.add(spec)
        entries.append((resolved, spec))
    return entries


def _validate_config(cfg: dict, *, runner, config_name: str):
    required = {
        "comparison_mode",
        "sampling_configs",
        "B_list",
        "B1_list",
        "baselines",
        "split_percentages_list",
        "n_runs",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"Config {config_name!r} missing required keys: {missing}")
    mode_names = {spec.name for spec in runner.comparison_modes()}
    if cfg["comparison_mode"] not in mode_names:
        raise ValueError(
            f"comparison_mode {cfg['comparison_mode']!r} not supported; available {sorted(mode_names)}"
        )
    mode_spec = next(
        s for s in runner.comparison_modes() if s.name == cfg["comparison_mode"]
    )
    cfg["metrics"] = normalize_metrics(
        cfg.get("metrics"), supported=getattr(runner, "supported_metrics", ("ks",))
    )
    validate_metric_dimensions(runner, cfg["metrics"])
    cfg["metric_params"] = normalize_metric_params(cfg.get("metric_params"))
    extra_metric_params = sorted(set(cfg["metric_params"]) - set(cfg["metrics"]))
    if extra_metric_params:
        raise ValueError(
            f"metric_params configured for inactive metrics: {extra_metric_params}"
        )
    cfg["primary_metric"] = str(cfg.get("primary_metric", cfg["metrics"][0])).lower()
    if cfg["primary_metric"] not in cfg["metrics"]:
        raise ValueError("primary_metric must be one of metrics")
    if "mmd" in cfg["metrics"] and not mode_spec.requires_reference_cache:
        raise ValueError(
            "mmd requires a comparison_mode backed by cached reference samples; "
            f"comparison_mode {cfg['comparison_mode']!r} is analytic"
        )
    if mode_spec.requires_reference_cache and uses_reference_samples(cfg["metrics"]):
        if "num_base_samples" not in cfg:
            raise ValueError(
                "num_base_samples is required when a metric uses cached reference samples"
            )
        if "reference_generation_config" not in cfg:
            raise ValueError(
                "reference_generation_config is required when comparison_mode "
                f"{cfg['comparison_mode']!r} requires cached reference samples"
            )
        cfg["reference_generation_config"] = dict(
            runner.normalize_reference_generation_config(
                cfg["comparison_mode"], cfg["reference_generation_config"]
            )
        )
    cifar_reference_metrics = {"ks", "mmd"}.intersection(cfg["metrics"])
    if (
        cfg["comparison_mode"] == "cifar10_dataset_samples"
        and cifar_reference_metrics
    ):
        if int(cfg.get("num_base_samples", 0)) != 50_000:
            raise ValueError(
                "CIFAR reference-sample metrics require num_base_samples=50000 "
                "(the complete CIFAR-10 training distribution)"
            )
        reference_cfg = cfg.get("reference_generation_config") or {}
        if (
            reference_cfg.get("method") != "torchvision_cifar10"
            or reference_cfg.get("split") != "train"
        ):
            raise ValueError(
                "CIFAR reference-sample metrics require the torchvision "
                "CIFAR-10 training set"
            )
    cfg["query_params"] = _normalize_query_params(cfg.get("query_params"))
    cfg["crossfit_q_mlp_params"] = _crossfit_q_normalize_mlp_params(
        cfg.get("crossfit_q_mlp_params")
    )
    cfg["crossfit_q_mlp_losses"] = _normalize_crossfit_q_mlp_losses(
        cfg.get("crossfit_q_mlp_losses"), cfg["crossfit_q_mlp_params"]
    )
    cfg["crossfit_q_mlp_run_parallelism"] = _crossfit_q_normalize_mlp_run_parallelism(
        cfg.get("crossfit_q_mlp_run_parallelism")
    )
    optimization_modes = _normalize_optimization_modes(
        cfg.get("optimization_modes", ["monotone"])
    )
    unknown = set(optimization_modes) - SUPPORTED_OPTIMIZATION_MODES
    if unknown:
        raise ValueError(f"Unknown optimization_modes: {sorted(unknown)}")
    cfg["optimization_modes"] = optimization_modes
    if int(cfg["n_runs"]) < 1 or int(cfg.get("n_parallel", 1)) < 1:
        raise ValueError("n_runs and n_parallel must be at least 1")
    cfg["max_sampling_batch_size"] = normalize_max_sampling_batch_size(
        cfg.get("max_sampling_batch_size")
    )
    if not cfg["B_list"]:
        raise ValueError("B_list must be non-empty")
    _validate_b1_lists(cfg)
    if not cfg["sampling_configs"]:
        raise ValueError("sampling_configs must be non-empty")
    for sampling_config in iter_budget_resolved_sampling_configs(cfg):
        runner.with_sampling_config(**dict(sampling_config))
    for split_percentages in cfg["split_percentages_list"]:
        validate_split_percentages([float(x) for x in split_percentages])
    parsed_baselines = [
        runner.parse_baseline_name(str(baseline)) for baseline in cfg["baselines"]
    ]
    for baseline in parsed_baselines:
        if baseline["mode"] != "solver_baseline":
            continue
        for schedule in _solver_step_schedules_for_baseline(cfg, baseline):
            for B in cfg["B_list"]:
                steps_for_budget(
                    schedule,
                    baseline["solver"],
                    int(B),
                    schedule_key=BASELINE_STEP_SCHEDULES,
                )


def _normalize_crossfit_q_mlp_losses(raw_losses, mlp_params):
    if raw_losses is None:
        raise ValueError("crossfit_q_mlp_losses is required")
    if isinstance(raw_losses, str):
        raw_losses = [raw_losses]
    losses = []
    seen = set()
    for raw_loss in raw_losses:
        loss = str(raw_loss).lower()
        params = _crossfit_q_normalize_mlp_params({**dict(mlp_params), "loss": loss})
        loss = str(params["loss"])
        if loss not in seen:
            losses.append(loss)
            seen.add(loss)
    if not losses:
        raise ValueError("crossfit_q_mlp_losses must be non-empty")
    return losses


def _normalize_optimization_modes(raw_modes):
    if isinstance(raw_modes, str):
        raw_modes = [raw_modes]
    modes = []
    seen = set()
    for raw_mode in raw_modes:
        mode = str(raw_mode)
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    if not modes:
        raise ValueError("optimization_modes must be non-empty")
    return modes


def _apply_overrides(cfg: dict, args):
    cfg = dict(cfg)
    if args.n_runs is not None:
        cfg["n_runs"] = int(args.n_runs)
    if args.n_parallel is not None:
        cfg["n_parallel"] = int(args.n_parallel)
    else:
        cfg["n_parallel"] = int(cfg.get("n_parallel", 1))
    if args.crossfit_q_mlp_run_parallelism is not None:
        cfg["crossfit_q_mlp_run_parallelism"] = int(
            args.crossfit_q_mlp_run_parallelism
        )
    return cfg


def _solver_step_schedules_for_baseline(cfg, baseline_spec):
    solver_kwargs = dict(baseline_spec.get("solver_kwargs") or {})
    if "sampling_steps" in solver_kwargs:
        return [
            SamplingStepSchedule(
                name="fixed",
                steps=int(solver_kwargs["sampling_steps"]),
            )
        ]
    return step_schedules_for_sampler(
        cfg,
        baseline_spec["solver"],
        schedule_key=BASELINE_STEP_SCHEDULES,
        allow_default=False,
    )


def _resolve_solver_baseline_for_budget(cfg, baseline_spec, B: int, step_schedule):
    spec = dict(baseline_spec)
    solver_kwargs = dict(spec.get("solver_kwargs") or {})
    if "sampling_steps" not in solver_kwargs:
        steps = steps_for_budget(
            step_schedule,
            spec["solver"],
            int(B),
            schedule_key=BASELINE_STEP_SCHEDULES,
        )
        if steps is not None:
            solver_kwargs["sampling_steps"] = int(steps)
    spec["solver_kwargs"] = solver_kwargs
    spec["step_schedule"] = str(step_schedule.name)
    return spec


def _crossfit_q_loss_options(cfg):
    return _normalize_crossfit_q_mlp_losses(
        cfg.get("crossfit_q_mlp_losses"),
        _crossfit_q_normalize_mlp_params(cfg.get("crossfit_q_mlp_params")),
    )


def _build_trial_specs(cfg, baselines):
    specs: List[Dict[str, Any]] = []
    optimization_modes = cfg.get("optimization_modes", ["monotone"])
    reuse_flags = cfg.get("reuse_flags", [True])
    schedule_baselines = [b for b in baselines if b["mode"] != "solver_baseline"]
    solver_baselines = [b for b in baselines if b["mode"] == "solver_baseline"]

    def make(B, B1, B1_spec, resolved, *, reuse, optimizer, mlp_loss):
        return {
            "mode": "estimate_and_sample",
            "B": int(B),
            "B1": int(B1),
            "B1_spec": B1_spec,
            "sampling_config": dict(resolved.sampling_config),
            "step_schedule": str(resolved.step_schedule),
            "optimization_mode": str(optimizer),
            "reuse_phase1_samples": bool(reuse),
            "crossfit_q_mlp_loss": str(mlp_loss),
        }

    for resolved in iter_budget_resolved_sampling_config_specs(cfg):
        for B1_entry, reuse, optimizer, mlp_loss in itertools.product(
            _resolve_unique_b1_entries(
                cfg["B1_list"], resolved.budget, config_key="B1_list"
            ),
            reuse_flags,
            optimization_modes,
            _crossfit_q_loss_options(cfg),
        ):
            B1, B1_spec = B1_entry
            if B1 >= int(resolved.budget):
                log.info(
                    "Skipping config with resolved B1=%s >= B=%s", B1, resolved.budget
                )
                continue
            specs.append(
                make(resolved.budget, B1, B1_spec, resolved, reuse=reuse,
                     optimizer=optimizer, mlp_loss=mlp_loss)
            )
        for baseline_spec in schedule_baselines:
            specs.append(
                {
                    **baseline_spec,
                    "B": int(resolved.budget),
                    "sampling_config": dict(resolved.sampling_config),
                    "step_schedule": str(resolved.step_schedule),
                }
            )

    for B in cfg["B_list"]:
        for baseline_spec in solver_baselines:
            for step_schedule in _solver_step_schedules_for_baseline(cfg, baseline_spec):
                resolved_baseline = _resolve_solver_baseline_for_budget(
                    cfg, baseline_spec, int(B), step_schedule
                )
                specs.append(
                    {**resolved_baseline, "B": int(B), "sampling_config": None}
                )
    return specs


def _run_trial(
    spec,
    *,
    base_runner,
    comparison_state,
    cfg,
    split_percentages,
    seed,
    n_runs=None,
    run_offset=0,
    return_trial_results=False,
):
    n_runs = int(cfg["n_runs"] if n_runs is None else n_runs)
    runner = (
        base_runner
        if spec["mode"] == "solver_baseline"
        else base_runner.with_sampling_config(**spec["sampling_config"])
    )
    common = dict(
        runner=runner,
        comparison_state=comparison_state,
        comparison_mode=cfg["comparison_mode"],
        metrics=cfg["metrics"],
        B=int(spec["B"]),
        n_runs=n_runs,
        seed=seed,
        debug=bool(cfg.get("debug", False)),
        n_parallel=int(cfg["n_parallel"]),
        run_offset=run_offset,
        return_trial_results=return_trial_results,
        max_sampling_batch_size=cfg.get("max_sampling_batch_size"),
    )
    if spec["mode"] == "solver_baseline":
        return run_solver_baseline_sampling(
            **common,
            solver=spec["solver"],
            solver_kwargs=spec.get("solver_kwargs", {}),
        )
    if spec["mode"] == "fixed_N":
        return run_fixed_N_sampling(
            **common,
            split_percentages=[],
            N_i_list=[],
        )
    if spec["mode"] == "uniform_c":
        return run_uniform_c_sampling(
            **common,
            split_percentages=split_percentages,
            c=float(spec["c"]),
        )
    if spec["mode"] == "ou_oracle":
        oracle = runner.oracle_definition(split_percentages)
        return run_fixed_N_sampling(
            **common,
            split_percentages=split_percentages,
            N_i_list=oracle["split_factors"],
            result_mode="ou_oracle",
            include_allocation=True,
        )
    crossfit_q_mlp_params = dict(cfg.get("crossfit_q_mlp_params") or {})
    crossfit_q_mlp_params["loss"] = str(spec["crossfit_q_mlp_loss"])
    return run_estimate_and_sample(
        **common,
        B1=int(spec["B1"]),
        split_percentages=split_percentages,
        crossfit_q_mlp_run_parallelism=int(
            cfg.get(
                "crossfit_q_mlp_run_parallelism",
                CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM,
            )
        ),
        crossfit_q_mlp_params=crossfit_q_mlp_params,
        optimization_mode=str(spec["optimization_mode"]),
        reuse_phase1_samples=bool(spec["reuse_phase1_samples"]),
        query_params=dict(cfg.get("query_params") or {}),
    )


def _runs_dir(output_dir):
    return os.path.join(output_dir, "runs")


def _split_tag(split_percentages):
    return "_".join(f"{float(x):g}" for x in split_percentages)


def _csv_path(output_dir, split_percentages):
    return os.path.join(
        output_dir, f"compare_results_{_split_tag(split_percentages)}.csv"
    )


def _outputs_manifest_path(output_dir):
    return os.path.join(output_dir, "compare_outputs.json")


def _file_identity(path):
    abspath = os.path.abspath(path)
    if abspath in _FILE_IDENTITY_CACHE:
        return dict(_FILE_IDENTITY_CACHE[abspath])
    if not os.path.exists(abspath):
        identity: Dict[str, Any] = {"missing_path": abspath}
        _FILE_IDENTITY_CACHE[abspath] = identity
        return dict(identity)

    digest = hashlib.sha256()
    with open(abspath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    identity = {
        "size": int(os.path.getsize(abspath)),
        "sha256": digest.hexdigest(),
    }
    _FILE_IDENTITY_CACHE[abspath] = identity
    return dict(identity)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k != "samples"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(dict(value)), sort_keys=True, separators=(",", ":"))


def _split_sampling_config_fields(sampling_config: Mapping[str, Any] | None):
    if not sampling_config:
        return "", "", "{}"
    values = dict(sampling_config)
    sampler = values.pop("sampler", "")
    sampling_steps = values.pop("sampling_steps", "")
    return sampler, sampling_steps, _compact_json(values)


def _split_solver_fields(spec: Mapping[str, Any]):
    if spec["mode"] != "solver_baseline":
        return "", "", "{}"
    solver_kwargs = dict(spec.get("solver_kwargs") or {})
    solver_steps = solver_kwargs.pop("sampling_steps", "")
    return str(spec["solver"]), solver_steps, _compact_json(solver_kwargs)


def _csv_mode(mode: str) -> str:
    return "adaptive" if mode == "estimate_and_sample" else str(mode)


def _method_display(row: Mapping[str, Any]) -> str:
    mode = row.get("mode")
    if mode == "fixed_N":
        return "fixed_N"
    if mode == "uniform_c":
        factors = row.get("N_i") or []
        return f"uniform_{float(factors[0]):g}" if factors else "uniform_c"
    if mode == "ou_oracle":
        return "ou_oracle"
    if mode == "solver_baseline":
        label = str(row.get("solver") or "solver")
        if row.get("solver_steps") not in (None, ""):
            label += f"/{row['solver_steps']}"
        params = row.get("solver_params") or "{}"
        if params != "{}":
            label += f" {params}"
        return label
    label = f"{'reuse' if row.get('reuse') else 'fresh'}"
    if row.get("crossfit_q_mlp_loss"):
        label += f" loss={row['crossfit_q_mlp_loss']}"
    optimizer = str(row.get("optimizer") or "")
    if optimizer:
        label += f" {optimizer}"
    return label.strip()


def _sampling_display(row: Mapping[str, Any]) -> str:
    sampler = str(row.get("sampler") or "sampling")
    steps = row.get("sampling_steps")
    params = row.get("sampling_params") or "{}"
    schedule = row.get("step_schedule") or "default"
    label = sampler
    if steps not in (None, ""):
        label += f"/{steps}"
    if params != "{}":
        label += f" {params}"
    return f"{label}, schedule={schedule}"


def _canonical_spec_for_cache(spec: Mapping[str, Any], *, runner, split_percentages):
    mode = spec["mode"]
    key: Dict[str, Any] = {
        "mode": mode,
        "B": int(spec["B"]),
    }
    if mode == "solver_baseline":
        solver = str(spec["solver"])
        key["solver"] = solver
        key["solver_kwargs"] = runner.solver_cache_key(
            solver, **spec.get("solver_kwargs", {})
        )
        return key

    if mode == "ou_oracle":
        key["oracle"] = runner.oracle_definition(split_percentages)
        return key

    key["runner_sampling_config"] = runner.sampling_cache_key()
    if mode == "fixed_N":
        return key

    key["split_percentages"] = [float(x) for x in split_percentages]

    if mode == "uniform_c":
        key["c"] = float(spec["c"])
        return key

    if mode == "estimate_and_sample":
        key["B1"] = int(spec["B1"])
        key["B1_spec"] = str(spec["B1_spec"])
        key["optimization_mode"] = str(spec["optimization_mode"])
        key["reuse_phase1_samples"] = bool(spec["reuse_phase1_samples"])
        return key

    raise ValueError(f"Unknown trial mode {mode!r}")


def _config_cache_key(
    args,
    cfg,
    spec,
    *,
    runner,
    reference_runner,
    reference_generation_config,
    split_percentages,
):
    """Everything a cached run depends on.

    `_validate_config` has already checked `comparison_mode` against the runner,
    so the mode spec is guaranteed to resolve.
    """
    mode_spec = next(
        s for s in runner.comparison_modes() if s.name == cfg["comparison_mode"]
    )
    key = {
        "runner": args.runner,
        "checkpoint": _file_identity(runner.checkpoint_path),
        "comparison_mode": cfg["comparison_mode"],
        "runner_target": _json_safe(runner.target_spec),
        "metric_config": _json_safe(
            metric_cache_key(runner, cfg["metrics"], cfg.get("metric_params") or {})
        ),
        "spec": _canonical_spec_for_cache(
            spec, runner=runner, split_percentages=split_percentages
        ),
    }
    if mode_spec.requires_reference_cache and uses_reference_samples(cfg["metrics"]):
        ref_key = reference_runner.reference_cache_key(
            cfg["comparison_mode"], reference_generation_config
        )
        key["reference_cache_key"] = _json_safe(dict(ref_key))
        key["num_base_samples"] = int(cfg["num_base_samples"])
        key["reference_samples_path"] = _file_identity(
            reference_samples_path_for_key(reference_runner.checkpoint_path, ref_key)
        )
    if spec["mode"] == "estimate_and_sample":
        key["query_params"] = _json_safe(_normalize_query_params(cfg["query_params"]))
        mlp_params = dict(cfg["crossfit_q_mlp_params"])
        mlp_params["loss"] = str(spec["crossfit_q_mlp_loss"])
        key["crossfit_q_mlp_params"] = _json_safe(
            _crossfit_q_normalize_mlp_params(mlp_params)
        )
    return key


def _config_hash(cache_key):
    return hashlib.sha256(
        json.dumps(
            _json_safe(cache_key), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _write_json_atomic(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    os.replace(tmp, path)


def _runs_jsonl_path(config_dir):
    return os.path.join(config_dir, "runs.jsonl")


def _trial_to_cache_record(spec, trial):
    metrics = dict(trial.get("metrics") or {})
    if "ks" not in metrics and "ks_distance" in trial:
        metrics["ks"] = float(trial["ks_distance"])
    record = {
        "metrics": metrics,
    }
    if "ks" in metrics:
        record["ks_distance"] = float(metrics["ks"])
    if trial.get("metric_payloads"):
        record["metric_payloads"] = trial["metric_payloads"]
    if spec["mode"] in {"estimate_and_sample", "ou_oracle"}:
        record["N_i"] = [float(x) for x in trial["N_i"]]
        record["n0"] = int(trial["n0"])
    if spec["mode"] == "estimate_and_sample":
        record["used_B1"] = int(trial["used_B1"])
    return record


def _load_cached_runs(config_dir, n_runs):
    path = _runs_jsonl_path(config_dir)
    if not os.path.exists(path):
        return {}
    cached: Dict[int, Dict[str, Any]] = {}
    with open(path) as f:
        for run_number, line in enumerate(f, start=1):
            if len(cached) >= n_runs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                cached[len(cached) + 1] = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Ignoring invalid line %s in %s: %s", run_number, path, exc)
    return cached


def _write_cached_runs(config_dir, cached):
    path = _runs_jsonl_path(config_dir)
    os.makedirs(config_dir, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        for _, record in sorted(cached.items()):
            if record is not None:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def _build_comparison_states(base_runner, cfg):
    mode = cfg["comparison_mode"]
    mode_spec = next(
        (s for s in base_runner.comparison_modes() if s.name == mode), None
    )
    if mode_spec is None:
        raise ValueError(
            f"Runner {base_runner.runner_name!r} does not support comparison_mode {mode!r}"
        )
    if not mode_spec.requires_reference_cache or not uses_reference_samples(
        cfg["metrics"]
    ):
        return mode_spec, {
            None: prepare_metric_states(
                base_runner,
                comparison_mode=mode,
                metrics=cfg["metrics"],
                metric_params=cfg.get("metric_params") or {},
            )
        }

    reference_generation_config = base_runner.normalize_reference_generation_config(
        mode, cfg["reference_generation_config"]
    )
    samples = load_reference_samples_for_runner(
        base_runner,
        mode,
        reference_generation_config,
        int(cfg["num_base_samples"]),
    )
    states: Dict[Any, Any] = {
        None: prepare_metric_states(
            base_runner,
            comparison_mode=mode,
            reference_samples=samples,
            metrics=cfg["metrics"],
            metric_params=cfg.get("metric_params") or {},
        )
    }
    return mode_spec, states


def _state_for_spec(states, mode_spec, spec, cfg):
    del mode_spec, spec, cfg
    return states[None]


def _runner_for_spec(base_runner, spec):
    if spec["mode"] == "solver_baseline":
        return base_runner
    return base_runner.with_sampling_config(**spec["sampling_config"])


def _aggregate_cached_runs(spec, trials, *, base_runner, split_percentages, cfg):
    runner = _runner_for_spec(base_runner, spec)
    metric_summary = aggregate_cached_metric_rows(trials, cfg["metrics"])
    if spec["mode"] == "solver_baseline":
        nfe_per_sample = runner.solver_cost(
            spec["solver"], **spec.get("solver_kwargs", {})
        )
    else:
        nfe_per_sample = runner.schedule_cost()
    sampler, sampling_steps, sampling_params = _split_sampling_config_fields(
        spec.get("sampling_config")
    )
    solver, solver_steps, solver_params = _split_solver_fields(spec)
    base = {
        "mode": _csv_mode(spec["mode"]),
        "sampler": sampler,
        "sampling_steps": int(sampling_steps) if sampling_steps != "" else "",
        "sampling_params": sampling_params,
        "step_schedule": str(spec.get("step_schedule", "default")),
        "B": int(spec["B"]),
        "B1": "",
        "B1_spec": "",
        "crossfit_q_mlp_loss": "",
        "reuse": "",
        "optimizer": "",
        "solver": solver,
        "solver_steps": int(solver_steps) if solver_steps != "" else "",
        "solver_params": solver_params,
        "nfe_per_sample": int(nfe_per_sample),
        "n0": "",
        "oracle_gap": "",
        "primary_metric": str(cfg.get("primary_metric", cfg["metrics"][0])),
        **metric_summary,
        "N_i": [],
        "N_i_std": [],
    }
    if spec["mode"] == "fixed_N":
        return base
    if spec["mode"] == "uniform_c":
        factors = [float(spec["c"])] * len(split_percentages)
        return {
            **base,
            "N_i": factors,
            "N_i_std": [0.0] * len(factors),
        }
    if spec["mode"] == "ou_oracle":
        oracle = runner.oracle_definition(split_percentages)
        factors = [float(value) for value in oracle["split_factors"]]
        return {
            **base,
            "n0": int(trials[0]["n0"]) if trials else "",
            "oracle_gap": float(oracle["relative_gap"]),
            "optimizer": "finite_query_minimax",
            "N_i": factors,
            "N_i_std": [0.0] * len(factors),
        }
    if spec["mode"] == "solver_baseline":
        return base
    return {
        **base,
        "B1": int(spec["B1"]),
        "B1_spec": str(spec["B1_spec"]),
        "crossfit_q_mlp_loss": str(spec.get("crossfit_q_mlp_loss") or ""),
        "reuse": bool(spec["reuse_phase1_samples"]),
        "optimizer": str(spec.get("optimization_mode", "monotone")),
        "N_i": mean_vector([t["N_i"] for t in trials]),
        "N_i_std": std_vector([t["N_i"] for t in trials]),
    }


def _sort_key(row):
    metric = row.get("primary_metric", "ks")
    value = row.get(f"mean_{metric}")
    valid = value is not None and not (
        isinstance(value, float) and np.isnan(value)
    )
    return (
        int(row["B"]),
        str(row.get("sampler", "")),
        str(row.get("sampling_params", "")),
        str(row.get("step_schedule", "")),
        str(row.get("mode", "")),
        str(row.get("solver", "")),
        int(row.get("solver_steps") or 0),
        str(row.get("solver_params", "")),
        not valid,
        value if valid else 0.0,
        str(row.get("B1_spec", "")),
        int(row["B1"] or 0),
        str(row.get("crossfit_q_mlp_loss", "")),
        str(row.get("reuse", "")),
        str(row.get("optimizer", "")),
    )


def _build_summary_rows(records):
    rows = []
    for r in sorted(records, key=_sort_key):
        row = {field: r.get(field, "") for field in CSV_FIELDS}
        row["N_i"] = ",".join(f"{float(x):.6g}" for x in (r.get("N_i") or []))
        row["N_i_std"] = ",".join(f"{float(x):.6g}" for x in (r.get("N_i_std") or []))
        rows.append(row)
    return rows


def _write_summary_csv(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _run_split(args, cfg, base_runner, baselines, split_percentages):
    split_percentages = [float(x) for x in split_percentages]
    selected_baselines = []
    for baseline in baselines:
        if baseline["mode"] == "uniform_c" and not uniform_c_allowed(
            baseline["c"], len(split_percentages)
        ):
            log.info(
                "Skipping uniform_c=%g for %d split points",
                baseline["c"],
                len(split_percentages),
            )
            continue
        selected_baselines.append(baseline)
    specs = _build_trial_specs(cfg, selected_baselines)
    runs_dir = _runs_dir(args.output_dir)
    entries: List[Dict[str, Any]] = []
    total_remaining = 0
    target_runs = int(cfg["n_runs"])
    mode_spec = next(
        (s for s in base_runner.comparison_modes() if s.name == cfg["comparison_mode"]),
        None,
    )
    if mode_spec is None:
        raise ValueError(
            f"Runner {base_runner.runner_name!r} does not support comparison_mode {cfg['comparison_mode']!r}"
        )
    reference_generation_config = None
    if mode_spec.requires_reference_cache and uses_reference_samples(cfg["metrics"]):
        reference_generation_config = base_runner.normalize_reference_generation_config(
            cfg["comparison_mode"], cfg["reference_generation_config"]
        )
    for spec in specs:
        runner_for_key = _runner_for_spec(base_runner, spec)
        cache_key = _config_cache_key(
            args,
            cfg,
            spec,
            runner=runner_for_key,
            reference_runner=base_runner,
            reference_generation_config=reference_generation_config,
            split_percentages=split_percentages,
        )
        digest = _config_hash(cache_key)
        config_id = digest[:16]
        seed = int(config_id, 16) % (2**63 - 1)
        config_dir = os.path.join(runs_dir, config_id)
        cached = _load_cached_runs(config_dir, target_runs)
        completed_runs = min(len(cached), target_runs)
        remaining_runs = target_runs - completed_runs
        total_remaining += remaining_runs
        entries.append(
            {
                "spec": spec,
                "seed": seed,
                "cache_key": cache_key,
                "config_id": config_id,
                "config_dir": config_dir,
                "cached": cached,
                "completed_runs": completed_runs,
                "remaining_runs": remaining_runs,
            }
        )

    if total_remaining:
        mode_spec, comparison_states = _build_comparison_states(base_runner, cfg)
        for entry in tqdm(
            entries, desc=f"Compare split {_split_tag(split_percentages)}"
        ):
            remaining_runs = int(entry["remaining_runs"])
            if remaining_runs <= 0:
                continue
            spec = entry["spec"]
            config_dir = entry["config_dir"]
            _write_json_atomic(
                os.path.join(config_dir, "config.json"),
                {
                    "run_id": entry["config_id"],
                    "cache_key": entry["cache_key"],
                },
            )
            if spec["mode"] == "ou_oracle":
                oracle_runner = _runner_for_spec(base_runner, spec)
                _write_json_atomic(
                    os.path.join(config_dir, "oracle.json"),
                    oracle_runner.oracle_definition(split_percentages),
                )
            comparison_state = _state_for_spec(comparison_states, mode_spec, spec, cfg)

            completed_runs = int(entry["completed_runs"])
            try:
                result = _run_trial(
                    spec,
                    base_runner=base_runner,
                    comparison_state=comparison_state,
                    cfg=cfg,
                    split_percentages=split_percentages,
                    seed=entry["seed"],
                    n_runs=remaining_runs,
                    run_offset=completed_runs,
                    return_trial_results=True,
                )
            except Exception as exc:
                log.warning(
                    "Failed config %s with %s remaining runs: %s",
                    entry["config_id"],
                    remaining_runs,
                    exc,
                )
                continue
            trial_results = result.get("trial_results") or []
            if len(trial_results) != remaining_runs:
                log.warning(
                    "Config %s returned %d/%d remaining runs; caching returned runs",
                    entry["config_id"],
                    len(trial_results),
                    remaining_runs,
                )
            for local_idx, trial in enumerate(trial_results[:remaining_runs]):
                run_number = completed_runs + local_idx + 1
                entry["cached"][run_number] = _trial_to_cache_record(spec, trial)
            _write_cached_runs(config_dir, entry["cached"])
    else:
        log.info(
            "All requested runs already cached for split %s",
            _split_tag(split_percentages),
        )

    records: List[Dict[str, Any]] = []
    completed = 0
    for entry in entries:
        cached = _load_cached_runs(entry["config_dir"], target_runs)
        completed += len(cached)
        trials = [cached[r] for r in sorted(cached)]
        records.append(
            _aggregate_cached_runs(
                entry["spec"],
                trials,
                base_runner=base_runner,
                split_percentages=split_percentages,
                cfg=cfg,
            )
        )

    csv_output = _csv_path(args.output_dir, split_percentages)
    _write_summary_csv(csv_output, _build_summary_rows(records))
    print(f"Saved CSV summary to {csv_output}")
    print(f"Completed {completed}/{len(specs) * target_runs} cached runs")
    print(f"Attempted {total_remaining} remaining runs")

    grouped: Dict[Any, Dict[str, Any]] = {}
    primary_metric = str(cfg.get("primary_metric", cfg["metrics"][0]))
    primary_field = f"mean_{primary_metric}"
    for r in records:
        value = r.get(primary_field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        key = (
            int(r["B"]),
            str(r.get("sampler", "")),
            str(r.get("sampling_params", "{}")),
            str(r.get("step_schedule", "default")),
        )
        if key not in grouped or r[primary_field] < grouped[key][primary_field]:
            grouped[key] = r
    for (_, _, _, _), best in sorted(grouped.items()):
        print(
            f"  B={best['B']}, {_sampling_display(best)}: best={_method_display(best)} "
            f"(B1={best.get('B1', '')}) {primary_field}={best[primary_field]:.6f}"
        )
    return csv_output


def main():
    args = parse_args()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    runner_cls = get_runner_class(args.runner)
    cfg = _apply_overrides(runner_cls.get_config(args.config), args)
    base_runner = runner_cls.load_from_checkpoint(
        device=args.device,
        no_compile=args.no_compile,
        **dict(cfg.get("runner_defaults") or {}),
    )
    cfg["debug"] = bool(args.debug)
    _validate_config(cfg, runner=base_runner, config_name=args.config)
    baselines = [
        base_runner.parse_baseline_name(str(name)) for name in cfg["baselines"]
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    _write_json_atomic(
        os.path.join(args.output_dir, "sweep.json"),
        {"runner_name": args.runner, "config_name": args.config, "config": cfg},
    )

    csv_outputs = []
    for split_percentages in cfg["split_percentages_list"]:
        csv_outputs.append(
            _run_split(args, cfg, base_runner, baselines, split_percentages)
        )
    manifest_path = _outputs_manifest_path(args.output_dir)
    _write_json_atomic(
        manifest_path,
        {
            "runner_name": args.runner,
            "config_name": args.config,
            "output_dir": args.output_dir,
            "csv_files": csv_outputs,
        },
    )
    print(f"Saved compare output manifest to {manifest_path}")
    print(f"Saved compact run cache to {_runs_dir(args.output_dir)}")


if __name__ == "__main__":
    main()
