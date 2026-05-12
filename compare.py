import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from tqdm import tqdm

from adaptive import run_estimate_and_sample
from baselines import run_fixed_N_sampling, run_solver_baseline_sampling
from reference_cache import (
    load_reference_samples_for_runner,
    reference_samples_path_for_key,
)
from runners.registry import get_runner_class, names
from trials import mean_vector, std_vector
from utils import validate_split_percentages

log = logging.getLogger("compare")

SUPPORTED_SIGMA_ESTIMATION_MODES = {"pilot_tree", "independent"}
_FILE_IDENTITY_CACHE: Dict[str, Dict[str, Any]] = {}

CSV_FIELDS = [
    "method_label",
    "B",
    "B1",
    "sampling_label",
    "nfe_per_sample",
    "mean_ks",
    "std_ks",
    "N_i",
    "N_i_std",
    "n_valid_runs",
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
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _parse_baseline_name(name: str, supported_solvers):
    if name == "fixed_N":
        return {"mode": "fixed_N", "method_label": "all_ones_baseline"}
    for solver in sorted(supported_solvers, key=len, reverse=True):
        prefix = f"{solver}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :].split("_")
        if len(rest) not in (1, 2):
            break
        try:
            steps = int(rest[0])
        except ValueError as exc:
            raise ValueError(f"baseline {name!r} must use integer steps") from exc
        if steps < 1:
            raise ValueError(f"baseline {name!r} must use steps >= 1")
        solver_kwargs: dict[str, Any] = {"sampling_steps": steps}
        if len(rest) == 2:
            tag = rest[1]
            if not tag.startswith("eta") or tag == "eta":
                break
            try:
                solver_kwargs["eta"] = float(tag[3:])
            except ValueError as exc:
                raise ValueError(f"baseline {name!r} has bad eta") from exc
        return {
            "mode": "solver_baseline",
            "method_label": name,
            "solver": solver,
            "solver_kwargs": solver_kwargs,
        }
    raise ValueError(
        f"Unknown baseline {name!r}. Expected fixed_N, <solver>_<steps>, "
        f"or <solver>_<steps>_eta<eta>"
    )


def _validate_config(cfg: dict, *, runner, config_name: str):
    required = {
        "comparison_mode",
        "sampling_configs",
        "B_list",
        "B1_list",
        "sigma_modes",
        "reuse_flags",
        "baselines",
        "split_percentages_list",
        "observable_config",
        "independent_n2",
        "num_base_samples",
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
    sigma_modes = set(cfg["sigma_modes"])
    if not sigma_modes <= SUPPORTED_SIGMA_ESTIMATION_MODES:
        raise ValueError(
            f"Unknown sigma_modes: {sorted(sigma_modes - SUPPORTED_SIGMA_ESTIMATION_MODES)}"
        )
    if int(cfg["independent_n2"]) < 2:
        raise ValueError("independent_n2 must be at least 2")
    if int(cfg["n_runs"]) < 1 or int(cfg.get("n_parallel", 1)) < 1:
        raise ValueError("n_runs and n_parallel must be at least 1")
    if not cfg["sampling_configs"]:
        raise ValueError("sampling_configs must be non-empty")
    for split_percentages in cfg["split_percentages_list"]:
        validate_split_percentages([float(x) for x in split_percentages])
    parsed_baselines = [
        _parse_baseline_name(str(baseline), set(runner.solver_names()))
        for baseline in cfg["baselines"]
    ]
    if (
        mode_spec.reference_uses_sampling_config
        and any(b["mode"] == "solver_baseline" for b in parsed_baselines)
        and "solver_reference_sampling_config" not in cfg
    ):
        raise ValueError(
            "solver_reference_sampling_config is required when solver baselines are "
            "used with a sampling-config-specific comparison mode"
        )


def _apply_overrides(cfg: dict, args):
    cfg = dict(cfg)
    if args.n_runs is not None:
        cfg["n_runs"] = int(args.n_runs)
    if args.n_parallel is not None:
        cfg["n_parallel"] = int(args.n_parallel)
    else:
        cfg["n_parallel"] = int(cfg.get("n_parallel", 1))
    return cfg


def _build_trial_specs(
    B_list, B1_list, sampling_configs, baselines, sigma_modes, reuse_flags
):
    specs: List[Dict[str, Any]] = []
    schedule_baselines = [b for b in baselines if b["mode"] != "solver_baseline"]
    solver_baselines = [b for b in baselines if b["mode"] == "solver_baseline"]

    for sampling_config in sampling_configs:
        sampling_config = dict(sampling_config)
        for B in B_list:
            for B1, sigma_mode, reuse in itertools.product(
                B1_list, sigma_modes, reuse_flags
            ):
                if int(B1) >= int(B):
                    log.info("Skipping config with B1=%s >= B=%s", B1, B)
                    continue
                specs.append(
                    {
                        "mode": "estimate_and_sample",
                        "B": int(B),
                        "B1": int(B1),
                        "sampling_config": sampling_config,
                        "sigma_estimation_mode": str(sigma_mode),
                        "reuse_phase1_samples": bool(reuse),
                        "method_label": f"{sigma_mode}_{'reuse' if reuse else 'fresh'}",
                    }
                )
            for baseline_spec in schedule_baselines:
                specs.append(
                    {
                        **baseline_spec,
                        "B": int(B),
                        "sampling_config": sampling_config,
                    }
                )
    for B in B_list:
        for baseline_spec in solver_baselines:
            specs.append({**baseline_spec, "B": int(B), "sampling_config": None})
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
        B=int(spec["B"]),
        n_runs=n_runs,
        seed=seed,
        debug=bool(cfg.get("debug", False)),
        n_parallel=int(cfg["n_parallel"]),
        run_offset=run_offset,
        return_trial_results=return_trial_results,
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
            split_percentages=split_percentages,
            N_i_list=[1.0] * len(split_percentages),
        )
    return run_estimate_and_sample(
        **common,
        B1=int(spec["B1"]),
        split_percentages=split_percentages,
        observable_config=cfg["observable_config"],
        independent_n2=int(cfg["independent_n2"]),
        variance_estimation_mode=spec["sigma_estimation_mode"],
        reuse_phase1_samples=bool(spec["reuse_phase1_samples"]),
    )


def _runs_dir(output_dir):
    return os.path.join(output_dir, "runs")


def _split_tag(split_percentages):
    return "_".join(f"{float(x):g}" for x in split_percentages)


def _csv_path(output_dir, split_percentages):
    return os.path.join(
        output_dir, f"compare_results_{_split_tag(split_percentages)}.csv"
    )


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


def _sampling_config_key(sampling_config: Mapping[str, Any]):
    return json.dumps(
        _json_safe(dict(sampling_config)), sort_keys=True, separators=(",", ":")
    )


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

    key["runner_sampling_config"] = runner.sampling_cache_key()
    key["split_percentages"] = [float(x) for x in split_percentages]
    if mode == "fixed_N":
        key["N_i"] = [1.0] * len(split_percentages)
        return key

    if mode == "estimate_and_sample":
        key.update(
            {
                "B1": int(spec["B1"]),
                "sigma_estimation_mode": str(spec["sigma_estimation_mode"]),
                "reuse_phase1_samples": bool(spec["reuse_phase1_samples"]),
            }
        )
        return key

    raise ValueError(f"Unknown trial mode {mode!r}")


def _config_cache_key(args, cfg, spec, *, runner, reference_runner, split_percentages):
    mode_spec = next(
        (s for s in runner.comparison_modes() if s.name == cfg["comparison_mode"]),
        None,
    )
    key = {
        "runner": args.runner,
        "checkpoint": _file_identity(runner.checkpoint_path),
        "comparison_mode": cfg["comparison_mode"],
        "runner_target": _json_safe(runner.target_spec),
        "reference_cache_key": _json_safe(
            dict(reference_runner.reference_cache_key(cfg["comparison_mode"]))
        ),
        "num_base_samples": int(cfg["num_base_samples"]),
        "spec": _canonical_spec_for_cache(
            spec, runner=runner, split_percentages=split_percentages
        ),
    }
    if spec["mode"] == "estimate_and_sample":
        key["observable_config"] = _json_safe(cfg["observable_config"])
        if spec.get("sigma_estimation_mode") == "independent":
            key["independent_n2"] = int(cfg["independent_n2"])

    if mode_spec is None:
        raise ValueError(
            f"Runner {runner.runner_name!r} does not support comparison_mode {cfg['comparison_mode']!r}"
        )
    if mode_spec.requires_reference_cache:
        cache_path = reference_samples_path_for_key(
            reference_runner.checkpoint_path,
            reference_runner.reference_cache_key(cfg["comparison_mode"]),
        )
        key["reference_samples_path"] = _file_identity(cache_path)
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
    record = {
        "ks_distance": float(trial["ks_distance"]),
        "extinct": bool(int(trial.get("leaf_count", 0)) == 0),
    }
    if spec["mode"] == "estimate_and_sample":
        record["N_i"] = [float(x) for x in trial["N_i"]]
        record["n0"] = int(trial["n0"])
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
    if not mode_spec.requires_reference_cache:
        return mode_spec, {
            None: base_runner.prepare_comparison_state(comparison_mode=mode)
        }

    states: Dict[Any, Any] = {}
    if mode_spec.reference_uses_sampling_config:
        seen = set()
        sampling_configs = list(cfg["sampling_configs"])
        if "solver_reference_sampling_config" in cfg:
            sampling_configs.append(cfg["solver_reference_sampling_config"])
        for sampling_config in sampling_configs:
            key = _sampling_config_key(sampling_config)
            if key in seen:
                continue
            seen.add(key)
            runner = base_runner.with_sampling_config(**sampling_config)
            legacy = getattr(runner, "reference_legacy_paths", lambda *_a, **_k: ())(
                mode
            )
            samples = load_reference_samples_for_runner(
                runner, mode, int(cfg["num_base_samples"]), legacy_paths=tuple(legacy)
            )
            states[key] = runner.prepare_comparison_state(
                comparison_mode=mode, reference_samples=samples
            )
    else:
        legacy = getattr(base_runner, "reference_legacy_paths", lambda *_a, **_k: ())(
            mode
        )
        samples = load_reference_samples_for_runner(
            base_runner, mode, int(cfg["num_base_samples"]), legacy_paths=tuple(legacy)
        )
        states[None] = base_runner.prepare_comparison_state(
            comparison_mode=mode, reference_samples=samples
        )
    return mode_spec, states


def _state_for_spec(states, mode_spec, spec, cfg):
    if not mode_spec.requires_reference_cache:
        return states[None]
    if mode_spec.reference_uses_sampling_config:
        sampling_config = (
            cfg["solver_reference_sampling_config"]
            if spec["mode"] == "solver_baseline"
            else spec["sampling_config"]
        )
        return states[_sampling_config_key(sampling_config)]
    return states[None]


def _runner_for_spec(base_runner, spec):
    if spec["mode"] == "solver_baseline":
        return base_runner
    return base_runner.with_sampling_config(**spec["sampling_config"])


def _reference_runner_for_spec(base_runner, cfg, mode_spec, spec):
    if not mode_spec.reference_uses_sampling_config:
        return base_runner
    sampling_config = (
        cfg["solver_reference_sampling_config"]
        if spec["mode"] == "solver_baseline"
        else spec["sampling_config"]
    )
    return base_runner.with_sampling_config(**sampling_config)


def _aggregate_cached_runs(spec, trials, *, base_runner, split_percentages):
    runner = _runner_for_spec(base_runner, spec)
    if not trials:
        values = np.array([], dtype=float)
    else:
        values = np.array(
            [t.get("ks_distance", np.nan) for t in trials],
            dtype=float,
        )
    valid = values[~np.isnan(values)]
    mean_ks = float(valid.mean()) if valid.size else float("nan")
    std_ks = float(valid.std()) if valid.size else float("nan")
    if spec["mode"] == "solver_baseline":
        nfe_per_sample = runner.solver_cost(
            spec["solver"], **spec.get("solver_kwargs", {})
        )
    else:
        nfe_per_sample = runner.schedule_cost()
    base = {
        "method_label": spec.get("method_label", ""),
        "B": int(spec["B"]),
        "B1": "",
        "sampling_label": (
            spec["method_label"]
            if spec["mode"] == "solver_baseline"
            else runner.format_sampling_label()
        ),
        "nfe_per_sample": int(nfe_per_sample),
        "mean_ks": mean_ks,
        "std_ks": std_ks,
        "N_i": [],
        "N_i_std": [],
        "n_valid_runs": int(valid.size),
    }
    if spec["mode"] == "fixed_N":
        return base
    if spec["mode"] == "solver_baseline":
        return base
    return {
        **base,
        "B1": int(spec["B1"]),
        "N_i": mean_vector([t["N_i"] for t in trials]),
        "N_i_std": std_vector([t["N_i"] for t in trials]),
    }


def _sort_key(row):
    mean_ks = row.get("mean_ks")
    valid = mean_ks is not None and not (
        isinstance(mean_ks, float) and np.isnan(mean_ks)
    )
    return (
        int(row["B"]),
        str(row["sampling_label"]),
        not valid,
        mean_ks if valid else 0.0,
        str(row.get("method_label", "")),
        int(row["B1"] or 0),
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
    specs = _build_trial_specs(
        cfg["B_list"],
        cfg["B1_list"],
        cfg["sampling_configs"],
        baselines,
        cfg["sigma_modes"],
        cfg["reuse_flags"],
    )
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
    for spec in specs:
        runner_for_key = _runner_for_spec(base_runner, spec)
        reference_runner_for_key = _reference_runner_for_spec(
            base_runner, cfg, mode_spec, spec
        )
        cache_key = _config_cache_key(
            args,
            cfg,
            spec,
            runner=runner_for_key,
            reference_runner=reference_runner_for_key,
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
            )
        )

    csv_output = _csv_path(args.output_dir, split_percentages)
    _write_summary_csv(csv_output, _build_summary_rows(records))
    print(f"Saved CSV summary to {csv_output}")
    print(f"Completed {completed}/{len(specs) * target_runs} cached runs")
    print(f"Attempted {total_remaining} remaining runs")

    grouped: Dict[Any, Dict[str, Any]] = {}
    for r in records:
        mean_ks = r.get("mean_ks")
        if mean_ks is None or (isinstance(mean_ks, float) and np.isnan(mean_ks)):
            continue
        key = (int(r["B"]), str(r["sampling_label"]))
        if key not in grouped or r["mean_ks"] < grouped[key]["mean_ks"]:
            grouped[key] = r
    for (B, sampling_label), best in sorted(grouped.items()):
        print(
            f"  B={B}, {sampling_label}: best={best.get('method_label')} "
            f"(B1={best.get('B1', '')}) mean ks={best['mean_ks']:.6f}"
        )


def main():
    args = parse_args()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    runner_cls = get_runner_class(args.runner)
    base_runner = runner_cls.load_from_checkpoint(
        device=args.device,
        no_compile=args.no_compile,
    )
    cfg = _apply_overrides(runner_cls.get_config(args.config), args)
    cfg["debug"] = bool(args.debug)
    _validate_config(cfg, runner=base_runner, config_name=args.config)
    baselines = [
        _parse_baseline_name(str(name), set(base_runner.solver_names()))
        for name in cfg["baselines"]
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    _write_json_atomic(
        os.path.join(args.output_dir, "sweep.json"),
        {"runner_name": args.runner, "config_name": args.config, "config": cfg},
    )

    for split_percentages in cfg["split_percentages_list"]:
        _run_split(args, cfg, base_runner, baselines, split_percentages)
    print(f"Saved compact run cache to {_runs_dir(args.output_dir)}")


if __name__ == "__main__":
    main()
