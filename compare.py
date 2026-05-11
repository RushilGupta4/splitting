import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ensure_samples import (
    extract_reference_samples_tensor,
    load_reference_payload,
    reference_samples_path,
    true_reference_samples_path,
)
from infer import (
    DDIM,
    _expected_cost_per_root,
    _load_model_and_stats,
    _mean_scalar,
    _mean_vector,
    _parse_split_percentages,
    _parse_x_grid,
    _resolve_split_percentages,
    _std_vector,
    prepare_reference_cdf_state,
    run_estimate_and_sample,
    run_fixed_N_sampling,
    run_solver_baseline_sampling,
    warm_reference_ks_kernel,
)

log = logging.getLogger("compare")

SUPPORTED_SOLVERS = ("ddim", "dpmpp_2m")
SUPPORTED_SIGMA_ESTIMATION_MODES = ("pilot_tree", "independent")
SUPPORTED_BIAS_TYPES = ("biased", "unbiased")

CSV_FIELDS = [
    "record_id",
    "B",
    "B1",
    "sampling_steps",
    "eta",
    "reference_mode",
    "reference_sampling_steps",
    "reference_eta",
    "solver",
    "solver_sampling_steps",
    "solver_eta",
    "solver_nfe",
    "method_label",
    "mode",
    "sigma_estimation_mode",
    "bias_type",
    "reuse_phase1_samples",
    "N_i",
    "N_i_std",
    "N_i_count",
    "n0",
    "expected_total_samples",
    "used_B1",
    "mean_ks",
    "std_ks",
    "var_ks",
    "n_valid_runs",
    "extinction_rate",
    "is_best_for_pair",
    "error",
]

BEST_RECORD_FIELDS = (
    "record_id",
    "B",
    "sampling_steps",
    "eta",
    "reference_mode",
    "solver",
    "solver_sampling_steps",
    "solver_eta",
    "method_label",
    "bias_type",
    "B1",
    "mean_ks",
    "std_ks",
    "var_ks",
    "N_i",
    "n0",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep step/eta pairs and compare adaptive splitting against an all-ones baseline"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_final.pt")
    parser.add_argument(
        "--B_list",
        type=str,
        required=True,
        help="Comma-separated total budgets, e.g. '1000000,2500000'",
    )
    parser.add_argument(
        "--B1_list",
        type=str,
        required=True,
        help="Comma-separated phase-1 budgets for adaptive methods",
    )
    parser.add_argument(
        "--step_eta_pairs",
        type=str,
        required=True,
        help="Comma-separated 'steps:eta' pairs, e.g. '1000:1.0,500:1.0'",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="fixed_N,dpmpp_2m_100",
        help=(
            "Comma-separated baselines. Use 'fixed_N' for the all-ones fixed_N "
            "baseline, or '<solver>_<steps>' / '<solver>_<steps>_eta<eta>' for a "
            f"solver baseline. Supported solvers: {', '.join(SUPPORTED_SOLVERS)}. "
            "Use '' to disable."
        ),
    )
    parser.add_argument(
        "--sigma_modes",
        type=str,
        default="pilot_tree,independent",
        help=(
            "Comma-separated sigma estimation modes for adaptive methods. "
            f"Supported modes: {', '.join(SUPPORTED_SIGMA_ESTIMATION_MODES)}."
        ),
    )
    parser.add_argument(
        "--bias_types",
        type=str,
        default="unbiased",
        help=(
            "Comma-separated sigma estimator bias types for adaptive methods. "
            f"Supported types: {', '.join(SUPPORTED_BIAS_TYPES)}."
        ),
    )
    parser.add_argument(
        "--reuse_flags",
        type=str,
        default="false,true",
        help="Comma-separated reuse_phase1_samples flags, e.g. 'false,true' or 'true'.",
    )
    parser.add_argument(
        "--reference_mode",
        choices=("true_dist", "true_samples", "ddpm_samples"),
        default="ddpm_samples",
        help=(
            "ddpm_samples uses cached DDPM/DDIM reference samples; true_samples uses "
            "cached target-distribution samples; true_dist uses the exact target CDF."
        ),
    )
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument(
        "--n_parallel",
        type=int,
        default=1,
        help="Number of trials to batch together within each setup.",
    )
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument(
        "--split_percentages",
        type=str,
        default="0.5",
        help="Comma-separated split percentages; each resolves to round(steps * p)",
    )
    parser.add_argument("--independent_n2", type=int, default=10)
    parser.add_argument("--x_grid", type=str, default="-2.0,-1.0,0.0,1.0,2.0")
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_base_samples", type=int, default=1000000)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for compact run cache and derived CSV summary.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile for timing/debug runs.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _parse_int_list(raw: str, name: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"{name} must have at least one value")
    return values


def _parse_step_eta_pairs(raw: str) -> List[Tuple[int, float]]:
    pairs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Each step_eta pair must be 'steps:eta', got '{item}'")
        steps_str, eta_str = item.split(":", 1)
        pairs.append((int(steps_str), float(eta_str)))
    if not pairs:
        raise ValueError("step_eta_pairs must contain at least one pair")
    return pairs


def _parse_baseline_name(name: str) -> Dict[str, Any]:
    if name == "fixed_N":
        return {
            "mode": "fixed_N",
            "method_label": "all_ones_baseline",
        }
    for solver in sorted(SUPPORTED_SOLVERS, key=len, reverse=True):
        prefix = f"{solver}_"
        if not name.startswith(prefix):
            continue

        remainder = name[len(prefix) :]
        parts = remainder.split("_")
        if len(parts) not in (1, 2):
            break

        try:
            solver_sampling_steps = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"Solver baseline '{name}' must use integer sampling steps"
            ) from exc
        if solver_sampling_steps < 1:
            raise ValueError(f"Solver baseline '{name}' must use sampling steps >= 1")

        solver_eta = 0.0
        if len(parts) == 2:
            eta_part = parts[1]
            if not eta_part.startswith("eta") or eta_part == "eta":
                break
            try:
                solver_eta = float(eta_part[3:])
            except ValueError as exc:
                raise ValueError(
                    f"Solver baseline '{name}' must use numeric eta in '_eta<eta>'"
                ) from exc

        return {
            "mode": "solver_baseline",
            "method_label": name,
            "solver": solver,
            "solver_sampling_steps": solver_sampling_steps,
            "solver_eta": solver_eta,
        }

    supported = ", ".join(SUPPORTED_SOLVERS)
    raise ValueError(
        f"Unknown baseline '{name}'. Expected 'fixed_N', '<solver>_<steps>', or "
        f"'<solver>_<steps>_eta<eta>' with supported solvers: {supported}"
    )


def _parse_baselines(raw: str) -> List[Dict[str, Any]]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    return [_parse_baseline_name(name) for name in names]


def _parse_sigma_modes(raw: str) -> List[str]:
    modes = [x.strip() for x in raw.split(",") if x.strip()]
    if not modes:
        raise ValueError("sigma_modes must have at least one value")
    unsupported = [
        mode for mode in modes if mode not in SUPPORTED_SIGMA_ESTIMATION_MODES
    ]
    if unsupported:
        supported = ", ".join(SUPPORTED_SIGMA_ESTIMATION_MODES)
        raise ValueError(
            f"Unsupported sigma_modes value(s): {', '.join(unsupported)}. "
            f"Supported modes: {supported}"
        )
    return modes


def _parse_bias_types(raw: str) -> List[str]:
    bias_types = [x.strip() for x in raw.split(",") if x.strip()]
    if not bias_types:
        raise ValueError("bias_types must have at least one value")
    unsupported = [
        bias_type for bias_type in bias_types if bias_type not in SUPPORTED_BIAS_TYPES
    ]
    if unsupported:
        supported = ", ".join(SUPPORTED_BIAS_TYPES)
        raise ValueError(
            f"Unsupported bias_types value(s): {', '.join(unsupported)}. "
            f"Supported types: {supported}"
        )
    return bias_types


def _parse_bool_list(raw: str, name: str) -> List[bool]:
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n"}
    values = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in truthy:
            values.append(True)
        elif item in falsy:
            values.append(False)
        else:
            raise ValueError(
                f"{name} must contain boolean values like true,false or 1,0; got '{item}'"
            )
    if not values:
        raise ValueError(f"{name} must have at least one value")
    return values


def _unique_step_eta_pairs(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    seen = set()
    unique = []
    for steps, eta in pairs:
        key = (int(steps), float(eta))
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _build_trial_specs(
    B_list: List[int],
    B1_list: List[int],
    step_eta_pairs: List[Tuple[int, float]],
    baselines: List[Dict[str, Any]],
    sigma_modes: List[str],
    bias_types: List[str],
    reuse_flags: List[bool],
) -> List[Dict[str, Any]]:
    """Every experiment (baselines + adaptive variants) as a flat list of spec dicts."""
    specs: List[Dict[str, Any]] = []
    for B, (steps, eta) in itertools.product(B_list, step_eta_pairs):
        for B1, sigma_mode, bias_type, reuse in itertools.product(
            B1_list, sigma_modes, bias_types, reuse_flags
        ):
            if B1 >= B:
                log.info("Skipping config with B1=%s >= B=%s", B1, B)
                continue
            specs.append(
                {
                    "mode": "estimate_and_sample",
                    "B": B,
                    "B1": B1,
                    "sampling_steps": steps,
                    "eta": eta,
                    "reference_sampling_steps": steps,
                    "reference_eta": eta,
                    "solver": None,
                    "solver_sampling_steps": None,
                    "solver_eta": None,
                    "solver_nfe": None,
                    "sigma_estimation_mode": sigma_mode,
                    "bias_type": bias_type,
                    "reuse_phase1_samples": reuse,
                    "method_label": (
                        f"{sigma_mode}_{bias_type}_{'reuse' if reuse else 'fresh'}"
                    ),
                }
            )
        for baseline_spec in baselines:
            spec = dict(baseline_spec)
            spec["B"] = B
            spec["B1"] = None
            spec["sampling_steps"] = steps
            spec["eta"] = eta
            spec["reference_sampling_steps"] = steps
            spec["reference_eta"] = eta
            spec["sigma_estimation_mode"] = None
            spec["bias_type"] = None
            spec["reuse_phase1_samples"] = None
            if spec["mode"] == "solver_baseline":
                spec["solver_nfe"] = int(spec["solver_sampling_steps"])
            else:
                spec["solver"] = None
                spec["solver_sampling_steps"] = None
                spec["solver_eta"] = None
                spec["solver_nfe"] = None
            specs.append(spec)
    return specs


def _run_trial(
    spec: Dict[str, Any],
    *,
    model,
    target_spec,
    data_mean,
    data_std,
    reference_cdf_state,
    args,
    split_percentages,
    x_grid,
    seed: int,
    n_runs: int | None = None,
    run_start_index: int = 0,
    return_trial_results: bool = False,
) -> Dict[str, Any]:
    n_runs = args.n_runs if n_runs is None else n_runs
    common = dict(
        model=model,
        target_spec=target_spec,
        data_mean=data_mean,
        data_std=data_std,
        reference_cdf_state=reference_cdf_state,
        B=spec["B"],
        T=args.T,
        sampling_steps=spec["sampling_steps"],
        eta=spec["eta"],
        split_percentages=split_percentages,
        n_runs=n_runs,
        seed=seed,
        device=args.device,
        reference_mode=args.reference_mode,
        debug=args.debug,
        n_parallel=args.n_parallel,
        run_start_index=run_start_index,
        return_trial_results=return_trial_results,
    )
    if spec["mode"] == "solver_baseline":
        return run_solver_baseline_sampling(
            model=model,
            target_spec=target_spec,
            data_mean=data_mean,
            data_std=data_std,
            reference_cdf_state=reference_cdf_state,
            solver=spec["solver"],
            B=spec["B"],
            T=args.T,
            sampling_steps=spec["solver_sampling_steps"],
            eta=spec["solver_eta"],
            n_runs=n_runs,
            seed=seed,
            device=args.device,
            reference_mode=args.reference_mode,
            debug=args.debug,
            n_parallel=args.n_parallel,
            run_start_index=run_start_index,
            return_trial_results=return_trial_results,
        )
    if spec["mode"] == "fixed_N":
        return run_fixed_N_sampling(
            **common,
            N_i_list=[1.0] * len(split_percentages),
        )
    return run_estimate_and_sample(
        **common,
        B1=spec["B1"],
        x_grid=x_grid,
        independent_n2=args.independent_n2,
        sigma_estimation_mode=spec["sigma_estimation_mode"],
        bias_type=spec["bias_type"],
        reuse_phase1_samples=spec["reuse_phase1_samples"],
    )


def _mean_ks_value(record: Dict[str, Any]):
    """Return mean_ks as float, or None when the record is an error / NaN."""
    if record.get("error"):
        return None
    val = record.get("mean_ks")
    if val is None:
        return None
    val = float(val)
    return None if np.isnan(val) else val


def _record_sort_key(record: Dict[str, Any]):
    mean_ks = _mean_ks_value(record)
    return (
        int(record["B"]),
        int(record["sampling_steps"]),
        float(record["eta"]),
        mean_ks is None,
        mean_ks if mean_ks is not None else 0.0,
        str(record["method_label"]),
    )


def _compute_best_ids(records: List[Dict[str, Any]]) -> set:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for record in records:
        key = (int(record["B"]), int(record["sampling_steps"]), float(record["eta"]))
        grouped.setdefault(key, []).append(record)
    best_ids: set = set()
    for group in grouped.values():
        scored = [(r, _mean_ks_value(r)) for r in group]
        valid = [(r, m) for r, m in scored if m is not None]
        if not valid:
            continue
        best = min(valid, key=lambda pair: pair[1])[0]
        best_ids.add(int(best["record_id"]))
    return best_ids


def _best_records(records: List[Dict[str, Any]], best_ids: set) -> List[Dict[str, Any]]:
    selected = [
        {k: record.get(k) for k in BEST_RECORD_FIELDS}
        for record in records
        if int(record["record_id"]) in best_ids
    ]
    selected.sort(
        key=lambda r: (int(r["B"]), int(r["sampling_steps"]), float(r["eta"]))
    )
    return selected


def _build_summary_rows(
    records: List[Dict[str, Any]], best_ids: set
) -> List[Dict[str, Any]]:
    rows = []
    for record in sorted(records, key=_record_sort_key):
        row = {field: record.get(field, "") for field in CSV_FIELDS}
        row["B1"] = "" if record.get("B1") is None else record.get("B1")
        row["reference_sampling_steps"] = record.get("reference_sampling_steps", "")
        row["reference_eta"] = record.get("reference_eta", "")
        row["solver"] = record.get("solver") or ""
        row["solver_sampling_steps"] = record.get("solver_sampling_steps", "")
        row["solver_eta"] = record.get("solver_eta", "")
        row["solver_nfe"] = record.get("solver_nfe", "")
        row["N_i"] = ",".join(f"{float(x):.6g}" for x in (record.get("N_i") or []))
        row["N_i_std"] = ",".join(
            f"{float(x):.6g}" for x in (record.get("N_i_std") or [])
        )
        row["N_i_count"] = record.get("N_i_count", "")
        row["is_best_for_pair"] = record["record_id"] in best_ids
        rows.append(row)
    return rows


def _write_summary_csv(csv_path: str, rows: List[Dict[str, Any]]):
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    # Drops any "samples" key defensively; per-run sample arrays are large and
    # not needed downstream of the CSV summary.
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


RUN_CACHE_SCHEMA_VERSION = 1


def _canonical_json(value) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _checkpoint_identity(path: str) -> Dict[str, Any]:
    identity = {"path": os.path.abspath(path)}
    if os.path.exists(path):
        stat = os.stat(path)
        identity["mtime_ns"] = int(stat.st_mtime_ns)
        identity["size"] = int(stat.st_size)
    return identity


def _runs_dir_for_output_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "runs")


def _csv_path_for_output_dir(
    output_dir: str, independent_n2: int, split_percentages: str
):
    split_tag = split_percentages.replace(",", "_")
    return os.path.join(
        output_dir, f"compare_results_{independent_n2},_{split_tag}.csv"
    )


def _config_cache_key(
    args,
    spec: Dict[str, Any],
    *,
    split_percentages: List[float],
    x_grid: List[float],
) -> Dict[str, Any]:
    cache_key = {
        "schema_version": RUN_CACHE_SCHEMA_VERSION,
        "checkpoint": _checkpoint_identity(args.checkpoint),
        "reference_mode": args.reference_mode,
        "num_base_samples": int(args.num_base_samples),
        "T": int(args.T),
        "spec": _json_safe(spec),
    }
    if spec["mode"] in {"estimate_and_sample", "fixed_N"}:
        cache_key["split_percentages"] = [float(x) for x in split_percentages]
    if spec["mode"] == "estimate_and_sample":
        cache_key["x_grid"] = [float(x) for x in x_grid]
        if spec.get("sigma_estimation_mode") == "independent":
            cache_key["independent_n2"] = int(args.independent_n2)
    if args.reference_mode == "ddpm_samples":
        cache_key["reference_samples"] = _checkpoint_identity(
            reference_samples_path(
                args.checkpoint,
                int(spec["reference_sampling_steps"]),
                float(spec["reference_eta"]),
            )
        )
    elif args.reference_mode == "true_samples":
        cache_key["reference_samples"] = _checkpoint_identity(
            true_reference_samples_path(args.checkpoint)
        )
    return cache_key


def _config_id(cache_key: Dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(cache_key).encode("utf-8")).hexdigest()
    return digest[:16]


def _seed_for_cache_key(cache_key: Dict[str, Any]) -> int:
    digest = hashlib.sha256(_canonical_json(cache_key).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _compact_runs_path(config_dir: str) -> str:
    return os.path.join(config_dir, "runs.txt")


def _write_json_atomic(path: str, payload: Dict[str, Any]):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    os.replace(tmp_path, path)


def _format_cache_float(value) -> str:
    return f"{float(value):.17g}"


def _compact_trial_for_cache(
    spec: Dict[str, Any], trial: Dict[str, Any]
) -> Dict[str, Any]:
    compact = {
        "ks_distance": float(trial["ks_distance"]),
        "extinct": bool(int(trial["leaf_count"]) == 0),
    }
    if spec["mode"] == "estimate_and_sample":
        compact.update(
            {
                "N_i": [float(x) for x in trial["N_i"]],
                "n0": int(trial["n0"]),
                "used_B1": int(trial["used_B1"]),
            }
        )
    return compact


def _compact_trial_to_line(spec: Dict[str, Any], trial: Dict[str, Any]) -> str:
    fields = [
        _format_cache_float(trial["ks_distance"]),
        "1" if trial["extinct"] else "0",
    ]
    if spec["mode"] == "estimate_and_sample":
        fields.extend([str(int(trial["n0"])), str(int(trial["used_B1"]))])
        fields.extend(_format_cache_float(x) for x in trial["N_i"])
    return "\t".join(fields)


def _parse_compact_trial_line(line: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    parts = line.rstrip("\n").split("\t")
    if spec["mode"] == "estimate_and_sample":
        if len(parts) < 5:
            raise ValueError("adaptive run lines need ks, extinct, n0, used_B1, and N_i")
        return {
            "ks_distance": float(parts[0]),
            "extinct": bool(int(parts[1])),
            "n0": int(parts[2]),
            "used_B1": int(parts[3]),
            "N_i": [float(x) for x in parts[4:]],
        }
    if len(parts) != 2:
        raise ValueError("baseline run lines need ks and extinct")
    return {
        "ks_distance": float(parts[0]),
        "extinct": bool(int(parts[1])),
    }


def _load_cached_run_payloads(
    config_dir: str, spec: Dict[str, Any], n_runs: int
) -> Dict[int, Dict[str, Any]]:
    cached = {}
    path = _compact_runs_path(config_dir)
    if not os.path.exists(path):
        return cached
    try:
        with open(path) as f:
            for run_number, line in enumerate(f, start=1):
                if run_number > n_runs:
                    break
                if not line.strip():
                    continue
                try:
                    trial = _parse_compact_trial_line(line, spec)
                except (TypeError, ValueError) as exc:
                    log.warning(
                        "Ignoring invalid compact run cache %s line %s: %s",
                        path,
                        run_number,
                        exc,
                    )
                    continue
                cached[run_number] = {"run_number": run_number, "trial": trial}
    except OSError as exc:
        log.warning("Ignoring unreadable compact run cache %s: %s", path, exc)
    return cached


def _write_compact_runs_atomic(
    config_dir: str,
    spec: Dict[str, Any],
    cached: Dict[int, Dict[str, Any]],
):
    path = _compact_runs_path(config_dir)
    os.makedirs(config_dir, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    max_run = max(cached, default=0)
    with open(tmp_path, "w") as f:
        for run_number in range(1, max_run + 1):
            payload = cached.get(run_number)
            if payload is None:
                f.write("\n")
                continue
            f.write(_compact_trial_to_line(spec, payload["trial"]) + "\n")
    os.replace(tmp_path, path)


def _iter_consecutive_ranges(values: List[int]):
    if not values:
        return
    sorted_values = sorted(values)
    start = prev = sorted_values[0]
    for value in sorted_values[1:]:
        if value == prev + 1:
            prev = value
            continue
        yield start, prev
        start = prev = value
    yield start, prev


def _summarize_cached_sampling_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    ks_distances = np.array(
        [trial.get("ks_distance", np.nan) for trial in trials], dtype=float
    )
    extinct = np.array([bool(trial["extinct"]) for trial in trials], dtype=bool)

    valid_ks = ks_distances[~np.isnan(ks_distances)]
    if valid_ks.size:
        mean_ks = float(np.mean(valid_ks))
        std_ks = float(np.std(valid_ks))
        var_ks = float(np.var(valid_ks))
    else:
        mean_ks = std_ks = var_ks = float("nan")

    return {
        "mean_ks": mean_ks,
        "std_ks": std_ks,
        "var_ks": var_ks,
        "n_valid_runs": int(valid_ks.size),
        "extinction_rate": float(np.mean(extinct)) if extinct.size else 0.0,
    }


def _aggregate_cached_runs(
    spec: Dict[str, Any],
    run_payloads: List[Dict[str, Any]],
    *,
    args,
    split_percentages: List[float],
    x_grid: List[float],
) -> Dict[str, Any]:
    trials = [
        payload["trial"]
        for payload in sorted(run_payloads, key=lambda p: p["run_number"])
    ]
    stats = _summarize_cached_sampling_trials(trials)

    if not trials:
        return {**spec, "error": "No completed cached runs", "N_i": []}

    if spec["mode"] == "solver_baseline":
        solver_sampling_steps = int(spec["solver_sampling_steps"])
        n0 = int(spec["B"] // solver_sampling_steps)
        return {
            "mode": "solver_baseline",
            "solver": spec["solver"],
            "B": int(spec["B"]),
            "B1": 0,
            "used_B1": 0,
            "B2": int(spec["B"]),
            "sampling_steps": int(spec["sampling_steps"]),
            "eta": float(spec["eta"]),
            "reference_mode": args.reference_mode,
            "solver_nfe": int(solver_sampling_steps),
            "N_i": [],
            "N_i_std": [],
            "N_i_count": int(len(trials)),
            "n0": int(n0),
            "final_root_count": int(n0),
            "expected_cost_per_root": float(solver_sampling_steps),
            "expected_total_samples": float(n0),
            **stats,
        }

    if spec["mode"] == "fixed_N":
        ddim = DDIM(
            T=args.T,
            device="cpu",
            eta=spec["eta"],
            sampling_steps=spec["sampling_steps"],
        )
        resolved_step_points, split_points = _resolve_split_percentages(
            ddim, split_percentages
        )
        split_factors = [1.0] * len(split_percentages)
        expected_cost_per_root = _expected_cost_per_root(
            ddim, split_points, split_factors
        )
        n0 = int(spec["B"] // expected_cost_per_root)
        expected_total_samples = float(n0)
        for split_factor in split_factors:
            expected_total_samples *= split_factor
        return {
            "mode": "fixed_N",
            "B": int(spec["B"]),
            "B1": 0,
            "used_B1": 0,
            "B2": int(spec["B"]),
            "sampling_steps": int(spec["sampling_steps"]),
            "eta": float(spec["eta"]),
            "reference_mode": args.reference_mode,
            "split_percentages": [float(x) for x in split_percentages],
            "split_step_points": [int(x) for x in resolved_step_points],
            "split_points": [int(x) for x in split_points],
            "input_N_i": split_factors,
            "N_i": split_factors,
            "N_i_std": [0.0] * len(split_factors),
            "N_i_count": int(len(trials)),
            "n0": int(n0),
            "final_root_count": int(n0),
            "expected_cost_per_root": float(expected_cost_per_root),
            "expected_total_samples": float(expected_total_samples),
            **stats,
        }

    expected_total_samples = [
        float(trial["n0"]) * float(np.prod(np.asarray(trial["N_i"], dtype=float)))
        for trial in trials
    ]
    return {
        "mode": "estimate_and_sample",
        "B": int(spec["B"]),
        "B1": int(spec["B1"]),
        "sampling_steps": int(spec["sampling_steps"]),
        "eta": float(spec["eta"]),
        "reference_mode": args.reference_mode,
        "split_percentages": [float(x) for x in split_percentages],
        "x_grid": [float(x) for x in x_grid],
        "sigma_estimation_mode": spec["sigma_estimation_mode"],
        "bias_type": spec["bias_type"],
        "independent_n2": int(args.independent_n2),
        "reuse_phase1_samples": bool(spec["reuse_phase1_samples"]),
        "N_i": _mean_vector([trial["N_i"] for trial in trials]),
        "N_i_std": _std_vector([trial["N_i"] for trial in trials]),
        "N_i_count": int(len(trials)),
        "n0": _mean_scalar([trial["n0"] for trial in trials]),
        "expected_total_samples": _mean_scalar(expected_total_samples),
        "used_B1": _mean_scalar([trial["used_B1"] for trial in trials]),
        **stats,
    }


def _build_config_payload(
    args,
    B_list,
    B1_list,
    step_eta_pairs,
    baselines,
    sigma_modes,
    bias_types,
    reuse_flags,
    split_percentages,
    x_grid,
    target_spec,
):
    config = {
        k: v
        for k, v in vars(args).items()
        if k
        not in {
            "output_dir",
            "debug",
            "B_list",
            "B1_list",
            "step_eta_pairs",
            "baselines",
            "sigma_modes",
            "bias_types",
            "reuse_flags",
            "split_percentages",
            "x_grid",
        }
    }
    config["B_list"] = B_list
    config["B1_list"] = B1_list
    config["step_eta_pairs"] = [
        {"sampling_steps": s, "eta": e} for s, e in step_eta_pairs
    ]
    config["baselines"] = [dict(spec) for spec in baselines]
    config["sigma_modes"] = sigma_modes
    config["bias_types"] = bias_types
    config["reuse_flags"] = reuse_flags
    config["reference_mode"] = args.reference_mode
    config["split_percentages"] = split_percentages
    config["x_grid"] = x_grid
    config["target_spec"] = target_spec
    return config


def _load_reference_cdf_states(
    checkpoint_path: str,
    step_eta_pairs: List[Tuple[int, float]],
    num_base_samples: int,
    device: str,
    reference_mode: str,
):
    reference_cdf_states = {}
    if reference_mode == "true_samples":
        path = true_reference_samples_path(checkpoint_path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing true reference samples: {path}. Run ensure_samples.py first."
            )
        payload = load_reference_payload(path, map_location="cpu")
        samples = extract_reference_samples_tensor(payload)
        if int(samples.shape[0]) < num_base_samples:
            raise ValueError(
                f"True reference samples only contain {samples.shape[0]} points, "
                f"but --num_base_samples={num_base_samples}. Run ensure_samples.py first."
            )
        selected_samples = samples[:num_base_samples]
        state = prepare_reference_cdf_state(selected_samples)
        reference_cdf_states["true_samples"] = state
        log.debug("Loaded true reference samples from %s", path)
        warm_reference_ks_kernel(state)
        return reference_cdf_states

    if reference_mode != "ddpm_samples":
        return reference_cdf_states

    for sampling_steps, eta in step_eta_pairs:
        path = reference_samples_path(checkpoint_path, sampling_steps, eta)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing reference samples for steps={sampling_steps}, eta={eta}: {path}. "
                "Run ensure_samples.py first."
            )

        payload = load_reference_payload(path, map_location="cpu")
        samples = extract_reference_samples_tensor(payload)
        if int(samples.shape[0]) < num_base_samples:
            raise ValueError(
                f"Reference samples for steps={sampling_steps}, eta={eta} only contain "
                f"{samples.shape[0]} points, but --num_base_samples={num_base_samples}. "
                "Run ensure_samples.py first."
            )

        if isinstance(payload, dict):
            payload_steps = payload.get("sampling_steps")
            payload_eta = payload.get("eta")
            if payload_steps is not None and int(payload_steps) != int(sampling_steps):
                raise ValueError(
                    f"Reference file {path} has sampling_steps={payload_steps}, expected {sampling_steps}"
                )
            if payload_eta is not None and float(payload_eta) != float(eta):
                raise ValueError(
                    f"Reference file {path} has eta={payload_eta}, expected {eta}"
                )

        selected_samples = samples[:num_base_samples]
        state = prepare_reference_cdf_state(selected_samples)
        reference_cdf_states[(int(sampling_steps), float(eta))] = state
        log.debug(
            "Loaded reference samples for steps=%s eta=%s from %s",
            sampling_steps,
            eta,
            path,
        )
    if reference_cdf_states:
        warm_reference_ks_kernel(next(iter(reference_cdf_states.values())))
    return reference_cdf_states


def _reference_state_for_spec(
    reference_cdf_states: Dict[Any, Any],
    spec: Dict[str, Any],
    reference_mode: str,
):
    if reference_mode == "true_samples":
        return reference_cdf_states["true_samples"]
    if reference_mode == "ddpm_samples":
        reference_key = (
            int(spec["reference_sampling_steps"]),
            float(spec["reference_eta"]),
        )
        return reference_cdf_states[reference_key]
    return None


def main():
    args = parse_args()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    B_list = _parse_int_list(args.B_list, "B_list")
    B1_list = _parse_int_list(args.B1_list, "B1_list")
    step_eta_pairs = _parse_step_eta_pairs(args.step_eta_pairs)
    baselines = _parse_baselines(args.baselines)
    sigma_modes = _parse_sigma_modes(args.sigma_modes)
    bias_types = _parse_bias_types(args.bias_types)
    reuse_flags = _parse_bool_list(args.reuse_flags, "reuse_flags")
    split_percentages = _parse_split_percentages(args.split_percentages)
    x_grid = _parse_x_grid(args.x_grid)
    if args.independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")
    if args.n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if args.n_parallel < 1:
        raise ValueError("n_parallel must be at least 1")

    trial_specs = _build_trial_specs(
        B_list,
        B1_list,
        step_eta_pairs,
        baselines,
        sigma_modes,
        bias_types,
        reuse_flags,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    runs_dir = _runs_dir_for_output_dir(args.output_dir)

    cache_entries = []
    total_missing = 0
    for record_id, spec in enumerate(trial_specs, start=1):
        cache_key = _config_cache_key(
            args,
            spec,
            split_percentages=split_percentages,
            x_grid=x_grid,
        )
        seed = _seed_for_cache_key(cache_key)
        config_id = _config_id(cache_key)
        config_dir = os.path.join(runs_dir, config_id)
        cached = _load_cached_run_payloads(config_dir, spec, args.n_runs)
        missing = [run for run in range(1, args.n_runs + 1) if run not in cached]
        total_missing += len(missing)
        cache_entries.append(
            {
                "record_id": record_id,
                "spec": spec,
                "seed": seed,
                "cache_key": cache_key,
                "config_id": config_id,
                "config_dir": config_dir,
                "cached": cached,
                "missing": missing,
            }
        )

    if total_missing:
        if args.reference_mode in {"true_samples", "ddpm_samples"}:
            reference_cdf_states = _load_reference_cdf_states(
                args.checkpoint,
                _unique_step_eta_pairs(step_eta_pairs),
                args.num_base_samples,
                args.device,
                args.reference_mode,
            )
        else:
            reference_cdf_states = {}

        model, target_spec, data_mean, data_std = _load_model_and_stats(args)
        log.debug("Loaded checkpoint %s onto %s", args.checkpoint, args.device)

        global_config = _build_config_payload(
            args,
            B_list,
            B1_list,
            step_eta_pairs,
            baselines,
            sigma_modes,
            bias_types,
            reuse_flags,
            split_percentages,
            x_grid,
            target_spec,
        )

        for entry in tqdm(cache_entries, desc="Compare sweep"):
            missing = entry["missing"]
            if not missing:
                continue

            spec = entry["spec"]
            config_dir = entry["config_dir"]
            config_id = entry["config_id"]
            _write_json_atomic(
                os.path.join(config_dir, "config.json"),
                {
                    "schema_version": RUN_CACHE_SCHEMA_VERSION,
                    "config_id": config_id,
                    "cache_key": entry["cache_key"],
                    "global_config": global_config,
                    "spec": spec,
                },
            )

            reference_cdf_state = None
            if args.reference_mode in {"true_samples", "ddpm_samples"}:
                reference_cdf_state = _reference_state_for_spec(
                    reference_cdf_states, spec, args.reference_mode
                )

            for start_run, end_run in _iter_consecutive_ranges(missing):
                n_segment_runs = end_run - start_run + 1
                try:
                    result = _run_trial(
                        spec,
                        model=model,
                        target_spec=target_spec,
                        data_mean=data_mean,
                        data_std=data_std,
                        reference_cdf_state=reference_cdf_state,
                        args=args,
                        split_percentages=split_percentages,
                        x_grid=x_grid,
                        seed=entry["seed"],
                        n_runs=n_segment_runs,
                        run_start_index=start_run - 1,
                        return_trial_results=True,
                    )
                except Exception as exc:
                    log.warning(
                        "Failed config %s runs %s-%s: %s",
                        config_id,
                        start_run,
                        end_run,
                        exc,
                    )
                    continue

                if result.get("error"):
                    log.warning(
                        "Failed config %s runs %s-%s: %s",
                        config_id,
                        start_run,
                        end_run,
                        result["error"],
                    )
                    continue

                trial_results = result.get("trial_results") or []
                if len(trial_results) != n_segment_runs:
                    log.warning(
                        "Config %s runs %s-%s returned %d/%d trials; not caching",
                        config_id,
                        start_run,
                        end_run,
                        len(trial_results),
                        n_segment_runs,
                    )
                    continue

                for local_idx, trial in enumerate(trial_results):
                    run_number = start_run + local_idx
                    entry["cached"][run_number] = {
                        "run_number": int(run_number),
                        "trial": _compact_trial_for_cache(spec, trial),
                    }
                _write_compact_runs_atomic(config_dir, spec, entry["cached"])
    else:
        log.info("All requested runs are already cached; skipping sweep execution")

    records: List[Dict[str, Any]] = []
    completed_runs = 0
    for entry in cache_entries:
        record_id = entry["record_id"]
        spec = entry["spec"]
        cached = _load_cached_run_payloads(entry["config_dir"], spec, args.n_runs)
        completed_runs += len(cached)
        result = _aggregate_cached_runs(
            spec,
            [cached[run] for run in sorted(cached)],
            args=args,
            split_percentages=split_percentages,
            x_grid=x_grid,
        )
        if len(cached) < args.n_runs:
            result["error"] = f"Only {len(cached)}/{args.n_runs} runs completed"
        result.update(
            {
                "record_id": record_id,
                "method_label": spec["method_label"],
                "sampling_steps": spec["sampling_steps"],
                "eta": spec["eta"],
                "reference_mode": args.reference_mode,
                "reference_sampling_steps": spec.get("reference_sampling_steps"),
                "reference_eta": spec.get("reference_eta"),
                "solver": spec.get("solver"),
                "solver_sampling_steps": spec.get("solver_sampling_steps"),
                "solver_eta": spec.get("solver_eta"),
                "solver_nfe": spec.get("solver_nfe"),
                "bias_type": spec.get("bias_type"),
            }
        )
        records.append(result)

    best_ids = _compute_best_ids(records)
    best_by_pair = _best_records(records, best_ids)
    summary_rows = _build_summary_rows(records, best_ids)
    log.debug("Completed compare sweep with %d records", len(records))

    csv_output = _csv_path_for_output_dir(
        args.output_dir, args.independent_n2, args.split_percentages
    )
    _write_summary_csv(csv_output, summary_rows)

    print(f"Saved compact run cache to {runs_dir}")
    print(f"Saved CSV summary to {csv_output}")
    print(f"Completed {completed_runs}/{len(trial_specs) * args.n_runs} cached runs")
    print(f"Attempted {total_missing} missing runs")
    for best in best_by_pair:
        print(
            f"  B={best['B']}, steps={best['sampling_steps']}, eta={best['eta']}: "
            f"best={best['method_label']} (B1={best['B1']}) mean KS={best['mean_ks']:.6f}"
        )


if __name__ == "__main__":
    main()
