import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from adaptive import run_estimate_and_sample
from baselines import run_fixed_N_sampling, run_solver_baseline_sampling
from ks import prepare_reference_cdf_state, warm_reference_ks_kernel
from model_io import load_model_and_stats
from reference_cache import (
    load_reference_samples_checked,
    reference_samples_path,
    true_reference_samples_path,
)
from trials import mean_vector, std_vector
from utils import parse_split_percentages, parse_step_eta_pairs, parse_x_grid

log = logging.getLogger("compare")

SUPPORTED_SOLVERS = ("ddim", "dpmpp_2m")
SUPPORTED_SIGMA_ESTIMATION_MODES = ("pilot_tree", "independent")

CSV_FIELDS = [
    "B", "B1", "sampling_steps", "eta",
    "solver", "solver_sampling_steps", "solver_eta", "solver_nfe",
    "method_label", "mode",
    "sigma_estimation_mode", "reuse_phase1_samples",
    "N_i", "N_i_std", "N_i_count",
    "mean_ks", "std_ks", "n_valid_runs",
    "error",
]

_BOOL_MAP = {
    **{k: True for k in ("1", "true", "t", "yes", "y")},
    **{k: False for k in ("0", "false", "f", "no", "n")},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep step/eta pairs and compare adaptive splitting against baselines"
    )
    p.add_argument("--checkpoint", type=str, default="checkpoints/model_final.pt")
    p.add_argument("--B_list", type=str, required=True)
    p.add_argument("--B1_list", type=str, required=True)
    p.add_argument("--step_eta_pairs", type=str, required=True,
                   help="Comma-separated 'steps:eta' pairs, e.g. '1000:1.0,500:1.0'")
    p.add_argument("--baselines", type=str, default="fixed_N,dpmpp_2m_100",
                   help="Comma-separated. 'fixed_N' or '<solver>_<steps>' or '<solver>_<steps>_eta<eta>'")
    p.add_argument("--sigma_modes", type=str, default="pilot_tree,independent")
    p.add_argument("--reuse_flags", type=str, default="false,true")
    p.add_argument("--reference_mode",
                   choices=("true_dist", "true_samples", "ddpm_samples"),
                   default="ddpm_samples")
    p.add_argument("--n_runs", type=int, default=100)
    p.add_argument("--n_parallel", type=int, default=1)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--split_percentages", type=str, default="0.5")
    p.add_argument("--independent_n2", type=int, default=10)
    p.add_argument("--x_grid", type=str, default="-2.0,-1.0,0.0,1.0,2.0")
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_base_samples", type=int, default=1000000)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# --- argument parsers -------------------------------------------------------


def _parse_csv_list(raw: str, name: str, item_fn=str, *, allowed=None):
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = item_fn(item)
        except ValueError as exc:
            raise ValueError(f"Invalid {name} value '{item}': {exc}") from exc
        if allowed is not None and value not in allowed:
            raise ValueError(f"{name} value '{value}' not in {sorted(allowed)}")
        out.append(value)
    if not out:
        raise ValueError(f"{name} must have at least one value")
    return out


def _parse_bool(s: str) -> bool:
    s = s.strip().lower()
    if s not in _BOOL_MAP:
        raise ValueError(f"expected boolean, got '{s}'")
    return _BOOL_MAP[s]


def _parse_baseline_name(name: str) -> Dict[str, Any]:
    if name == "fixed_N":
        return {"mode": "fixed_N", "method_label": "all_ones_baseline"}
    for solver in sorted(SUPPORTED_SOLVERS, key=len, reverse=True):
        prefix = f"{solver}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):].split("_")
        if len(rest) not in (1, 2):
            break
        try:
            steps = int(rest[0])
        except ValueError as exc:
            raise ValueError(f"baseline '{name}' must use integer steps") from exc
        if steps < 1:
            raise ValueError(f"baseline '{name}' must use steps >= 1")
        eta = 0.0
        if len(rest) == 2:
            tag = rest[1]
            if not tag.startswith("eta") or tag == "eta":
                break
            try:
                eta = float(tag[3:])
            except ValueError as exc:
                raise ValueError(f"baseline '{name}' has bad eta") from exc
        return {
            "mode": "solver_baseline", "method_label": name,
            "solver": solver, "solver_sampling_steps": steps, "solver_eta": eta,
        }
    raise ValueError(
        f"Unknown baseline '{name}'. Expected fixed_N, <solver>_<steps>, "
        f"or <solver>_<steps>_eta<eta>"
    )


def _unique_step_eta_pairs(pairs):
    seen = set()
    out = []
    for steps, eta in pairs:
        key = (int(steps), float(eta))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# --- trial spec builder -----------------------------------------------------


def _build_trial_specs(B_list, B1_list, step_eta_pairs, baselines, sigma_modes, reuse_flags):
    specs: List[Dict[str, Any]] = []
    schedule_baselines = [b for b in baselines if b["mode"] != "solver_baseline"]
    solver_baselines = [b for b in baselines if b["mode"] == "solver_baseline"]
    first_steps, first_eta = step_eta_pairs[0]

    for B in B_list:
        for steps, eta in step_eta_pairs:
            for B1, sigma_mode, reuse in itertools.product(
                B1_list, sigma_modes, reuse_flags
            ):
                if B1 >= B:
                    log.info("Skipping config with B1=%s >= B=%s", B1, B)
                    continue
                specs.append({
                    "mode": "estimate_and_sample",
                    "B": B, "B1": B1,
                    "sampling_steps": steps, "eta": eta,
                    "sigma_estimation_mode": sigma_mode,
                    "reuse_phase1_samples": reuse,
                    "method_label": f"{sigma_mode}_{'reuse' if reuse else 'fresh'}",
                })
            for baseline_spec in schedule_baselines:
                spec = dict(baseline_spec)
                spec["B"] = B
                spec["sampling_steps"] = steps
                spec["eta"] = eta
                specs.append(spec)
        for baseline_spec in solver_baselines:
            spec = dict(baseline_spec)
            spec["B"] = B
            spec["sampling_steps"] = first_steps
            spec["eta"] = first_eta
            spec["solver_nfe"] = int(spec["solver_sampling_steps"])
            specs.append(spec)
    return specs


# --- trial executor ---------------------------------------------------------


def _run_trial(
    spec, *, model, target_spec, data_mean, data_std, reference_cdf_state, args,
    split_percentages, x_grid, seed, n_runs=None, run_start_index=0,
    return_trial_results=False,
):
    n_runs = args.n_runs if n_runs is None else n_runs
    common = dict(
        model=model, target_spec=target_spec, data_mean=data_mean, data_std=data_std,
        reference_cdf_state=reference_cdf_state, B=spec["B"], T=args.T,
        n_runs=n_runs, seed=seed, device=args.device,
        reference_mode=args.reference_mode, debug=args.debug,
        n_parallel=args.n_parallel, run_start_index=run_start_index,
        return_trial_results=return_trial_results,
    )
    if spec["mode"] == "solver_baseline":
        return run_solver_baseline_sampling(
            **common, solver=spec["solver"],
            sampling_steps=spec["solver_sampling_steps"], eta=spec["solver_eta"],
        )
    common["sampling_steps"] = spec["sampling_steps"]
    common["eta"] = spec["eta"]
    if spec["mode"] == "fixed_N":
        return run_fixed_N_sampling(
            **common, split_percentages=split_percentages,
            N_i_list=[1.0] * len(split_percentages),
        )
    return run_estimate_and_sample(
        **common, B1=spec["B1"], split_percentages=split_percentages, x_grid=x_grid,
        independent_n2=args.independent_n2,
        sigma_estimation_mode=spec["sigma_estimation_mode"],
        reuse_phase1_samples=spec["reuse_phase1_samples"],
    )


# --- cache layout -----------------------------------------------------------


def _runs_dir(output_dir):
    return os.path.join(output_dir, "runs")


def _csv_path(output_dir, split_percentages):
    return os.path.join(
        output_dir,
        f"compare_results_{split_percentages.replace(',', '_')}.csv",
    )


def _checkpoint_identity(path):
    identity: Dict[str, Any] = {"path": os.path.abspath(path)}
    if os.path.exists(path):
        s = os.stat(path)
        identity["mtime_ns"] = int(s.st_mtime_ns)
        identity["size"] = int(s.st_size)
    return identity


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


def _config_cache_key(args, spec, *, split_percentages, x_grid):
    key = {
        "checkpoint": _checkpoint_identity(args.checkpoint),
        "reference_mode": args.reference_mode,
        "num_base_samples": int(args.num_base_samples),
        "T": int(args.T),
        "spec": _json_safe(spec),
    }
    if spec["mode"] in {"estimate_and_sample", "fixed_N"}:
        key["split_percentages"] = [float(x) for x in split_percentages]
    if spec["mode"] == "estimate_and_sample":
        key["x_grid"] = [float(x) for x in x_grid]
        if spec.get("sigma_estimation_mode") == "independent":
            key["independent_n2"] = int(args.independent_n2)
    if args.reference_mode == "ddpm_samples":
        key["reference_samples"] = _checkpoint_identity(
            reference_samples_path(
                args.checkpoint, int(spec["sampling_steps"]), float(spec["eta"])
            )
        )
    elif args.reference_mode == "true_samples":
        key["reference_samples"] = _checkpoint_identity(
            true_reference_samples_path(args.checkpoint)
        )
    return key


def _config_hash(cache_key):
    return hashlib.sha256(
        json.dumps(_json_safe(cache_key), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json_atomic(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    os.replace(tmp, path)


# --- JSON-Lines per-run cache -----------------------------------------------


def _runs_jsonl_path(config_dir):
    return os.path.join(config_dir, "runs.jsonl")


def _trial_to_cache_record(spec, trial):
    record = {
        "ks_distance": float(trial["ks_distance"]),
        "extinct": bool(int(trial["leaf_count"]) == 0),
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
            if run_number > n_runs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                cached[run_number] = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Ignoring invalid line %s in %s: %s", run_number, path, exc)
    return cached


def _write_cached_runs(config_dir, cached):
    path = _runs_jsonl_path(config_dir)
    os.makedirs(config_dir, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    max_run = max(cached, default=0)
    with open(tmp, "w") as f:
        for run_number in range(1, max_run + 1):
            record = cached.get(run_number)
            f.write("\n" if record is None else json.dumps(record, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def _iter_consecutive_ranges(values):
    if not values:
        return
    s = sorted(values)
    start = prev = s[0]
    for v in s[1:]:
        if v == prev + 1:
            prev = v
            continue
        yield start, prev
        start = prev = v
    yield start, prev


# --- reference state --------------------------------------------------------


def _load_reference_cdf_states(checkpoint_path, step_eta_pairs, num_base_samples, reference_mode):
    states: Dict[Any, Any] = {}
    if reference_mode == "true_samples":
        states["true_samples"] = prepare_reference_cdf_state(
            load_reference_samples_checked(
                true_reference_samples_path(checkpoint_path), num_base_samples
            )
        )
    elif reference_mode == "ddpm_samples":
        for steps, eta in step_eta_pairs:
            path = reference_samples_path(checkpoint_path, steps, eta)
            states[(int(steps), float(eta))] = prepare_reference_cdf_state(
                load_reference_samples_checked(
                    path, num_base_samples,
                    expected_sampling_steps=steps, expected_eta=eta,
                )
            )
    if states:
        warm_reference_ks_kernel(next(iter(states.values())))
    return states


def _reference_state_for_spec(states, spec, reference_mode):
    if reference_mode == "true_samples":
        return states["true_samples"]
    if reference_mode == "ddpm_samples":
        return states[(int(spec["sampling_steps"]), float(spec["eta"]))]
    return None


# --- aggregation + CSV ------------------------------------------------------


def _aggregate_cached_runs(spec, trials, *, split_percentages):
    if not trials:
        return {**spec, "error": "No completed cached runs"}
    ks = np.array([t.get("ks_distance", np.nan) for t in trials], dtype=float)
    extinct = np.array([bool(t.get("extinct", False)) for t in trials], dtype=bool)
    valid = ks[~np.isnan(ks)]
    base = {
        "mode": spec["mode"], "B": int(spec["B"]),
        "sampling_steps": int(spec["sampling_steps"]),
        "eta": float(spec["eta"]),
        "method_label": spec.get("method_label", ""),
        "N_i_count": int(len(trials)),
        "mean_ks": float(valid.mean()) if valid.size else float("nan"),
        "std_ks": float(valid.std()) if valid.size else float("nan"),
        "n_valid_runs": int(valid.size),
        "extinction_rate": float(extinct.mean()) if extinct.size else 0.0,
    }
    if spec["mode"] == "solver_baseline":
        return {
            **base, "solver": spec["solver"],
            "solver_sampling_steps": int(spec["solver_sampling_steps"]),
            "solver_eta": float(spec["solver_eta"]),
            "solver_nfe": int(spec["solver_sampling_steps"]),
            "N_i": [], "N_i_std": [],
        }
    if spec["mode"] == "fixed_N":
        factors = [1.0] * len(split_percentages)
        return {**base, "N_i": factors, "N_i_std": [0.0] * len(factors)}
    return {
        **base, "B1": int(spec["B1"]),
        "sigma_estimation_mode": spec["sigma_estimation_mode"],
        "reuse_phase1_samples": bool(spec["reuse_phase1_samples"]),
        "N_i": mean_vector([t["N_i"] for t in trials]),
        "N_i_std": std_vector([t["N_i"] for t in trials]),
    }


def _sort_key(r):
    mean_ks = r.get("mean_ks")
    valid = mean_ks is not None and not (
        isinstance(mean_ks, float) and np.isnan(mean_ks)
    )
    return (
        int(r["B"]),
        int(r["sampling_steps"]),
        float(r["eta"]),
        not valid,
        mean_ks if valid else 0.0,
        str(r.get("method_label", "")),
    )


def _build_summary_rows(records):
    rows = []
    for r in sorted(records, key=_sort_key):
        row = {field: r.get(field, "") for field in CSV_FIELDS}
        row["B1"] = "" if r.get("B1") is None else r.get("B1")
        row["solver"] = r.get("solver") or ""
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


# --- main -------------------------------------------------------------------


def main():
    args = parse_args()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    B_list = _parse_csv_list(args.B_list, "B_list", int)
    B1_list = _parse_csv_list(args.B1_list, "B1_list", int)
    step_eta_pairs = parse_step_eta_pairs(args.step_eta_pairs)
    baselines = (
        _parse_csv_list(args.baselines, "baselines", _parse_baseline_name)
        if args.baselines.strip() else []
    )
    sigma_modes = _parse_csv_list(args.sigma_modes, "sigma_modes", str,
                                  allowed=set(SUPPORTED_SIGMA_ESTIMATION_MODES))
    reuse_flags = _parse_csv_list(args.reuse_flags, "reuse_flags", _parse_bool)
    split_percentages = parse_split_percentages(args.split_percentages)
    x_grid = parse_x_grid(args.x_grid)
    if args.independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")
    if args.n_runs < 1 or args.n_parallel < 1:
        raise ValueError("n_runs and n_parallel must be at least 1")

    specs = _build_trial_specs(
        B_list, B1_list, step_eta_pairs, baselines, sigma_modes, reuse_flags
    )
    os.makedirs(args.output_dir, exist_ok=True)
    runs_dir = _runs_dir(args.output_dir)

    entries: List[Dict[str, Any]] = []
    total_missing = 0
    for spec in specs:
        cache_key = _config_cache_key(args, spec,
                                      split_percentages=split_percentages, x_grid=x_grid)
        digest = _config_hash(cache_key)
        config_id = digest[:16]
        seed = int(config_id, 16) % (2 ** 63 - 1)
        config_dir = os.path.join(runs_dir, config_id)
        cached = _load_cached_runs(config_dir, args.n_runs)
        missing = [r for r in range(1, args.n_runs + 1) if r not in cached]
        total_missing += len(missing)
        entries.append({
            "spec": spec, "seed": seed, "cache_key": cache_key,
            "config_id": config_id, "config_dir": config_dir,
            "cached": cached, "missing": missing,
        })

    if total_missing:
        reference_cdf_states = (
            _load_reference_cdf_states(
                args.checkpoint, _unique_step_eta_pairs(step_eta_pairs),
                args.num_base_samples, args.reference_mode,
            )
            if args.reference_mode in {"true_samples", "ddpm_samples"} else {}
        )
        model, target_spec, data_mean, data_std = load_model_and_stats(
            args.checkpoint, args.device, no_compile=args.no_compile,
        )
        log.debug("Loaded checkpoint %s onto %s", args.checkpoint, args.device)

        global_config = {
            **{k: v for k, v in vars(args).items() if k != "output_dir"},
            "step_eta_pairs": [{"sampling_steps": s, "eta": e} for s, e in step_eta_pairs],
            "baselines": [dict(b) for b in baselines],
            "split_percentages": split_percentages,
            "x_grid": x_grid,
            "target_spec": target_spec,
        }

        for entry in tqdm(entries, desc="Compare sweep"):
            missing = entry["missing"]
            if not missing:
                continue
            spec = entry["spec"]
            config_dir = entry["config_dir"]
            _write_json_atomic(
                os.path.join(config_dir, "config.json"),
                {
                    "config_id": entry["config_id"],
                    "cache_key": entry["cache_key"],
                    "global_config": global_config,
                    "spec": spec,
                },
            )
            reference_cdf_state = (
                _reference_state_for_spec(reference_cdf_states, spec, args.reference_mode)
                if args.reference_mode in {"true_samples", "ddpm_samples"} else None
            )

            for start_run, end_run in _iter_consecutive_ranges(missing):
                n = end_run - start_run + 1
                try:
                    result = _run_trial(
                        spec, model=model, target_spec=target_spec,
                        data_mean=data_mean, data_std=data_std,
                        reference_cdf_state=reference_cdf_state, args=args,
                        split_percentages=split_percentages, x_grid=x_grid,
                        seed=entry["seed"], n_runs=n, run_start_index=start_run - 1,
                        return_trial_results=True,
                    )
                except Exception as exc:
                    log.warning("Failed config %s runs %s-%s: %s",
                                entry["config_id"], start_run, end_run, exc)
                    continue
                trial_results = result.get("trial_results") or []
                if len(trial_results) != n:
                    log.warning("Config %s runs %s-%s returned %d/%d; not caching",
                                entry["config_id"], start_run, end_run,
                                len(trial_results), n)
                    continue
                for local_idx, trial in enumerate(trial_results):
                    entry["cached"][start_run + local_idx] = _trial_to_cache_record(spec, trial)
                _write_cached_runs(config_dir, entry["cached"])
    else:
        log.info("All requested runs already cached; skipping execution")

    records: List[Dict[str, Any]] = []
    completed = 0
    for entry in entries:
        cached = _load_cached_runs(entry["config_dir"], args.n_runs)
        completed += len(cached)
        trials = [cached[r] for r in sorted(cached)]
        record = _aggregate_cached_runs(
            entry["spec"], trials, split_percentages=split_percentages
        )
        if len(cached) < args.n_runs:
            record["error"] = f"Only {len(cached)}/{args.n_runs} runs completed"
        records.append(record)

    csv_output = _csv_path(args.output_dir, args.split_percentages)
    _write_summary_csv(csv_output, _build_summary_rows(records))

    print(f"Saved compact run cache to {runs_dir}")
    print(f"Saved CSV summary to {csv_output}")
    print(f"Completed {completed}/{len(specs) * args.n_runs} cached runs")
    print(f"Attempted {total_missing} missing runs")

    # Print best per (B, steps, eta).
    grouped: Dict[Any, Dict[str, Any]] = {}
    for r in records:
        mean_ks = r.get("mean_ks")
        if mean_ks is None or (isinstance(mean_ks, float) and np.isnan(mean_ks)):
            continue
        key = (int(r["B"]), int(r["sampling_steps"]), float(r["eta"]))
        if key not in grouped or r["mean_ks"] < grouped[key]["mean_ks"]:
            grouped[key] = r
    for (B, steps, eta), best in sorted(grouped.items()):
        b1 = best.get("B1", "")
        print(
            f"  B={B}, steps={steps}, eta={eta}: best={best.get('method_label')} "
            f"(B1={b1}) mean KS={best['mean_ks']:.6f}"
        )


if __name__ == "__main__":
    main()
