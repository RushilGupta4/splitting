import argparse
import csv
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
)
from infer import (
    _load_model_and_stats,
    _parse_split_percentages,
    _parse_x_grid,
    prepare_reference_cdf_state,
    run_estimate_and_sample,
    run_fixed_N_sampling,
    run_solver_baseline_sampling,
    warm_reference_ks_kernel,
)

log = logging.getLogger("compare")

# SIGMA_MODES = ("pilot_tree", "independent")
SIGMA_MODES = ("pilot_tree",)
# SIGMA_MODES = ("independent",)
REUSE_FLAGS = (False, True)
SUPPORTED_SOLVERS = ("ddim", "dpmpp_2m")

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
        "--solver_baselines",
        type=str,
        # default="ddim_40_eta0,dpmpp_2m_40,dpmpp_2m_100,dpmpp_2m_200",
        default="dpmpp_2m_40",
        help=(
            "Comma-separated solver baselines in '<solver>_<steps>' or "
            "'<solver>_<steps>_eta<eta>' format. Supported solvers: "
            f"{', '.join(SUPPORTED_SOLVERS)}. Use '' to disable."
        ),
    )
    parser.add_argument(
        "--reference_mode",
        choices=("samples", "true_dist"),
        default="samples",
        help="samples uses cached DDPM/DDIM reference samples; true_dist uses the exact target CDF.",
    )
    parser.add_argument("--n_runs", type=int, default=100)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_base_samples", type=int, default=1000000)
    parser.add_argument("--output", type=str, default="outputs/compare_results.json")
    parser.add_argument(
        "--csv_output",
        type=str,
        default=None,
        help="Path to summary CSV (default: --output with .csv)",
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


def _parse_solver_baseline_name(name: str) -> Dict[str, Any]:
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
        f"Unknown solver baseline '{name}'. Expected '<solver>_<steps>' or "
        f"'<solver>_<steps>_eta<eta>' with supported solvers: {supported}"
    )


def _parse_solver_baselines(raw: str) -> List[Dict[str, Any]]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    return [_parse_solver_baseline_name(name) for name in names]


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
    solver_baselines: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Every experiment (baseline + adaptive variants) as a flat list of spec dicts."""
    specs: List[Dict[str, Any]] = []
    for B, (steps, eta) in itertools.product(B_list, step_eta_pairs):
        specs.append(
            {
                "mode": "fixed_N",
                "B": B,
                "B1": None,
                "sampling_steps": steps,
                "eta": eta,
                "reference_sampling_steps": steps,
                "reference_eta": eta,
                "solver": None,
                "solver_sampling_steps": None,
                "solver_eta": None,
                "solver_nfe": None,
                "sigma_estimation_mode": None,
                "reuse_phase1_samples": None,
                "method_label": "all_ones_baseline",
            }
        )
        for B1, sigma_mode, reuse in itertools.product(
            B1_list, SIGMA_MODES, REUSE_FLAGS
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
                    "reuse_phase1_samples": reuse,
                    "method_label": f"{sigma_mode}_{'reuse' if reuse else 'fresh'}",
                }
            )
        for solver_spec in solver_baselines:
            spec = dict(solver_spec)
            spec["B"] = B
            spec["B1"] = None
            spec["sampling_steps"] = steps
            spec["eta"] = eta
            spec["reference_sampling_steps"] = steps
            spec["reference_eta"] = eta
            spec["sigma_estimation_mode"] = None
            spec["reuse_phase1_samples"] = None
            spec["solver_nfe"] = int(spec["solver_sampling_steps"])
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
) -> Dict[str, Any]:
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
        n_runs=args.n_runs,
        seed=seed,
        device=args.device,
        reference_mode=args.reference_mode,
        debug=args.debug,
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
            n_runs=args.n_runs,
            seed=seed,
            device=args.device,
            reference_mode=args.reference_mode,
            debug=args.debug,
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
    # Drops any "samples" key to keep the payload compact — per-run sample arrays
    # are large and not needed downstream of the CSV summary.
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


def _build_config_payload(
    args,
    B_list,
    B1_list,
    step_eta_pairs,
    solver_baselines,
    split_percentages,
    x_grid,
    target_spec,
):
    config = {
        k: v
        for k, v in vars(args).items()
        if k
        not in {
            "output",
            "csv_output",
            "debug",
            "B_list",
            "B1_list",
            "step_eta_pairs",
            "solver_baselines",
            "split_percentages",
            "x_grid",
        }
    }
    config["B_list"] = B_list
    config["B1_list"] = B1_list
    config["step_eta_pairs"] = [
        {"sampling_steps": s, "eta": e} for s, e in step_eta_pairs
    ]
    config["solver_baselines"] = [dict(spec) for spec in solver_baselines]
    config["reference_mode"] = args.reference_mode
    config["split_percentages"] = split_percentages
    config["x_grid"] = x_grid
    config["target_spec"] = target_spec
    return config


def _load_reference_cdf_states(
    checkpoint_path: str,
    step_eta_pairs: List[Tuple[int, float]],
    num_base_samples: int,
):
    reference_cdf_states = {}
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
        reference_cdf_states[(int(sampling_steps), float(eta))] = (
            prepare_reference_cdf_state(selected_samples)
        )
        log.debug(
            "Loaded reference samples for steps=%s eta=%s from %s",
            sampling_steps,
            eta,
            path,
        )
    if reference_cdf_states:
        warm_reference_ks_kernel(next(iter(reference_cdf_states.values())))
    return reference_cdf_states


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
    solver_baselines = _parse_solver_baselines(args.solver_baselines)
    if args.reference_mode == "samples":
        reference_cdf_states = _load_reference_cdf_states(
            args.checkpoint,
            _unique_step_eta_pairs(step_eta_pairs),
            args.num_base_samples,
        )
    else:
        reference_cdf_states = {}
    split_percentages = _parse_split_percentages(args.split_percentages)
    x_grid = _parse_x_grid(args.x_grid)
    if args.independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")

    model, target_spec, data_mean, data_std = _load_model_and_stats(args)
    log.debug("Loaded checkpoint %s onto %s", args.checkpoint, args.device)

    trial_specs = _build_trial_specs(B_list, B1_list, step_eta_pairs, solver_baselines)

    records: List[Dict[str, Any]] = []
    for record_id, spec in enumerate(tqdm(trial_specs, desc="Compare sweep"), start=1):
        # Offset the seed by record_id so different methods on the same (B, steps, eta)
        # do not share initial noise — avoids method-correlated bias.
        seed = args.seed + record_id
        log.debug("Trial %d: %s", record_id, spec)
        try:
            reference_cdf_state = None
            if args.reference_mode == "samples":
                reference_key = (
                    int(spec["reference_sampling_steps"]),
                    float(spec["reference_eta"]),
                )
                reference_cdf_state = reference_cdf_states[reference_key]
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
                seed=seed,
            )
        except Exception as exc:
            result = {**spec, "error": str(exc), "N_i": []}
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
            }
        )
        records.append(result)

    best_ids = _compute_best_ids(records)
    best_by_pair = _best_records(records, best_ids)
    summary_rows = _build_summary_rows(records, best_ids)
    log.debug("Completed compare sweep with %d records", len(records))

    payload = {
        "config": _build_config_payload(
            args,
            B_list,
            B1_list,
            step_eta_pairs,
            solver_baselines,
            split_percentages,
            x_grid,
            target_spec,
        ),
        "records": _json_safe(records),
        "best_by_pair": _json_safe(best_by_pair),
    }

    json_parent = os.path.dirname(args.output)
    if json_parent:
        os.makedirs(json_parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    csv_output = args.csv_output or os.path.splitext(args.output)[0] + ".csv"
    _write_summary_csv(csv_output, summary_rows)

    print(f"Saved comparison results to {args.output}")
    print(f"Saved CSV summary to {csv_output}")
    print(f"Ran {len(trial_specs)} experiments")
    for best in best_by_pair:
        print(
            f"  B={best['B']}, steps={best['sampling_steps']}, eta={best['eta']}: "
            f"best={best['method_label']} (B1={best['B1']}) mean KS={best['mean_ks']:.6f}"
        )


if __name__ == "__main__":
    main()
