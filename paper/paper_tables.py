#!/usr/bin/env python3
"""Validate completed experiment records and export the paper tables as CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from paper_plots import (
    LEARNED_C_OPTIMIZER,
    MODELS,
    PILOT_STAGES,
    SCHEDULES,
    TIMING_BUDGET,
    TIMING_METHODS,
    TIMING_MODELS,
    TIMING_STAGES,
    ResultRow,
    _expected_uniform_c,
    _load_and_verify_rows,
    _normal_reduction,
    _row_at_budget,
    _timing_mean_ci,
    _timing_overhead,
    _timing_rows,
)

LEARNED_C_ALLOCATION = "Learned c"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "tables"
RESULT_TABLE_STEMS = {
    "simple_ou": "complete_ou",
    "coupled_double_well_langevin": "complete_langevin",
    "edm_default": "complete_edm",
    "ddpm_cifar10_hf_mmd": "complete_ddpm",
    "ldm_ffhq_mmd": "complete_ffhq",
}
RESULT_FIELDS = [
    "split_points",
    "allocation",
    "B",
    "metric",
    "mean_method",
    "mean_independent",
    "reduction_percent",
    "reduction_ci_lower_percent",
    "reduction_ci_upper_percent",
    "n_method",
    "n_independent",
]
ORACLE_FIELDS = [
    "split_points",
    "B",
    "oracle_ks_reduction_percent",
    "oracle_ks_reduction_ci_lower_percent",
    "oracle_ks_reduction_ci_upper_percent",
    "learned_ks_reduction_percent",
    "learned_ks_reduction_ci_lower_percent",
    "learned_ks_reduction_ci_upper_percent",
    "learned_to_oracle_reduction_ratio_percent",
]
TIMING_TABLE_FIELDS = [
    "model",
    "split_points",
    "allocation",
    "B",
    *TIMING_STAGES,
    "total_seconds",
    "total_ci_lower_seconds",
    "total_ci_upper_seconds",
    "scoring_seconds",
    "overhead_percent",
    "overhead_ci_lower_percent",
    "overhead_ci_upper_percent",
    "pilot_share_percent",
    "n",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=ROOT / "outputs_paper_final",
        help="Root containing the completed experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Destination for generated CSV files " f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Export available partial results with missing entries left blank.",
    )
    args = parser.parse_args()
    if args.debug and args.output_dir is None:
        parser.error("--debug requires an explicit --output-dir")
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR
    return args


def _write_csv(
    output_dir: Path,
    stem: str,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    path = output_dir / f"{stem}.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _numerical_schedule_rows() -> tuple[list[str], list[dict[str, Any]]]:
    by_directory = {model["directory"]: model for model in MODELS}
    ou = by_directory["simple_ou"]
    langevin = by_directory["coupled_double_well_langevin"]
    edm = by_directory["edm_default"]
    ddpm = by_directory["ddpm_cifar10_hf_mmd"]
    ffhq = by_directory["ldm_ffhq_mmd"]

    if ou["budgets"] != langevin["budgets"] or ou["steps"] != langevin["steps"]:
        raise RuntimeError("OU and Langevin must use the same Euler schedule")

    def step_map(model: dict[str, Any]) -> dict[int, int]:
        return dict(zip(model["budgets"], model["steps"]))

    euler_steps = step_map(ou)
    edm_steps = step_map(edm)
    ddpm_steps = step_map(ddpm)
    ffhq_steps = step_map(ffhq)
    if any(
        ddpm_steps.get(budget, steps) != steps for budget, steps in ffhq_steps.items()
    ):
        raise RuntimeError("CIFAR-10 and FFHQ must share the DDPM step schedule")
    budgets = sorted(set(euler_steps) | set(edm_steps) | set(ddpm_steps))
    rows = []
    for budget in budgets:
        edm_step = edm_steps.get(budget)
        rows.append(
            {
                "B": budget,
                "euler_steps": euler_steps.get(budget, ""),
                "edm_steps": edm_step if edm_step is not None else "",
                "edm_nfe": 2 * edm_step - 1 if edm_step is not None else "",
                "ddpm_steps": ddpm_steps.get(budget, ""),
            }
        )
    return ["B", "euler_steps", "edm_steps", "edm_nfe", "ddpm_steps"], rows


def _reference_source(model: dict[str, Any]) -> str:
    config = model["reference_generation"]
    method = config["method"]
    if method == "sde_terminal_samples":
        return f"Euler-Maruyama with {config['sampling_steps']} steps"
    if method == "target_samples":
        return "Direct draws from the Gaussian mixture"
    if method == "hf_ddpm_scheduler":
        return f"Full {config['sampling_steps']}-step DDPM sampler"
    if method == "ldm_ddpm_samples":
        return (
            f"Full {config['sampling_steps']}-step latent DDPM sampler, "
            "decoded to 256x256 images"
        )
    raise RuntimeError(f"Unsupported reference method {method!r}")


def _reference_sample_rows() -> tuple[list[str], list[dict[str, Any]]]:
    display_names = {
        "simple_ou": "OU process",
        "coupled_double_well_langevin": "Overdamped Langevin",
        "edm_default": "EDM",
        "ddpm_cifar10_hf_mmd": "CIFAR-10 DDPM",
        "ldm_ffhq_mmd": "FFHQ LDM",
    }
    rows = [
        {
            "model": display_names[model["directory"]],
            "reference_source": _reference_source(model),
            "samples": model["reference_size"],
        }
        for model in MODELS
    ]
    return ["model", "reference_source", "samples"], rows


def _complete_result_rows(
    model: dict[str, Any],
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    *,
    debug: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    rows_by_schedule = all_rows.get(model["directory"], {})

    def result_row(
        schedule: tuple[float, ...],
        budget: int,
        *,
        allocation: str,
        mode: str,
        uniform_c: float | None = None,
        optimizer: str | None = None,
    ) -> dict[str, Any]:
        schedule_rows = rows_by_schedule.get(schedule, [])
        independent = _row_at_budget(schedule_rows, budget, "fixed_N")
        method = _row_at_budget(
            schedule_rows, budget, mode, optimizer=optimizer, uniform_c=uniform_c
        )
        if independent is None or method is None:
            if not debug:
                raise RuntimeError(
                    f"Missing {allocation} result or fixed_N baseline for "
                    f"{model['directory']}, {len(schedule)} splits, B={budget}"
                )
            values = [""] * 7
        else:
            values = [
                method.mean,
                independent.mean,
                *_normal_reduction(independent, method),
                method.n,
                independent.n,
            ]
        return dict(
            zip(RESULT_FIELDS, [len(schedule), allocation, budget, model["metric"], *values])
        )

    rows: list[dict[str, Any]] = []
    expected_keys: list[tuple[int, str, int]] = []
    for schedule, _ in SCHEDULES:
        for budget in model["budgets"]:
            expected_keys.append((len(schedule), "Learned", budget))
            rows.append(
                result_row(
                    schedule,
                    budget,
                    allocation="Learned",
                    mode="adaptive",
                    optimizer="monotone",
                )
            )
            if any(
                row.budget == budget and row.optimizer == LEARNED_C_OPTIMIZER
                for row in rows_by_schedule.get(schedule, [])
            ):
                expected_keys.append((len(schedule), LEARNED_C_ALLOCATION, budget))
                rows.append(
                    result_row(
                        schedule,
                        budget,
                        allocation=LEARNED_C_ALLOCATION,
                        mode="adaptive",
                        optimizer=LEARNED_C_OPTIMIZER,
                    )
                )
            for c in _expected_uniform_c(model, schedule):
                allocation = f"Uniform (c={c:g})"
                expected_keys.append((len(schedule), allocation, budget))
                rows.append(
                    result_row(
                        schedule,
                        budget,
                        allocation=allocation,
                        mode="uniform_c",
                        uniform_c=c,
                    )
                )

    actual_keys = [
        (int(row["split_points"]), str(row["allocation"]), int(row["B"]))
        for row in rows
    ]
    if actual_keys != expected_keys or len(set(actual_keys)) != len(actual_keys):
        raise RuntimeError(
            f"Unexpected complete-result row order or duplicate key for "
            f"{model['directory']}"
        )

    independent_means: dict[tuple[int, int], Any] = {}
    for row in rows:
        if row["mean_independent"] == "":
            continue
        key = (int(row["split_points"]), int(row["B"]))
        previous = independent_means.setdefault(key, row["mean_independent"])
        if previous != row["mean_independent"]:
            raise RuntimeError(
                f"Methods do not share the fixed_N baseline for "
                f"{model['directory']}, {key[0]} splits, B={key[1]}"
            )
        if not (
            row["reduction_ci_lower_percent"]
            <= row["reduction_percent"]
            <= row["reduction_ci_upper_percent"]
        ):
            raise RuntimeError(
                f"Invalid reduction confidence interval for {model['directory']}, "
                f"{key[0]} splits, B={key[1]}, {row['allocation']}"
            )
    return RESULT_FIELDS, rows


def _ou_oracle_rows(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    *,
    debug: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    learned_by_schedule = all_rows.get("simple_ou", {})
    oracle_by_schedule = all_rows.get("ou_oracle", {})
    rows: list[dict[str, Any]] = []
    for schedule, _ in SCHEDULES:
        oracle_rows = oracle_by_schedule.get(schedule, [])
        learned_rows = learned_by_schedule.get(schedule, [])
        common_budgets = {row.budget for row in oracle_rows} & {
            row.budget for row in learned_rows
        }
        if not common_budgets:
            if not debug:
                raise RuntimeError(
                    "OU learned and oracle results have no common budget"
                )
            rows.append(dict(zip(ORACLE_FIELDS, [len(schedule)] + [""] * 8)))
            continue
        display_budget = max(common_budgets)
        oracle = _row_at_budget(oracle_rows, display_budget, "ou_oracle")
        independent = _row_at_budget(learned_rows, display_budget, "fixed_N")
        learned = _row_at_budget(learned_rows, display_budget, "adaptive", "monotone")
        if debug:
            means = [
                "missing" if row is None else f"{row.mean:.12g}"
                for row in (independent, oracle, learned)
            ]
            print(
                f"[debug] OU KS sanity: splits={len(schedule)}, B={display_budget}, "
                f"shared fixed_N mean_ks={means[0]}, "
                f"oracle mean_ks={means[1]}, learned mean_ks={means[2]}"
            )
        if independent is None or oracle is None or learned is None:
            values = [""] * 7
        else:
            oracle_reduction = _normal_reduction(independent, oracle)
            learned_reduction = _normal_reduction(independent, learned)
            values = [
                *oracle_reduction,
                *learned_reduction,
                100.0 * learned_reduction[0] / oracle_reduction[0],
            ]
        rows.append(dict(zip(ORACLE_FIELDS, [len(schedule), display_budget, *values])))
    return ORACLE_FIELDS, rows


def _timing_table_rows(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    *,
    debug: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    allocations = {"monotone": "Learned", LEARNED_C_OPTIMIZER: LEARNED_C_ALLOCATION}
    expected = [(None, None)] + [
        (schedule, optimizer)
        for schedule, _ in SCHEDULES
        for _, optimizer in TIMING_METHODS
    ]
    rows: list[dict[str, Any]] = []
    for model in TIMING_MODELS:
        found = {
            (schedule, optimizer): row
            for schedule, optimizer, row in _timing_rows(model, all_rows)
        }
        independent = found.get((None, None))
        for schedule, optimizer in expected:
            allocation = "Independent" if optimizer is None else allocations[optimizer]
            prefix = [
                model["title"],
                "" if schedule is None else len(schedule),
                allocation,
                TIMING_BUDGET,
            ]
            row = found.get((schedule, optimizer))
            if row is None:
                if not debug:
                    raise RuntimeError(
                        f"Missing {allocation} timing for {model['directory']}, "
                        f"{'' if schedule is None else len(schedule)} splits"
                    )
                rows.append(dict(zip(TIMING_TABLE_FIELDS, prefix)))
                continue
            stages = [
                float(row.timings[field].mean()) if np.all(np.isfinite(row.timings[field])) else ""
                for field in TIMING_STAGES
            ]
            total = _timing_mean_ci(row.timings["total_seconds"])
            if not total[1] <= total[0] <= total[2]:
                raise RuntimeError(
                    f"Invalid timing confidence interval for {model['directory']}, "
                    f"{allocation}"
                )
            if row is independent or independent is None:
                overhead = ["", "", ""]
                pilot_share = ""
            else:
                overhead = list(_timing_overhead(independent, row))
                pilot_share = 100.0 * sum(
                    float(row.timings[field].mean()) for field in PILOT_STAGES
                ) / total[0]
            rows.append(
                dict(
                    zip(
                        TIMING_TABLE_FIELDS,
                        [
                            *prefix,
                            *stages,
                            *total,
                            float(row.timings["scoring_seconds"].mean()),
                            *overhead,
                            pilot_share,
                            row.n,
                        ],
                    )
                )
            )
    return TIMING_TABLE_FIELDS, rows


def main() -> None:
    args = _parse_args()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _load_and_verify_rows(outputs_root, debug=args.debug)

    fieldnames, rows = _numerical_schedule_rows()
    written = [_write_csv(output_dir, "numerical_schedules", fieldnames, rows)]

    fieldnames, rows = _reference_sample_rows()
    written.append(_write_csv(output_dir, "reference_samples", fieldnames, rows))

    fieldnames, rows = _ou_oracle_rows(all_rows, debug=args.debug)
    written.append(_write_csv(output_dir, "ou_oracle_reductions", fieldnames, rows))

    for model in MODELS:
        fieldnames, rows = _complete_result_rows(model, all_rows, debug=args.debug)
        stem = RESULT_TABLE_STEMS[model["directory"]]
        written.append(_write_csv(output_dir, stem, fieldnames, rows))

    fieldnames, rows = _timing_table_rows(all_rows, debug=args.debug)
    written.append(_write_csv(output_dir, "timing", fieldnames, rows))

    print(
        f"Validated publication records; wrote {len(written)} CSV tables "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
