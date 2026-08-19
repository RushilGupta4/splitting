#!/usr/bin/env python3
"""Validate completed experiment records and export the paper tables as CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from paper_plots import (
    MODELS,
    SCHEDULES,
    ResultRow,
    _expected_uniform_c,
    _load_and_verify_rows,
    _normal_reduction,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "tables"
RESULT_TABLE_STEMS = {
    "simple_ou": "complete_ou",
    "coupled_double_well_langevin": "complete_langevin",
    "edm_default": "complete_edm",
    "ddpm_cifar10_hf_mmd": "complete_ddpm",
}


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

    if ou["budgets"] != langevin["budgets"] or ou["steps"] != langevin["steps"]:
        raise RuntimeError("OU and Langevin must use the same Euler schedule")

    def step_map(model: dict[str, Any]) -> dict[int, int]:
        return dict(zip(model["budgets"], model["steps"]))

    euler_steps = step_map(ou)
    edm_steps = step_map(edm)
    ddpm_steps = step_map(ddpm)
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
    raise RuntimeError(f"Unsupported reference method {method!r}")


def _reference_sample_rows() -> tuple[list[str], list[dict[str, Any]]]:
    display_names = {
        "simple_ou": "OU process",
        "coupled_double_well_langevin": "Overdamped Langevin",
        "edm_default": "EDM",
        "ddpm_cifar10_hf_mmd": "CIFAR-10 DDPM",
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


def _row_at_budget(
    rows: list[ResultRow],
    budget: int,
    *,
    mode: str,
    uniform_c: float | None = None,
) -> ResultRow | None:
    matches = [
        row
        for row in rows
        if row.budget == budget
        and row.mode == mode
        and (uniform_c is None or row.uniform_c == uniform_c)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate {mode} result at B={budget}")
    return matches[0] if matches else None


def _metric_reduction(
    rows: list[ResultRow],
    budget: int,
    *,
    mode: str,
    uniform_c: float | None = None,
) -> tuple[float, float, float] | None:
    independent = _row_at_budget(rows, budget, mode="fixed_N")
    method = _row_at_budget(rows, budget, mode=mode, uniform_c=uniform_c)
    if independent is None or method is None:
        return None
    return _normal_reduction(independent, method)


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
    ) -> dict[str, Any]:
        schedule_rows = rows_by_schedule.get(schedule, [])
        reduction = _metric_reduction(
            schedule_rows, budget, mode=mode, uniform_c=uniform_c
        )
        if reduction is None:
            if not debug:
                raise RuntimeError(
                    f"Missing {allocation} result or fixed_N baseline for "
                    f"{model['directory']}, {len(schedule)} splits, B={budget}"
                )
            observed = lower = upper = ""
            n_method = n_independent = ""
            mean_method = mean_independent = ""
        else:
            observed, lower, upper = reduction
            independent = _row_at_budget(schedule_rows, budget, mode="fixed_N")
            method = _row_at_budget(
                schedule_rows, budget, mode=mode, uniform_c=uniform_c
            )
            assert independent is not None and method is not None
            n_method = method.n
            n_independent = independent.n
            mean_method = method.mean
            mean_independent = independent.mean
        return {
            "split_points": len(schedule),
            "allocation": allocation,
            "B": budget,
            "metric": model["metric"],
            "mean_method": mean_method,
            "mean_independent": mean_independent,
            "reduction_percent": observed,
            "reduction_ci_lower_percent": lower,
            "reduction_ci_upper_percent": upper,
            "n_method": n_method,
            "n_independent": n_independent,
        }

    rows: list[dict[str, Any]] = []
    expected_keys: list[tuple[int, str, int]] = []
    for schedule, _ in SCHEDULES:
        for budget in model["budgets"]:
            expected_keys.append((len(schedule), "Learned", budget))
            rows.append(
                result_row(schedule, budget, allocation="Learned", mode="adaptive")
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
    fieldnames = [
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
    return fieldnames, rows


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
            rows.append(
                {
                    "split_points": len(schedule),
                    "B": "",
                    "oracle_ks_reduction_percent": "",
                    "oracle_ks_reduction_ci_lower_percent": "",
                    "oracle_ks_reduction_ci_upper_percent": "",
                    "learned_ks_reduction_percent": "",
                    "learned_ks_reduction_ci_lower_percent": "",
                    "learned_ks_reduction_ci_upper_percent": "",
                    "learned_to_oracle_reduction_ratio_percent": "",
                }
            )
            continue
        display_budget = max(common_budgets)
        oracle = _row_at_budget(oracle_rows, display_budget, mode="ou_oracle")
        shared_independent = _row_at_budget(
            learned_rows, display_budget, mode="fixed_N"
        )
        learned = _row_at_budget(
            learned_rows,
            display_budget,
            mode="adaptive",
        )
        if debug:
            shared_fixed_mean = (
                "missing"
                if shared_independent is None
                else f"{shared_independent.mean:.12g}"
            )
            oracle_mean = "missing" if oracle is None else f"{oracle.mean:.12g}"
            learned_mean = "missing" if learned is None else f"{learned.mean:.12g}"
            print(
                f"[debug] OU KS sanity: splits={len(schedule)}, B={display_budget}, "
                f"shared fixed_N mean_ks={shared_fixed_mean}, "
                f"oracle mean_ks={oracle_mean}, learned mean_ks={learned_mean}"
            )
        if any(value is None for value in (shared_independent, oracle, learned)):
            oracle_reduction = learned_reduction = oracle_captured = ""
            oracle_lower = oracle_upper = learned_lower = learned_upper = ""
        else:
            assert shared_independent is not None and oracle is not None
            assert learned is not None
            oracle_reduction, oracle_lower, oracle_upper = _normal_reduction(
                shared_independent, oracle
            )
            learned_reduction, learned_lower, learned_upper = _normal_reduction(
                shared_independent, learned
            )
            oracle_captured = 100.0 * learned_reduction / oracle_reduction

        rows.append(
            {
                "split_points": len(schedule),
                "B": display_budget,
                "oracle_ks_reduction_percent": oracle_reduction,
                "oracle_ks_reduction_ci_lower_percent": oracle_lower,
                "oracle_ks_reduction_ci_upper_percent": oracle_upper,
                "learned_ks_reduction_percent": learned_reduction,
                "learned_ks_reduction_ci_lower_percent": learned_lower,
                "learned_ks_reduction_ci_upper_percent": learned_upper,
                "learned_to_oracle_reduction_ratio_percent": oracle_captured,
            }
        )
    fieldnames = [
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
    return fieldnames, rows


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
        written.append(
            _write_csv(
                output_dir,
                RESULT_TABLE_STEMS[model["directory"]],
                fieldnames,
                rows,
            )
        )

    print(
        f"Validated publication records; wrote {len(written)} CSV tables "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
