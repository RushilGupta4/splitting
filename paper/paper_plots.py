#!/usr/bin/env python3
"""Validate completed experiment records and regenerate the paper assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

Z_975 = 1.96
OUTPUT_SCALE = 1.5

# Shared styling for the three four-model, 2x2 publication figures.  Keeping
# these values in one place prevents small visual differences between panels
# that are intended to be read as a set.
FOUR_MODEL_FIGSIZE = (64.0 / 9.0, 4.0)
# Full-HD output for the standardized 16:9 figures.
FOUR_MODEL_OUTPUT_WIDTH = 1_920
FOUR_MODEL_DPI = FOUR_MODEL_OUTPUT_WIDTH / FOUR_MODEL_FIGSIZE[0]
OU_ORACLE_DPI = 276.225
DATA_LINEWIDTH = 1.5
REFERENCE_LINEWIDTH = 0.8
GRID_LINEWIDTH = 0.55
INTERVAL_ALPHA = 0.14
FOUR_MODEL_LAYOUT = {
    "rect": (0.025, 0.055, 0.99, 0.92),
    "h_pad": 0.5,
    "w_pad": 0.8,
}

FOUR = (0.8, 0.6, 0.4, 0.2)
NINE = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
SINGLE_01 = (0.1,)
SINGLE_02 = (0.2,)
NINETEEN = tuple(round(value / 20, 2) for value in range(19, 0, -1))
SCHEDULES = (
    (FOUR, "4 splits"),
    (NINE, "9 splits"),
    (NINETEEN, "19 splits"),
)
SCHEDULE_COLORS = {
    SINGLE_01: "#009E73",
    SINGLE_02: "#CC79A7",
    FOUR: "#0072B2",
    NINE: "#D55E00",
    NINETEEN: "#228833",
}

# Colour encodes the split count, dash pattern the allocation: learned stays solid.
UNIFORM_C_BY_SPLIT_COUNT = {4: (1.1, 1.25, 1.5), 9: (1.1, 1.25), 19: (1.1,)}
UNIFORM_C_LINESTYLES = {
    1.1: (0, (1, 1.2)),
    1.25: (0, (4, 1.5)),
    1.5: (0, (6, 1.5, 1, 1.5)),
}
UNIFORM_C_VALUES = tuple(
    sorted({value for values in UNIFORM_C_BY_SPLIT_COUNT.values() for value in values})
)
UNIFORM_C_DIRECTORIES = frozenset(
    {"simple_ou", "coupled_double_well_langevin", "edm_default"}
)

EXPECTED_MLP_PARAMS = {
    "hidden_dims": [128, 64],
    "activation": "silu",
    "epochs": 5,
    "batch_size": 24_000,
    "lr": 0.005,
    "weight_decay": 0.0003,
    "loss": "mse",
    "device": "runner",
    "num_threads": 2,
}
EXPECTED_QUERY_PARAMS = {
    "k_max": 64,
    "mass_min": 0.05,
    "mass_max": 0.95,
}

OU_TARGET = {
    "name": "simple_ou",
    "label": "Simple OU",
    "dimension": 2,
    "coupling_strength": 0.25,
    "terminal_time": 1.0,
    "initial_distribution": {
        "kind": "diagonal_normal",
        "dimension": 2,
        "mean": [0.0, 0.0],
        "variance": [1.0, 1.0],
    },
    "params": {
        "theta": 1.35,
        "mu": -0.2,
        "sigma": 0.65,
        "initial_distribution": {
            "kind": "normal",
            "mean": 0.0,
            "variance": 1.0,
        },
    },
    "cdf": None,
}

LANGEVIN_TARGET = {
    "name": "coupled_double_well_langevin",
    "label": "Coupled double-well overdamped Langevin",
    "dimension": 2,
    "coupling_strength": 0.0,
    "terminal_time": 2.0,
    "topology": "periodic_nearest_neighbor_ring",
    "initial_distribution": {
        "kind": "diagonal_normal",
        "dimension": 2,
        "mean": [-1.0, -1.0],
        "variance": [0.05, 0.05],
    },
    "params": {
        "barrier_coefficient": 4.0,
        "ring_coupling": 0.5,
        "inverse_temperature": 1.0,
        "diffusion_scale": math.sqrt(2.0),
    },
    "cdf": None,
}

EDM_TARGET = {
    "name": "gaussian_mixture_2d",
    "weights": [0.5, 0.5],
    "means": [[-1.0, 1.0], [1.0, -1.0]],
    "covariances": [
        [[0.2, 0.05], [0.05, 0.3]],
        [[0.3, -0.08], [-0.08, 0.15]],
    ],
}

MODELS = (
    {
        "directory": "simple_ou",
        "runner": "simple_ou",
        "config_name": "default",
        "title": "OU Process",
        "plot_title": "OU Process (KS)",
        "metric": "ks",
        "n_runs": 2_500,
        "num_queries": 1_024,
        "pilot_coefficient": 5.0,
        "pilot_exponent": 0.66,
        "budgets": (100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000),
        "steps": (130, 160, 220, 280, 350, 475),
        "sampler": "euler",
        "reference_size": 2_500_000,
        "target": OU_TARGET,
        "reference_generation": {
            "method": "sde_terminal_samples",
            "sampler": "euler",
            "sampling_steps": 20_000,
            "terminal_time": 1.0,
        },
    },
    {
        "directory": "coupled_double_well_langevin",
        "runner": "coupled_double_well_langevin",
        "config_name": "default",
        "title": "Overdamped Langevin",
        "plot_title": "Overdamped Langevin (KS)",
        "metric": "ks",
        "n_runs": 2_500,
        "num_queries": 1_024,
        "pilot_coefficient": 5.0,
        "pilot_exponent": 0.66,
        "budgets": (100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000),
        "steps": (130, 160, 220, 280, 350, 475),
        "sampler": "euler",
        "reference_size": 2_500_000,
        "target": LANGEVIN_TARGET,
        "reference_generation": {
            "method": "sde_terminal_samples",
            "sampler": "euler",
            "sampling_steps": 20_000,
            "terminal_time": 2.0,
        },
    },
    {
        "directory": "edm_default",
        "runner": "edm_gmm2d",
        "config_name": "default",
        "title": "EDM Gaussian mixture",
        "plot_title": "EDM Gaussian mixture (KS)",
        "metric": "ks",
        "n_runs": 2_500,
        "num_queries": 1_024,
        "pilot_coefficient": 10.0,
        "pilot_exponent": 0.66,
        "budgets": (
            100_000,
            200_000,
            500_000,
            1_000_000,
            2_000_000,
            5_000_000,
        ),
        "steps": (40, 45, 55, 63, 73, 87),
        "sampler": "edm_stochastic",
        "reference_size": 5_000_000,
        "target": EDM_TARGET,
        "reference_generation": {"method": "target_samples"},
    },
    {
        "directory": "ddpm_cifar10_hf_mmd",
        "runner": "ddpm_cifar10_hf",
        "config_name": "mmd",
        "title": "CIFAR-10 DDPM",
        "plot_title": "CIFAR-10 DDPM (MMD)",
        "metric": "mmd",
        "n_runs": 50,
        "num_queries": 1_024,
        "pilot_coefficient": 10.0,
        "pilot_exponent": 0.66,
        "budgets": (50_000, 100_000, 200_000, 500_000, 1_000_000),
        "steps": (200, 252, 317, 430, 542),
        "sampler": "ddpm",
        "reference_size": 20_000,
        "target": None,
        "reference_generation": {
            "method": "hf_ddpm_scheduler",
            "sampler": "ddpm",
            "T": 1_000,
            "sampling_steps": 1_000,
            "timestep_spacing": "trailing",
            "seed": 0,
        },
    },
)

OU_ORACLE_MODEL = {
    **MODELS[0],
    "directory": "ou_oracle",
    "runner": "ou_oracle",
    "config_name": "default",
    "title": "OU oracle",
    "n_runs": 10_000,
    "budgets": (MODELS[0]["budgets"][-1],),
    "steps": (MODELS[0]["steps"][-1],),
}


@dataclass
class ResultRow:
    model: dict[str, Any]
    schedule: tuple[float, ...]
    mode: str
    budget: int
    pilot_budget: int | None
    pilot_rule: str | None
    sampler: str
    sampling_steps: int
    optimizer: str | None
    loss: str | None
    reuse: bool | None
    mean: float
    std: float
    n: int
    csv_path: Path
    raw: dict[str, str]
    samples: np.ndarray | None = None
    split_factors: np.ndarray | None = None
    uniform_c: float | None = None

    @property
    def is_adaptive(self) -> bool:
        return self.mode == "adaptive"

    @property
    def is_independent(self) -> bool:
        return self.mode == "fixed_N"

    @property
    def is_uniform_c(self) -> bool:
        return self.mode == "uniform_c"

    @property
    def is_oracle(self) -> bool:
        return self.mode == "ou_oracle"

    def mean_ci(self) -> tuple[float, float]:
        if self.n < 2 or not math.isfinite(self.std):
            return self.mean, self.mean
        half = Z_975 * self.std / math.sqrt(self.n)
        return self.mean - half, self.mean + half


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
        default=ROOT / "plots",
        help="Destination for generated PNG figures.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Plot available partial runs and schedules instead of failing.",
    )
    return parser.parse_args()


def _as_bool(value: str) -> bool | None:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Unexpected boolean value {value!r}")


def _schedule_name(schedule: tuple[float, ...]) -> str:
    return "_".join(f"{value:g}" for value in schedule)


def _schedule_from_filename(path: Path) -> tuple[float, ...]:
    expected = {
        f"compare_results_{_schedule_name(schedule)}": schedule
        for schedule, _ in SCHEDULES
    }
    try:
        return expected[path.stem]
    except KeyError as exc:
        raise ValueError(
            f"Unexpected split schedule in {path.name}; expected one of the "
            "configured one-, four-, nine-, or nineteen-split schedules"
        ) from exc


def _expected_uniform_c(
    model: dict[str, Any], schedule: tuple[float, ...]
) -> tuple[float, ...]:
    if model["directory"] not in UNIFORM_C_DIRECTORIES:
        return ()
    return UNIFORM_C_BY_SPLIT_COUNT[len(schedule)]


def _uniform_c_factor(
    raw: dict[str, str], schedule: tuple[float, ...], csv_path: Path
) -> float:
    """Recover c from a constant-allocation row; N_i is exactly [c] * len(schedule)."""
    factors = [float(value) for value in raw["N_i"].split(",")]
    factor_stds = [float(value) for value in raw["N_i_std"].split(",")]
    if (
        len(factors) != len(schedule)
        or len(factor_stds) != len(schedule)
        or any(abs(value - factors[0]) > 1e-9 for value in factors)
        or any(value != 0.0 for value in factor_stds)
        or raw["optimizer"] != "uniform"
        or int(raw["n0"]) < 1
    ):
        raise RuntimeError(f"Invalid uniform_c row in {csv_path}")
    return factors[0]


def _resolved_pilot_budget(model: dict[str, Any], budget: int) -> int:
    value = model["pilot_coefficient"] * budget ** model["pilot_exponent"]
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest)
    return math.floor(value)


def _load_manifest_rows(
    outputs_root: Path,
    models: tuple[dict[str, Any], ...] = MODELS,
    *,
    allow_partial_runs: bool = False,
    allow_partial_schedules: bool = False,
) -> dict[str, dict[tuple[float, ...], list[ResultRow]]]:
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]] = {}
    expected_csv_names = {
        f"compare_results_{_schedule_name(schedule)}.csv" for schedule, _ in SCHEDULES
    }

    for model in models:
        partial_run_counts: set[int] = set()
        skipped_zero_run_rows = 0
        experiment_dir = outputs_root / model["directory"]
        manifest_path = experiment_dir / "compare_outputs.json"
        if allow_partial_schedules:
            listed = [
                str(experiment_dir / name)
                for name in sorted(expected_csv_names)
                if (experiment_dir / name).is_file()
            ]
            if not listed:
                raise RuntimeError(
                    f"No recognized split-schedule CSVs in {experiment_dir}"
                )
        else:
            with manifest_path.open() as handle:
                manifest = json.load(handle)
            if manifest.get("runner_name") != model["runner"]:
                raise RuntimeError(f"Unexpected runner in {manifest_path}")
            if manifest.get("config_name") != model["config_name"]:
                raise RuntimeError(f"Unexpected config in {manifest_path}")

            listed = manifest.get("csv_files", [])
            if (
                len(listed) != len(SCHEDULES)
                or {Path(value).name for value in listed} != expected_csv_names
            ):
                raise RuntimeError(
                    f"{manifest_path} must list all {len(SCHEDULES)} completed "
                    "split-schedule CSVs"
                )

        schedule_rows: dict[tuple[float, ...], list[ResultRow]] = {}
        for listed_path_text in listed:
            csv_path = experiment_dir / Path(listed_path_text).name
            schedule = _schedule_from_filename(csv_path)
            rows: list[ResultRow] = []
            with csv_path.open(newline="") as handle:
                for raw in csv.DictReader(handle):
                    metric = model["metric"]
                    metric_n = raw.get(f"n_valid_{metric}", "")
                    n = int(metric_n or raw["n_valid_ks"])
                    if allow_partial_runs and n == 0:
                        skipped_zero_run_rows += 1
                        continue
                    if n != model["n_runs"]:
                        if not allow_partial_runs or not 1 <= n < model["n_runs"]:
                            raise RuntimeError(
                                f"{csv_path}: expected {model['n_runs']} runs, "
                                f"found {n}"
                            )
                        partial_run_counts.add(n)
                    row = ResultRow(
                        model=model,
                        schedule=schedule,
                        mode=raw["mode"],
                        budget=int(raw["B"]),
                        pilot_budget=int(raw["B1"]) if raw["B1"] else None,
                        pilot_rule=raw["B1_spec"] or None,
                        sampler=raw["sampler"],
                        sampling_steps=int(raw["sampling_steps"]),
                        optimizer=raw["optimizer"] or None,
                        loss=raw["crossfit_q_mlp_loss"] or None,
                        reuse=_as_bool(raw["reuse"]),
                        mean=float(raw[f"mean_{metric}"]),
                        std=float(raw[f"std_{metric}"]),
                        n=n,
                        csv_path=csv_path,
                        raw=raw,
                    )
                    if row.mode not in {"adaptive", "fixed_N", "uniform_c"}:
                        raise RuntimeError(
                            f"Unexpected mode {row.mode!r} in {csv_path}"
                        )
                    if row.is_uniform_c:
                        row.uniform_c = _uniform_c_factor(raw, schedule, csv_path)
                    rows.append(row)

            budgets = {row.budget for row in rows}
            expected_budgets = set(model["budgets"])
            if not budgets.issubset(expected_budgets) or (
                not allow_partial_runs and budgets != expected_budgets
            ):
                raise RuntimeError(f"Unexpected budget set in {csv_path}: {budgets}")
            expected_steps = dict(zip(model["budgets"], model["steps"]))
            for budget in sorted(budgets):
                group = [row for row in rows if row.budget == budget]
                adaptive_rows = [row for row in group if row.is_adaptive]
                independent_rows = [row for row in group if row.is_independent]
                if len(adaptive_rows) > 1 or len(independent_rows) > 1:
                    raise RuntimeError(f"Duplicate method row at B={budget}")
                if not allow_partial_runs and (
                    len(adaptive_rows) != 1 or len(independent_rows) != 1
                ):
                    raise RuntimeError(
                        f"Expected one adaptive and one independent row at B={budget}"
                    )
                expected_c = tuple(sorted(_expected_uniform_c(model, schedule)))
                observed_c = tuple(
                    sorted(row.uniform_c for row in group if row.is_uniform_c)
                )
                if len(observed_c) != len(set(observed_c)):
                    raise RuntimeError(f"Duplicate uniform_c row at B={budget}")
                if any(value not in expected_c for value in observed_c) or (
                    not allow_partial_runs and observed_c != expected_c
                ):
                    raise RuntimeError(
                        f"{csv_path}: uniform_c set at B={budget} is {observed_c}, "
                        f"expected {expected_c}; rerun compare.py for this experiment"
                    )
                if any(
                    row.sampler != model["sampler"]
                    or row.sampling_steps != expected_steps[budget]
                    for row in group
                ):
                    raise RuntimeError(f"Unexpected numerical schedule at B={budget}")
                expected_pilot = _resolved_pilot_budget(model, budget)
                expected_rule = (
                    f"power:{model['pilot_coefficient']:g},{model['pilot_exponent']:g}"
                )
                if adaptive_rows:
                    splitting = adaptive_rows[0]
                    if (
                        splitting.pilot_rule != expected_rule
                        or splitting.pilot_budget != expected_pilot
                        or splitting.optimizer != "monotone"
                        or splitting.loss != "mse"
                        or splitting.reuse is not True
                    ):
                        raise RuntimeError(f"Unexpected splitting rule at B={budget}")
                if independent_rows:
                    independent = independent_rows[0]
                    if (
                        independent.pilot_budget is not None
                        or independent.pilot_rule is not None
                    ):
                        raise RuntimeError(f"Independent row has a pilot at B={budget}")
            if rows:
                schedule_rows[schedule] = rows

        expected_schedules = {schedule for schedule, _ in SCHEDULES}
        if not allow_partial_schedules and set(schedule_rows) != expected_schedules:
            raise RuntimeError(f"Missing split-schedule data in {manifest_path}")
        missing_schedules = expected_schedules - set(schedule_rows)
        if missing_schedules:
            missing_labels = ", ".join(
                label for schedule, label in SCHEDULES if schedule in missing_schedules
            )
            warnings.warn(
                f"Loading partial {model['title']} results; "
                f"missing schedules: {missing_labels}.",
                RuntimeWarning,
                stacklevel=2,
            )
        if partial_run_counts:
            low = min(partial_run_counts)
            high = max(partial_run_counts)
            observed = str(low) if low == high else f"between {low} and {high}"
            warnings.warn(
                f"Loading partial {model['title']} results: "
                f"expected {model['n_runs']} runs, found {observed}.",
                RuntimeWarning,
                stacklevel=2,
            )
        if skipped_zero_run_rows:
            warnings.warn(
                f"Loading partial {model['title']} results: skipped "
                f"{skipped_zero_run_rows} rows with no valid runs.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not schedule_rows:
            raise RuntimeError(f"No usable partial results in {experiment_dir}")
        all_rows[model["directory"]] = schedule_rows
    return all_rows


def _load_ou_oracle_rows(
    outputs_root: Path,
    *,
    allow_partial_runs: bool = False,
    allow_partial_schedules: bool = False,
) -> dict[tuple[float, ...], list[ResultRow]]:
    model = OU_ORACLE_MODEL
    experiment_dir = outputs_root / model["directory"]
    manifest_path = experiment_dir / "compare_outputs.json"
    expected_names = {
        f"compare_results_{_schedule_name(schedule)}.csv" for schedule, _ in SCHEDULES
    }
    if allow_partial_schedules:
        listed = [
            str(experiment_dir / name)
            for name in sorted(expected_names)
            if (experiment_dir / name).is_file()
        ]
        if not listed:
            raise RuntimeError(f"No recognized oracle CSVs in {experiment_dir}")
    else:
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        if (
            manifest.get("runner_name") != model["runner"]
            or manifest.get("config_name") != model["config_name"]
        ):
            raise RuntimeError(f"Unexpected OU oracle manifest {manifest_path}")
        listed = manifest.get("csv_files", [])
        if (
            len(listed) != len(expected_names)
            or {Path(value).name for value in listed} != expected_names
        ):
            raise RuntimeError(f"{manifest_path} does not list every oracle schedule")

    expected_steps = dict(zip(model["budgets"], model["steps"]))
    by_schedule: dict[tuple[float, ...], list[ResultRow]] = {}
    partial_run_counts: set[int] = set()
    skipped_zero_run_rows = 0
    for listed_path in listed:
        csv_path = experiment_dir / Path(listed_path).name
        schedule = _schedule_from_filename(csv_path)
        rows: list[ResultRow] = []
        with csv_path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                if raw["mode"] not in {"fixed_N", "ou_oracle"}:
                    raise RuntimeError(f"Unexpected oracle mode in {csv_path}")
                if raw["mode"] == "fixed_N":
                    # Legacy oracle runs included a separate independent baseline.
                    # Paper comparisons always use the ordinary Simple OU fixed_N row.
                    continue
                n = int(raw["n_valid_ks"])
                if allow_partial_runs and n == 0:
                    skipped_zero_run_rows += 1
                    continue
                if n != model["n_runs"] and not (
                    allow_partial_runs and 1 <= n < model["n_runs"]
                ):
                    raise RuntimeError(
                        f"{csv_path}: expected {model['n_runs']} runs, found {n}"
                    )
                if n != model["n_runs"]:
                    partial_run_counts.add(n)
                row = ResultRow(
                    model=model,
                    schedule=schedule,
                    mode=raw["mode"],
                    budget=int(raw["B"]),
                    pilot_budget=None,
                    pilot_rule=None,
                    sampler=raw["sampler"],
                    sampling_steps=int(raw["sampling_steps"]),
                    optimizer=raw["optimizer"] or None,
                    loss=None,
                    reuse=None,
                    mean=float(raw["mean_ks"]),
                    std=float(raw["std_ks"]),
                    n=n,
                    csv_path=csv_path,
                    raw=raw,
                )
                if row.is_oracle:
                    factors = [float(value) for value in raw["N_i"].split(",")]
                    factor_stds = [float(value) for value in raw["N_i_std"].split(",")]
                    if (
                        len(factors) != len(schedule)
                        or any(value < 1.0 for value in factors)
                        or len(factor_stds) != len(schedule)
                        or any(value != 0.0 for value in factor_stds)
                        or row.optimizer != "finite_query_minimax"
                        or float(raw["oracle_gap"]) > 1e-6
                        or int(raw["n0"]) < 1
                    ):
                        raise RuntimeError(f"Invalid OU oracle row in {csv_path}")
                rows.append(row)

        budgets = {row.budget for row in rows}
        expected_budgets = set(model["budgets"])
        if not budgets.issubset(expected_budgets) or (
            not allow_partial_runs and budgets != expected_budgets
        ):
            raise RuntimeError(f"Unexpected oracle budgets in {csv_path}")
        for budget in sorted(budgets):
            group = [row for row in rows if row.budget == budget]
            oracle_count = sum(row.is_oracle for row in group)
            if (
                oracle_count > 1
                or (not allow_partial_runs and oracle_count != 1)
                or any(
                    row.sampler != model["sampler"]
                    or row.sampling_steps != expected_steps[budget]
                    for row in group
                )
            ):
                raise RuntimeError(f"Invalid OU oracle methods at B={budget}")
        if rows:
            by_schedule[schedule] = rows

    expected_schedules = {schedule for schedule, _ in SCHEDULES}
    if not allow_partial_schedules and set(by_schedule) != expected_schedules:
        raise RuntimeError(f"Missing OU oracle schedules in {manifest_path}")
    missing_schedules = expected_schedules - set(by_schedule)
    if missing_schedules:
        missing_labels = ", ".join(
            label for schedule, label in SCHEDULES if schedule in missing_schedules
        )
        warnings.warn(
            f"Loading partial OU oracle results; missing schedules: "
            f"{missing_labels}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if partial_run_counts:
        low = min(partial_run_counts)
        high = max(partial_run_counts)
        observed = str(low) if low == high else f"between {low} and {high}"
        warnings.warn(
            f"Loading partial OU oracle results: expected {model['n_runs']} runs, "
            f"found {observed}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if skipped_zero_run_rows:
        warnings.warn(
            f"Loading partial OU oracle results: skipped {skipped_zero_run_rows} "
            "rows with no valid runs.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not by_schedule:
        raise RuntimeError(f"No usable partial oracle results in {experiment_dir}")
    return by_schedule


def _cifar_target_is_expected(target: dict[str, Any]) -> bool:
    return (
        target.get("kind") == "hf_ddpm_cifar10_model_samples"
        and target.get("model_id") == "google/ddpm-cifar10-32"
        and target.get("image_shape") == [3, 32, 32]
        and target.get("sample_dim") == 3_072
        and target.get("postprocess") == "clamp_0_1_flat_v1"
    )


def _metric_config_is_expected(model: dict[str, Any], config: dict[str, Any]) -> bool:
    metric_config = config.get("metric_config", {})
    if metric_config.get("metrics") != [model["metric"]]:
        return False
    if model["metric"] == "ks":
        return metric_config == {"metrics": ["ks"]}
    mmd = metric_config.get("mmd", {})
    return (
        mmd.get("kind") == "target_space_random_fourier_mmd"
        and mmd.get("version") == 1
        and mmd.get("estimator") == "biased_empirical_root"
        and mmd.get("kernel") == "equal_weight_multiscale_gaussian"
        and mmd.get("standardization") == "reference_coordinate_zscore_v1"
        and mmd.get("representation") == "quantized_image_target"
        and mmd.get("num_frequencies") == 1_024
        and mmd.get("bandwidth_multipliers")
        == [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        and mmd.get("bandwidth_pairs") == 8_192
        and mmd.get("seed") == 0
        and mmd.get("quantize_images") is True
    )


def _sampling_config_is_expected(
    model: dict[str, Any], sampling: dict[str, Any], steps: int
) -> bool:
    if model["sampler"] == "euler":
        return sampling == {
            "sampler": "euler",
            "sampling_steps": steps,
            "terminal_time": model["target"]["terminal_time"],
        }
    if model["sampler"] == "edm_stochastic":
        return sampling == {
            "sampler": "edm_stochastic",
            "sampling_steps": steps,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 3.0,
            "sampler_params": {
                "S_churn": 40.0,
                "S_min": 0.0,
                "S_max": 80.0,
                "S_noise": 1.0,
            },
        }
    return sampling == {
        "sampler": "ddpm",
        "T": 1_000,
        "sampling_steps": steps,
        "timestep_spacing": "trailing",
    }


def _static_config_is_expected(model: dict[str, Any], key: dict[str, Any]) -> bool:
    target = key.get("runner_target", {})
    reference = key.get("reference_cache_key", {})
    if model["runner"] == "ddpm_cifar10_hf":
        targets_match = _cifar_target_is_expected(target) and _cifar_target_is_expected(
            reference.get("target_spec", {})
        )
    else:
        targets_match = (
            target == model["target"]
            and reference.get("target_spec") == model["target"]
        )
    return (
        key.get("runner") == model["runner"]
        and key.get("comparison_mode") == "true_samples"
        and targets_match
        and _metric_config_is_expected(model, key)
        and key.get("num_base_samples") == model["reference_size"]
        and reference.get("reference_generation_config")
        == model["reference_generation"]
    )


def _config_matches(row: ResultRow, config: dict[str, Any]) -> bool:
    key = config.get("cache_key", {})
    spec = key.get("spec", {})
    if not _static_config_is_expected(row.model, key):
        return False
    if spec.get("B") != row.budget or not _sampling_config_is_expected(
        row.model, spec.get("runner_sampling_config", {}), row.sampling_steps
    ):
        return False
    if row.is_independent:
        return spec.get("mode") == "fixed_N"

    if row.is_uniform_c:
        return (
            spec.get("mode") == "uniform_c"
            and tuple(float(value) for value in spec.get("split_percentages", []))
            == row.schedule
            and math.isclose(
                float(spec.get("uniform_c", math.nan)),
                float(row.uniform_c),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

    return (
        spec.get("mode") == "estimate_and_sample"
        and tuple(float(value) for value in spec.get("split_percentages", []))
        == row.schedule
        and spec.get("B1") == row.pilot_budget
        and spec.get("optimization_mode") == row.optimizer
        and spec.get("reuse_phase1_samples") == row.reuse
        and key.get("query_params")
        == {"num_queries": row.model["num_queries"], **EXPECTED_QUERY_PARAMS}
        and key.get("crossfit_q_mlp_params") == EXPECTED_MLP_PARAMS
    )


def _read_valid_records(
    path: Path, metric: str, n_runs: int
) -> tuple[np.ndarray, np.ndarray | None]:
    values: list[float] = []
    factors: list[list[float]] = []
    factor_presence: bool | None = None
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            value = record.get("metrics", {}).get(metric)
            if value is None and metric == "ks":
                value = record.get("ks_distance")
            if value is None or not math.isfinite(float(value)):
                continue
            has_factors = "N_i" in record
            if factor_presence is None:
                factor_presence = has_factors
            if factor_presence != has_factors:
                raise RuntimeError(f"Inconsistent N_i presence in {path}")
            values.append(float(value))
            if has_factors:
                factors.append([float(item) for item in record["N_i"]])
            if len(values) == n_runs:
                break
    if len(values) != n_runs:
        raise RuntimeError(f"{path} contains only {len(values)} valid {metric} records")
    factor_array = np.asarray(factors, dtype=float) if factors else None
    return np.asarray(values, dtype=float), factor_array


def _attach_and_verify_caches(
    outputs_root: Path,
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    models: tuple[dict[str, Any], ...] = MODELS,
) -> None:
    record_cache: dict[tuple[Path, str, int], tuple[np.ndarray, np.ndarray | None]] = {}
    for model in models:
        run_root = outputs_root / model["directory"] / "runs"
        configs: list[tuple[Path, dict[str, Any]]] = []
        for config_path in sorted(run_root.glob("*/config.json")):
            with config_path.open() as handle:
                configs.append((config_path, json.load(handle)))

        for rows in all_rows[model["directory"]].values():
            for row in rows:
                matches: list[tuple[np.ndarray, np.ndarray | None]] = []
                for config_path, config in configs:
                    if not _config_matches(row, config):
                        continue
                    runs_path = config_path.with_name("runs.jsonl")
                    if not runs_path.is_file():
                        continue
                    cache_key = (runs_path, model["metric"], row.n)
                    if cache_key not in record_cache:
                        record_cache[cache_key] = _read_valid_records(
                            runs_path, model["metric"], row.n
                        )
                    samples, factors = record_cache[cache_key]
                    mean_matches = math.isclose(
                        float(samples.mean()), row.mean, rel_tol=1e-11, abs_tol=1e-13
                    )
                    std_matches = row.n == 1 or math.isclose(
                        float(samples.std(ddof=1)),
                        row.std,
                        rel_tol=1e-11,
                        abs_tol=1e-13,
                    )
                    if mean_matches and std_matches:
                        matches.append((samples, factors))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one validated cache for {row.csv_path.name}, "
                        f"B={row.budget}, B1={row.pilot_budget}; found {len(matches)}"
                    )
                row.samples, row.split_factors = matches[0]
                if row.is_uniform_c:
                    if row.split_factors is None:
                        raise RuntimeError("uniform_c cache is missing N_i")
                    if not np.allclose(
                        row.split_factors, row.uniform_c, rtol=0.0, atol=1e-9
                    ):
                        raise RuntimeError(
                            f"uniform_c N_i mismatch for {row.csv_path.name}, "
                            f"B={row.budget}"
                        )
                if row.is_adaptive:
                    if row.split_factors is None:
                        raise RuntimeError("Adaptive cache is missing N_i")
                    csv_mean = np.asarray(
                        [float(value) for value in row.raw["N_i"].split(",")]
                    )
                    csv_std = np.asarray(
                        [float(value) for value in row.raw["N_i_std"].split(",")]
                    )
                    mean_matches = np.allclose(
                        row.split_factors.mean(axis=0), csv_mean, rtol=6e-6, atol=6e-6
                    )
                    std_matches = row.n == 1 or np.allclose(
                        row.split_factors.std(axis=0, ddof=1),
                        csv_std,
                        rtol=6e-6,
                        atol=6e-6,
                    )
                    if not mean_matches or not std_matches:
                        raise RuntimeError(
                            f"N_i aggregate mismatch for {row.csv_path.name}, B={row.budget}"
                        )


def _load_and_verify_rows(
    outputs_root: Path,
    debug: bool,
) -> dict[str, dict[tuple[float, ...], list[ResultRow]]]:
    if not debug:
        all_rows = _load_manifest_rows(outputs_root)
        _attach_and_verify_caches(outputs_root, all_rows)
        all_rows["ou_oracle"] = _load_ou_oracle_rows(outputs_root)
        return all_rows

    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]] = {}
    for model in MODELS:
        try:
            model_rows = _load_manifest_rows(
                outputs_root,
                models=(model,),
                allow_partial_runs=True,
                allow_partial_schedules=True,
            )
            _attach_and_verify_caches(outputs_root, model_rows, models=(model,))
        except (OSError, KeyError, RuntimeError, ValueError) as exc:
            warnings.warn(
                f"Skipping {model['title']} in debug mode: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        all_rows.update(model_rows)
    try:
        all_rows["ou_oracle"] = _load_ou_oracle_rows(
            outputs_root,
            allow_partial_runs=True,
            allow_partial_schedules=True,
        )
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        warnings.warn(
            f"Skipping OU oracle in debug mode: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return all_rows


def _row_at_budget(rows: list[ResultRow], budget: int, mode: str) -> ResultRow | None:
    matches = [row for row in rows if row.budget == budget and row.mode == mode]
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate {mode} result at B={budget}")
    return matches[0] if matches else None


def _validated_independent_rows(
    rows_by_schedule: dict[tuple[float, ...], list[ResultRow]],
) -> dict[int, ResultRow]:
    """Return one independent baseline per budget after checking duplicates."""
    independent_by_budget: dict[int, ResultRow] = {}
    for schedule, _ in SCHEDULES:
        if schedule not in rows_by_schedule:
            continue
        for row in rows_by_schedule[schedule]:
            if not row.is_independent:
                continue
            reference = independent_by_budget.get(row.budget)
            if reference is None:
                independent_by_budget[row.budget] = row
                continue
            metadata_match = (
                row.sampler == reference.sampler
                and row.sampling_steps == reference.sampling_steps
            )
            shared_n = min(row.n, reference.n)
            samples_share_prefix = (
                row.samples is not None
                and reference.samples is not None
                and np.array_equal(row.samples[:shared_n], reference.samples[:shared_n])
            )
            same_summary = row.n != reference.n or (
                math.isclose(row.mean, reference.mean, rel_tol=0.0, abs_tol=1e-15)
                and math.isclose(row.std, reference.std, rel_tol=0.0, abs_tol=1e-15)
            )
            if not metadata_match or not samples_share_prefix or not same_summary:
                raise RuntimeError(
                    f"Independent baselines disagree at B={row.budget} across "
                    "split-schedule result files"
                )
            if row.n > reference.n:
                independent_by_budget[row.budget] = row
    return independent_by_budget


def _uniform_rows_at_budget(
    rows: list[ResultRow], budget: int
) -> dict[float, ResultRow]:
    return {
        row.uniform_c: row for row in rows if row.budget == budget and row.is_uniform_c
    }


def _normal_reduction(
    independent: ResultRow, splitting: ResultRow
) -> tuple[float, float, float]:
    if independent.mean <= 0.0:
        raise RuntimeError("Independent-path mean metric must be positive")
    observed = 100.0 * (1.0 - splitting.mean / independent.mean)
    if independent.n < 2 or splitting.n < 2:
        return observed, math.nan, math.nan
    variance = 100.0**2 * (
        splitting.mean**2 * independent.std**2 / (independent.mean**4 * independent.n)
        + splitting.std**2 / (independent.mean**2 * splitting.n)
    )
    if not math.isfinite(variance) or variance < 0.0:
        raise RuntimeError("Invalid delta-method variance for metric reduction")
    half = Z_975 * math.sqrt(variance)
    return observed, observed - half, observed + half


def _compute_reductions(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
) -> dict[tuple[str, tuple[float, ...], int, Any], tuple[float, float, float]]:
    """Reductions against the shared fixed_N row, keyed by ``"learned"``/``("uniform", c)``."""
    reductions: dict[
        tuple[str, tuple[float, ...], int, Any], tuple[float, float, float]
    ] = {}
    for model in MODELS:
        if model["directory"] not in all_rows:
            continue
        for schedule, _ in SCHEDULES:
            if schedule not in all_rows[model["directory"]]:
                continue
            rows = all_rows[model["directory"]][schedule]
            for budget in sorted({row.budget for row in rows}):
                independent = _row_at_budget(rows, budget, "fixed_N")
                if independent is None:
                    continue
                key = (model["directory"], schedule, budget)
                splitting = _row_at_budget(rows, budget, "adaptive")
                if splitting is not None:
                    reductions[(*key, "learned")] = _normal_reduction(
                        independent, splitting
                    )
                for c, row in _uniform_rows_at_budget(rows, budget).items():
                    reductions[(*key, ("uniform", c))] = _normal_reduction(
                        independent, row
                    )
    return reductions


def _budget_tick(value: float, _: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    return f"{value / 1_000:g}k"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _four_model_style() -> None:
    """Use compact, publication-readable type for the standardized 2x2 figures."""
    _style()
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )


def _four_model_title(
    model: dict[str, Any], *, include_metric: bool = True
) -> str:
    """Return a concise panel title, optionally retaining the metric identifier."""
    title = {
        "simple_ou": "OU",
        "coupled_double_well_langevin": "Langevin",
        "edm_default": "EDM-GMM",
        "ddpm_cifar10_hf_mmd": "CIFAR-10",
    }[model["directory"]]
    return f"{title} ({model['metric'].upper()})" if include_metric else title


def _label_visible_grid(axes: np.ndarray, *, xlabel: str, ylabel: str) -> None:
    for column in range(axes.shape[1]):
        visible = [
            axes[row, column]
            for row in range(axes.shape[0])
            if axes[row, column].get_visible()
        ]
        if visible:
            visible[-1].set_xlabel(xlabel)
    for row in range(axes.shape[0]):
        visible = [
            axes[row, column]
            for column in range(axes.shape[1])
            if axes[row, column].get_visible()
        ]
        if visible:
            visible[0].set_ylabel(ylabel)


def _hide_repeated_x_ticklabels(axes: np.ndarray) -> None:
    """Show x tick labels only on the lowest visible panel in each column."""
    for column in range(axes.shape[1]):
        visible = [
            axes[row, column]
            for row in range(axes.shape[0])
            if axes[row, column].get_visible()
        ]
        for ax in visible[:-1]:
            ax.tick_params(axis="x", which="both", labelbottom=False)


def _set_shared_axis_labels(
    fig: plt.Figure, *, xlabel: str, ylabel: str
) -> None:
    """Place shared labels close to the axes without double-counting their margins."""
    x_label = fig.supxlabel(xlabel, x=0.52, y=0.025)
    y_label = fig.supylabel(ylabel, x=0.015, y=0.49)
    x_label.set_in_layout(False)
    y_label.set_in_layout(False)


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    tight: bool = True,
    dpi: float | None = None,
) -> None:
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=dpi if dpi is not None else round(240 * OUTPUT_SCALE),
        bbox_inches="tight" if tight else None,
        metadata={"Software": "paper_plots.py"},
    )
    plt.close(fig)


def _finish_four_model_figure(
    fig: plt.Figure,
    legend_handles: list[Any],
    legend_labels: list[str],
    *,
    legend_ncol: int | None = None,
    layout: dict[str, Any] | None = None,
) -> None:
    """Apply the common legend, spacing, and exact-size export layout."""
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol if legend_ncol is not None else len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.tight_layout(**(layout if layout is not None else FOUR_MODEL_LAYOUT))


def _plot_splitting_diagram(output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
        }
    )
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.07, top=0.97)
    path_color = "#2b2b2b"
    guide_color = "#c7c7c7"
    text_color = "#111111"
    label_fontsize = 26
    x = [0.0, 1.6, 3.2, 4.8, 6.4, 8.0]
    levels = [
        [(x[0], 0.0)],
        [(x[1], 0.70), (x[1], -0.80)],
        [
            (x[2], 2.20),
            (x[2], 1.45),
            (x[2], 0.85),
            (x[2], -0.45),
            (x[2], -1.25),
            (x[2], -1.95),
        ],
        [
            (x[3], 2.45),
            (x[3], 1.15),
            (x[3], 1.05),
            (x[3], -0.15),
            (x[3], -1.55),
            (x[3], -2.25),
        ],
        [
            (x[4], 3.20),
            (x[4], 2.70),
            (x[4], 1.95),
            (x[4], 1.35),
            (x[4], 1.55),
            (x[4], 0.70),
            (x[4], 0.25),
            (x[4], -0.40),
            (x[4], -1.05),
            (x[4], -1.75),
            (x[4], -1.95),
            (x[4], -2.60),
        ],
        [
            (x[5], 3.00),
            (x[5], 3.35),
            (x[5], 1.75),
            (x[5], 1.65),
            (x[5], 1.15),
            (x[5], 0.95),
            (x[5], 0.50),
            (x[5], -0.10),
            (x[5], -0.85),
            (x[5], -1.55),
            (x[5], -2.15),
            (x[5], -2.80),
        ],
    ]

    for x_value in x:
        ax.plot(
            [x_value, x_value],
            [-3.8, 4.3],
            linestyle=":",
            linewidth=1.0,
            color=guide_color,
            zorder=0,
        )

    def connect(
        parent: tuple[float, float], child: tuple[float, float], width: float
    ) -> None:
        ax.plot(
            [parent[0], child[0]],
            [parent[1], child[1]],
            linewidth=width,
            color=path_color,
            zorder=2,
        )

    for child in levels[1]:
        connect(levels[0][0], child, 2.3)
    for parent_index, parent in enumerate(levels[1]):
        for child in levels[2][3 * parent_index : 3 * (parent_index + 1)]:
            connect(parent, child, 2.0)
    for parent, child in zip(levels[2], levels[3]):
        connect(parent, child, 1.9)
    for parent_index, parent in enumerate(levels[3]):
        for child in levels[4][2 * parent_index : 2 * (parent_index + 1)]:
            connect(parent, child, 1.85)
    for parent, child in zip(levels[4], levels[5]):
        connect(parent, child, 1.75)

    for level, size in zip(levels, [42, 40, 38, 38, 34, 32]):
        ax.scatter(
            [point[0] for point in level],
            [point[1] for point in level],
            s=size,
            color=path_color,
            zorder=3,
        )

    for index, factor in enumerate([2, 3, 1, 2], start=1):
        ax.text(
            x[index],
            3.55,
            rf"$N_{index}={factor}$",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color=text_color,
        )
    time_labels = [r"$t_0=0$", r"$t_1$", r"$t_2$", r"$t_3$", r"$t_4$", r"$t_5=T$"]
    for x_value, label in zip(x, time_labels):
        ax.text(
            x_value,
            -4.00,
            label,
            ha="center",
            va="top",
            fontsize=label_fontsize,
            color=text_color,
        )
    for index, count in enumerate([1, 2, 6, 6, 12]):
        ax.text(
            x[index],
            -4.75,
            rf"$R_{index}={count}$",
            ha="center",
            va="top",
            fontsize=label_fontsize,
            color=text_color,
        )
    ax.text(
        4.0,
        -5.55,
        r"$R_i=\prod_{j=1}^{i}N_j$",
        ha="center",
        va="top",
        fontsize=28,
        color=text_color,
    )
    ax.set_xlim(-0.4, 8.4)
    ax.set_ylim(-6.40, 4.30)
    ax.axis("off")
    fig.savefig(
        output_dir / "splitting_diagram.png",
        dpi=128,
        metadata={"Software": "paper_plots.py"},
    )
    plt.close(fig)


def _reduction_legend_handles(
    schedules: set[tuple[float, ...]],
    *,
    include_learned: bool,
    uniform_c_values: set[float],
) -> list[Line2D]:
    """Colour encodes the split count; dash pattern encodes the allocation."""
    handles = [
        Line2D([], [], color=SCHEDULE_COLORS[schedule], linewidth=1.2, label=label)
        for schedule, label in SCHEDULES
        if schedule in schedules
    ]
    if include_learned:
        handles.append(Line2D([], [], color="#555555", linewidth=1.2, label="Learned"))
    handles.extend(
        Line2D(
            [],
            [],
            color="#555555",
            linewidth=1.0,
            linestyle=UNIFORM_C_LINESTYLES[c],
            label=rf"$c={c:g}$",
        )
        for c in UNIFORM_C_VALUES
        if c in uniform_c_values
    )
    return handles


def _plot_reductions(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    reductions: dict[
        tuple[str, tuple[float, ...], int, Any], tuple[float, float, float]
    ],
    output_dir: Path,
) -> bool:
    _four_model_style()
    fig, axes = plt.subplots(2, 2, figsize=FOUR_MODEL_FIGSIZE, sharey=True)
    interval_extrema: list[float] = []
    plotted_schedules: set[tuple[float, ...]] = set()
    plotted_uniform_c: set[float] = set()
    plotted_learned = False

    for ax, model in zip(axes.flat, MODELS):
        if model["directory"] not in all_rows:
            ax.set_visible(False)
            continue
        rows_by_schedule = all_rows[model["directory"]]
        panel_has_data = False
        for schedule, label in SCHEDULES:
            if schedule not in rows_by_schedule:
                continue
            candidate_budgets = sorted(
                {row.budget for row in rows_by_schedule[schedule]}
            )
            learned = [
                (
                    budget,
                    reductions.get((model["directory"], schedule, budget, "learned")),
                )
                for budget in candidate_budgets
            ]
            learned = [
                (budget, value) for budget, value in learned if value is not None
            ]
            if learned:
                budgets = [budget for budget, _ in learned]
                values = np.asarray([value for _, value in learned])
                observed, lower, upper = values.T
                interval_extrema.extend(observed.tolist())
                finite = np.isfinite(lower) & np.isfinite(upper)
                interval_extrema.extend(lower[finite].tolist())
                interval_extrema.extend(upper[finite].tolist())
                if finite.any():
                    ax.fill_between(
                        budgets,
                        lower,
                        upper,
                        where=finite,
                        color=SCHEDULE_COLORS[schedule],
                        alpha=INTERVAL_ALPHA,
                        linewidth=0,
                    )
                ax.plot(
                    budgets,
                    observed,
                    color=SCHEDULE_COLORS[schedule],
                    linestyle="-",
                    linewidth=1.2,
                    label=label,
                    zorder=3,
                )
                panel_has_data = True
                plotted_schedules.add(schedule)
                plotted_learned = True
            for c in _expected_uniform_c(model, schedule):
                uniform = [
                    (
                        budget,
                        reductions.get(
                            (model["directory"], schedule, budget, ("uniform", c))
                        ),
                    )
                    for budget in candidate_budgets
                ]
                uniform = [
                    (budget, value) for budget, value in uniform if value is not None
                ]
                if not uniform:
                    continue
                uniform_budgets = [budget for budget, _ in uniform]
                uniform_observed = [value[0] for _, value in uniform]
                interval_extrema.extend(uniform_observed)
                # No band: nine curves per panel with bands would be unreadable.
                ax.plot(
                    uniform_budgets,
                    uniform_observed,
                    color=SCHEDULE_COLORS[schedule],
                    linestyle=UNIFORM_C_LINESTYLES[c],
                    linewidth=1.0,
                    alpha=0.6,
                    zorder=2,
                )
                panel_has_data = True
                plotted_schedules.add(schedule)
                plotted_uniform_c.add(c)
        if not panel_has_data:
            ax.set_visible(False)
            continue
        ax.axhline(0.0, color="#555555", linewidth=REFERENCE_LINEWIDTH, linestyle=":")
        ax.set_xscale("log")
        ax.set_title(_four_model_title(model))
        ax.grid(axis="y", color="#D8D8D8", linewidth=GRID_LINEWIDTH)
        ax.xaxis.set_major_formatter(FuncFormatter(_budget_tick))
        ax.tick_params(axis="x", which="minor", bottom=False)

    if not interval_extrema:
        plt.close(fig)
        warnings.warn(
            "No paired method and independent results; metric-gain figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    low = min(interval_extrema)
    high = max(interval_extrema)
    pad = 0.08 * max(high - low, 1.0)
    for ax in axes.flat:
        ax.set_ylim(math.floor(low - pad), math.ceil(high + pad))
    visible_row_count = sum(
        any(ax.get_visible() for ax in axes[row, :]) for row in range(axes.shape[0])
    )
    if visible_row_count > 1:
        _label_visible_grid(axes, xlabel="", ylabel="")
        _hide_repeated_x_ticklabels(axes)
        _set_shared_axis_labels(
            fig,
            xlabel="Budget $B$",
            ylabel="Mean Metric Reduction (%)",
        )
    else:
        _label_visible_grid(
            axes, xlabel="Budget $B$", ylabel="Mean Metric Reduction (%)"
        )
    legend_handles = _reduction_legend_handles(
        plotted_schedules,
        include_learned=plotted_learned,
        uniform_c_values=plotted_uniform_c,
    )
    legend_labels = [handle.get_label() for handle in legend_handles]
    _finish_four_model_figure(
        fig,
        legend_handles,
        legend_labels,
        legend_ncol=len(legend_labels),
    )
    _save_figure(
        fig,
        output_dir,
        "experiment_metric_gain",
        tight=False,
        dpi=FOUR_MODEL_DPI,
    )
    return True


def _plot_absolute_metrics(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    output_dir: Path,
) -> bool:
    _four_model_style()
    fig, axes = plt.subplots(2, 2, figsize=FOUR_MODEL_FIGSIZE)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    plotted_any = False

    for ax, model in zip(axes.flat, MODELS):
        if model["directory"] not in all_rows:
            ax.set_visible(False)
            continue
        rows_by_schedule = all_rows[model["directory"]]
        panel_has_data = False
        independent_by_budget = _validated_independent_rows(rows_by_schedule)
        baseline_budgets = sorted(independent_by_budget)
        baseline_rows = [independent_by_budget[budget] for budget in baseline_budgets]
        baseline_means = np.asarray([row.mean for row in baseline_rows])
        baseline_intervals = np.asarray([row.mean_ci() for row in baseline_rows])

        if np.any(baseline_means <= 0.0):
            raise RuntimeError("Absolute metric plot requires positive baseline values")
        if baseline_rows:
            positive_interval = baseline_intervals[:, 0] > 0.0
            if positive_interval.any():
                ax.fill_between(
                    baseline_budgets,
                    baseline_intervals[:, 0],
                    baseline_intervals[:, 1],
                    where=positive_interval,
                    color="#333333",
                    alpha=INTERVAL_ALPHA,
                    linewidth=0,
                )
            ax.plot(
                baseline_budgets,
                baseline_means,
                color="#333333",
                linestyle="--",
                linewidth=DATA_LINEWIDTH,
                label="Independent MC",
            )
            panel_has_data = True

        for schedule, label in SCHEDULES:
            if schedule not in rows_by_schedule:
                continue
            splitting_rows = sorted(
                (row for row in rows_by_schedule[schedule] if row.is_adaptive),
                key=lambda row: row.budget,
            )
            if not splitting_rows:
                continue
            budgets = [row.budget for row in splitting_rows]
            means = np.asarray([row.mean for row in splitting_rows])
            intervals = np.asarray([row.mean_ci() for row in splitting_rows])
            if np.any(means <= 0.0):
                raise RuntimeError("Absolute metric plot requires positive values")
            positive_interval = intervals[:, 0] > 0.0
            if positive_interval.any():
                ax.fill_between(
                    budgets,
                    intervals[:, 0],
                    intervals[:, 1],
                    where=positive_interval,
                    color=SCHEDULE_COLORS[schedule],
                    alpha=INTERVAL_ALPHA,
                    linewidth=0,
                )
            ax.plot(
                budgets,
                means,
                color=SCHEDULE_COLORS[schedule],
                linestyle="-",
                linewidth=DATA_LINEWIDTH,
                label=label,
            )
            panel_has_data = True

        if not panel_has_data:
            ax.set_visible(False)
            continue
        plotted_any = True

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(_four_model_title(model))
        ax.grid(axis="y", which="both", color="#D8D8D8", linewidth=GRID_LINEWIDTH)
        ax.xaxis.set_major_formatter(FuncFormatter(_budget_tick))
        ax.tick_params(axis="x", which="minor", bottom=False)
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

    if not plotted_any:
        plt.close(fig)
        warnings.warn(
            "No metric results; absolute-metric figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    _label_visible_grid(axes, xlabel="", ylabel="")
    _hide_repeated_x_ticklabels(axes)
    _set_shared_axis_labels(
        fig,
        xlabel="Budget $B$",
        ylabel="Mean Error Metric",
    )
    _finish_four_model_figure(
        fig,
        legend_handles,
        legend_labels,
        layout={**FOUR_MODEL_LAYOUT, "w_pad": 1.8},
    )
    _save_figure(
        fig,
        output_dir,
        "experiment_metric_absolute",
        tight=False,
        dpi=FOUR_MODEL_DPI,
    )
    return True


def _allocation_curve(
    schedule: tuple[float, ...], cumulative: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return post-step coordinates and pointwise confidence limits for R(t)."""
    split_times = 1.0 - np.asarray(schedule, dtype=float)
    if np.any(np.diff(split_times) <= 0.0):
        raise RuntimeError("Split times must be strictly increasing")

    mean_at_splits = cumulative.mean(axis=0)
    if cumulative.shape[0] > 1:
        standard_error = cumulative.std(axis=0, ddof=1) / math.sqrt(cumulative.shape[0])
        lower_at_splits = mean_at_splits - Z_975 * standard_error
        upper_at_splits = mean_at_splits + Z_975 * standard_error
    else:
        lower_at_splits = mean_at_splits
        upper_at_splits = mean_at_splits

    times = np.concatenate(([0.0], split_times, [1.0]))

    def extend(values: np.ndarray, initial: float) -> np.ndarray:
        return np.concatenate(([initial], values, [values[-1]]))

    return (
        times,
        extend(mean_at_splits, 1.0),
        extend(lower_at_splits, 1.0),
        extend(upper_at_splits, 1.0),
    )


def _plot_ou_oracle_allocations(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]], output_dir: Path
) -> bool:
    """Compare finite-query minimax and learned OU allocations."""
    rows_by_schedule = all_rows["simple_ou"]
    oracle_rows_by_schedule = all_rows["ou_oracle"]
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), sharex=True, sharey=True)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    plotted_any = False

    for ax, (schedule, schedule_label) in zip(axes, SCHEDULES):
        rows = rows_by_schedule.get(schedule, [])
        oracle_rows = oracle_rows_by_schedule.get(schedule, [])
        learned_budgets = {row.budget for row in rows if row.is_adaptive}
        oracle_budgets = {row.budget for row in oracle_rows if row.is_oracle}
        common_budgets = learned_budgets & oracle_budgets
        if not common_budgets:
            ax.set_visible(False)
            continue
        display_budget = max(common_budgets)
        splitting = _row_at_budget(rows, display_budget, "adaptive")
        oracle_row = _row_at_budget(oracle_rows, display_budget, "ou_oracle")
        assert splitting is not None and oracle_row is not None
        if splitting.split_factors is None:
            raise RuntimeError("Missing OU split-factor samples")
        cumulative = np.cumprod(splitting.split_factors, axis=1)
        times, learned_mean, _, _ = _allocation_curve(schedule, cumulative)
        factors = np.asarray(
            [float(value) for value in oracle_row.raw["N_i"].split(",")],
            dtype=float,
        )
        oracle = np.concatenate(([1.0], np.cumprod(factors)))
        oracle_curve = np.concatenate((oracle, [oracle[-1]]))

        ax.step(
            times,
            oracle_curve,
            where="post",
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label="Finite-query minimax",
            zorder=3,
        )
        ax.step(
            times,
            learned_mean,
            where="post",
            color=SCHEDULE_COLORS[schedule],
            linewidth=1.4,
            label="Learned mean",
            zorder=4,
        )
        ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_title(schedule_label)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.95, 3.5)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55)
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        warnings.warn(
            "No paired learned and OU-oracle allocations; oracle figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel("Elapsed fraction of trajectory")
    next(ax for ax in axes if ax.get_visible()).set_ylabel(
        r"Cumulative allocation $R_i$"
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.0)
    _save_figure(fig, output_dir, "ou_oracle_allocations", dpi=OU_ORACLE_DPI)
    return True


def _plot_allocations(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]], output_dir: Path
) -> bool:
    _four_model_style()
    fig, axes = plt.subplots(2, 2, figsize=FOUR_MODEL_FIGSIZE)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    plotted_any = False

    for ax, model in zip(axes.flat, MODELS):
        if model["directory"] not in all_rows:
            ax.set_visible(False)
            continue
        curves: list[
            tuple[
                tuple[float, ...],
                str,
                np.ndarray,
                np.ndarray,
            ]
        ] = []
        for schedule, label in SCHEDULES:
            if schedule not in all_rows[model["directory"]]:
                continue
            rows = all_rows[model["directory"]][schedule]
            splitting_rows = [row for row in rows if row.is_adaptive]
            if not splitting_rows:
                continue
            splitting = max(splitting_rows, key=lambda row: row.budget)
            if splitting.split_factors is None:
                raise RuntimeError("Missing split-factor samples")
            expected_shape = (splitting.n, len(schedule))
            if splitting.split_factors.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected split-factor shape for {model['directory']}: "
                    f"{splitting.split_factors.shape}, expected {expected_shape}"
                )
            split_factors = splitting.split_factors
            cumulative = np.cumprod(split_factors, axis=1)
            if np.any(cumulative < 1.0 - 1e-10) or np.any(
                np.diff(cumulative, axis=1) < -1e-10
            ):
                raise RuntimeError(f"Invalid allocation for {model['directory']}")
            times, mean, _, _ = _allocation_curve(schedule, cumulative)
            curves.append((schedule, label, times, mean))

        if not curves:
            ax.set_visible(False)
            continue
        plotted_any = True

        # Dense schedules go down first; shorter schedules stay visible where
        # their horizontal segments overlap the denser curves.
        for schedule, label, times, mean in reversed(curves):
            ax.step(
                times,
                mean,
                where="post",
                color=SCHEDULE_COLORS[schedule],
                linestyle="-",
                linewidth=DATA_LINEWIDTH,
                label=label,
                zorder=3,
            )
        ax.axhline(1.0, color="#555555", linewidth=REFERENCE_LINEWIDTH, linestyle=":")
        ax.set_title(_four_model_title(model, include_metric=False))
        ax.set_xlim(0.0, 1.0)
        ax.set_yscale("log", base=2)
        ax.set_ylim(bottom=0.95)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.tick_params(axis="y", which="minor", left=False)
        ax.grid(axis="y", which="major", color="#D8D8D8", linewidth=GRID_LINEWIDTH)
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
    if not plotted_any:
        plt.close(fig)
        warnings.warn(
            "No learned allocations; allocation figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    _label_visible_grid(axes, xlabel="", ylabel="")
    _hide_repeated_x_ticklabels(axes)
    _set_shared_axis_labels(
        fig,
        xlabel="Elapsed fraction of trajectory",
        ylabel=r"Learned $R_i$",
    )
    label_order = {label: index for index, (_, label) in enumerate(SCHEDULES)}
    ordered_legend = sorted(
        zip(legend_handles, legend_labels),
        key=lambda item: label_order[item[1]],
    )
    legend_handles = [handle for handle, _ in ordered_legend]
    legend_labels = [label for _, label in ordered_legend]
    _finish_four_model_figure(
        fig,
        legend_handles,
        legend_labels,
        layout={**FOUR_MODEL_LAYOUT, "w_pad": 1.4},
    )
    _save_figure(
        fig,
        output_dir,
        "experiment_allocations_four_models",
        tight=False,
        dpi=FOUR_MODEL_DPI,
    )
    return True


def main() -> None:
    args = _parse_args()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _load_and_verify_rows(outputs_root, debug=args.debug)
    _plot_splitting_diagram(output_dir)
    if all_rows:
        reductions = _compute_reductions(all_rows)
        _plot_reductions(all_rows, reductions, output_dir)
        _plot_absolute_metrics(all_rows, output_dir)
        _plot_allocations(all_rows, output_dir)
        if "simple_ou" in all_rows and "ou_oracle" in all_rows:
            _plot_ou_oracle_allocations(all_rows, output_dir)
    else:
        warnings.warn(
            "No complete runners found; experiment figures were not generated.",
            RuntimeWarning,
            stacklevel=2,
        )

    publication_models = (*MODELS, OU_ORACLE_MODEL)
    completed_models = [
        model for model in publication_models if model["directory"] in all_rows
    ]

    def run_count_summary(model: dict[str, Any]) -> str:
        observed = sorted(
            {row.n for rows in all_rows[model["directory"]].values() for row in rows}
        )
        if len(observed) == 1:
            count = f"{observed[0]} runs"
        else:
            count = f"{observed[0]}-{observed[-1]} runs per result"
        return f"{model['title']}: {count}"

    counts = ", ".join(run_count_summary(model) for model in completed_models)
    validated = counts or "no experiment runners"
    print(f"Validated {validated}; wrote publication PNGs to {output_dir}")


if __name__ == "__main__":
    main()
