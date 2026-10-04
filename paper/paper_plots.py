#!/usr/bin/env python3
"""Validate completed experiment records and regenerate the paper assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

sys.path.insert(0, str(ROOT / "src"))

from runners.common_configs import UNIFORM_C_BY_SPLIT_COUNT, split_schedules

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

# A two-sided 90% interval leaves 5% in each tail, hence Phi^-1(0.95).
Z_TWO_SIDED_90 = 1.6448536269514722
FIGURE_DPI = 300

FOUR_MODEL_FIGSIZE = (8.4, 8.4 / 1.9)
# 2520x1326 output for the standardized 1.9:1 figures at 300 DPI.
DATA_LINEWIDTH = 1.5
LEARNED_LINEWIDTH = 1.25
LEARNED_C_LINEWIDTH = 1.15
UNIFORM_C_LINEWIDTH = 1.05
UNIFORM_C_ALPHA = 0.8
REFERENCE_LINEWIDTH = 0.8
GRID_LINEWIDTH = 0.55
INTERVAL_ALPHA = 0.14
# Margins in inches, so every figure gets the same legend and x-label room.
MARGIN_BOTTOM_IN = 0.31
MARGIN_TOP_IN = 0.37
LEGEND_TOP_IN = 0.056
XLABEL_BOTTOM_IN = 0.14


def _four_model_layout(height: float, *, h_pad: float, w_pad: float) -> dict[str, Any]:
    return {
        "rect": (0.03, MARGIN_BOTTOM_IN / height, 0.99, 1.0 - MARGIN_TOP_IN / height),
        "h_pad": h_pad,
        "w_pad": w_pad,
    }


GAIN_LAYOUT = _four_model_layout(FOUR_MODEL_FIGSIZE[1], h_pad=0.76, w_pad=0.89)
ALLOCATION_LAYOUT = _four_model_layout(FOUR_MODEL_FIGSIZE[1], h_pad=0.67, w_pad=1.65)
# Five panels: the 2x2 grid plus a centred third row at the same panel size.
ABSOLUTE_FIGSIZE = (FOUR_MODEL_FIGSIZE[0], 1.5 * FOUR_MODEL_FIGSIZE[1])
ABSOLUTE_LAYOUT = _four_model_layout(ABSOLUTE_FIGSIZE[1], h_pad=0.95, w_pad=1.8)

SINGLE_01 = (0.1,)
SINGLE_02 = (0.2,)
# Taken from the sweep configuration rather than restated: the schedules are
# rounded to two decimals, and half-to-even ties there do not agree with plain
# round(), so a hand-written copy silently stops matching the CSV stems.
NINE, NINETEEN, THIRTY_NINE = (tuple(s) for s in split_schedules())
SCHEDULES = (
    (NINE, "9 splits"),
    (NINETEEN, "19 splits"),
    (THIRTY_NINE, "39 splits"),
)
SCHEDULE_COLORS = {
    SINGLE_01: "#009E73",
    SINGLE_02: "#CC79A7",
    NINE: "#D55E00",
    NINETEEN: "#228833",
    THIRTY_NINE: "#0072B2",
}

# Colour encodes the split count, dash pattern the allocation.
UNIFORM_C_LINESTYLE = (0, (2, 1.4))
# Shares mode "adaptive" with the free allocation; lookups must name the optimizer.
LEARNED_C_OPTIMIZER = "learned_c"
LEARNED_C_LINESTYLE = (0, (5, 1.6))
LEARNED_C_LABEL = r"Learned $c$"

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
# num_queries and k_max are per-model (see MODELS); the rest are sweep-wide.
EXPECTED_QUERY_PARAMS = {
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
        "metric": "ks",
        "n_runs": 2_500,
        "num_queries": 1_024,
        "k_max": 64,
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
        "metric": "ks",
        "n_runs": 2_500,
        "num_queries": 1_024,
        "k_max": 64,
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
        "metric": "ks",
        "n_runs": 2_500,
        "num_queries": 1_024,
        "k_max": 64,
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
        "steps": (40, 46, 55, 63, 73, 87),
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
        "metric": "mmd_inception",
        "mmd_representations": {
            "mmd_inception": "float_target",
            "mmd_dino": "float_target",
        },
        "phase1": "mmd_sibling",
        "n_runs": 50,
        "num_queries": 4_096,
        "k_max": 512,
        "pilot_coefficient": 10.0,
        "pilot_exponent": 0.66,
        "budgets": (50_000, 100_000, 200_000, 500_000),
        "steps": (200, 252, 317, 430),
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
    {
        "directory": "ldm_ffhq_mmd",
        "runner": "ldm_ffhq",
        "config_name": "mmd",
        "title": "FFHQ LDM",
        "metric": "mmd_inception",
        "mmd_representations": {
            "mmd_inception": "float_target",
            "mmd_dino": "float_target",
        },
        "phase1": "mmd_sibling",
        "n_runs": 50,
        "num_queries": 4_096,
        "k_max": 1_024,
        "pilot_coefficient": 10.0,
        "pilot_exponent": 0.66,
        "budgets": (50_000, 100_000, 200_000, 500_000),
        "steps": (200, 252, 317, 430),
        "sampler": "ddpm",
        "reference_size": 20_000,
        "target": None,
        "reference_generation": {
            "method": "ldm_ddpm_samples",
            "sampler": "ddpm",
            "T": 1_000,
            "sampling_steps": 1_000,
            "timestep_spacing": "trailing",
            "seed": 0,
        },
    },
)
MODELS_BY_DIRECTORY = {model["directory"]: model for model in MODELS}
# The 2x2 figures carry FFHQ in place of CIFAR-10; the tables keep both.
PLOT_MODELS = tuple(
    MODELS_BY_DIRECTORY[directory]
    for directory in (
        "simple_ou",
        "coupled_double_well_langevin",
        "edm_default",
        "ldm_ffhq_mmd",
    )
)
# Only the absolute-metric figure has room for CIFAR-10, in a centred bottom row.
ABSOLUTE_MODELS = (*PLOT_MODELS, MODELS_BY_DIRECTORY["ddpm_cifar10_hf_mmd"])

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


TIMING_BUDGET = 500_000
# Per-run timings from the `timing` configs: one run at a time after a warm-up.
TIMING_MODELS = tuple(
    {
        **MODELS_BY_DIRECTORY[directory],
        "directory": directory.replace("_mmd", "_timing"),
        "title": f"{MODELS_BY_DIRECTORY[directory]['title']} timing",
        "config_name": "timing",
        "n_runs": 10,
        "budgets": (TIMING_BUDGET,),
        "steps": (
            dict(zip(MODELS_BY_DIRECTORY[directory]["budgets"], MODELS_BY_DIRECTORY[directory]["steps"]))[
                TIMING_BUDGET
            ],
        ),
        "timing": True,
    }
    for directory in ("ddpm_cifar10_hf_mmd", "ldm_ffhq_mmd")
)
TIMING_FIELDS = (
    "phase1_simulation_seconds",
    "query_seconds",
    "estimation_seconds",
    "optimization_seconds",
    "phase2_seconds",
    "total_seconds",
    "scoring_seconds",
)
# total_seconds is the sum of these; scoring is timed separately.
TIMING_STAGES = TIMING_FIELDS[:5]
PILOT_STAGES = TIMING_STAGES[:4]
INDEPENDENT_TIMING_FIELDS = frozenset({"phase2_seconds", "total_seconds", "scoring_seconds"})
TIMING_METHODS = (("Learned", "monotone"), (LEARNED_C_LABEL, LEARNED_C_OPTIMIZER))
TIMING_SEGMENTS = (
    ("Phase 1 pilot", ("phase1_simulation_seconds",), "#E69F00"),
    ("Allocation design", ("query_seconds", "estimation_seconds", "optimization_seconds"), "#CC79A7"),
    ("Phase 2", ("phase2_seconds",), "#56B4E9"),
)


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
    timings: dict[str, np.ndarray] | None = None

    @property
    def is_adaptive(self) -> bool:
        return self.mode == "adaptive"

    @property
    def is_free_allocation(self) -> bool:
        """Adaptive with L free monotone split factors (the default optimizer)."""
        return self.mode == "adaptive" and self.optimizer == "monotone"

    @property
    def is_learned_c(self) -> bool:
        """Adaptive restricted to one learned branching factor for every split."""
        return self.mode == "adaptive" and self.optimizer == LEARNED_C_OPTIMIZER

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
        half = Z_TWO_SIDED_90 * self.std / math.sqrt(self.n)
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
            "configured one-, nine-, nineteen-, or thirty-nine-split schedules"
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
    """Read the requested constant c from a constant-allocation row.

    `N_i` holds what the tree mixture realized, which brackets c rather than
    equalling it, so c is carried in its own column.
    """
    factors = [float(value) for value in raw["N_i"].split(",")]
    factor_stds = [float(value) for value in raw["N_i_std"].split(",")]
    if (
        len(factors) != len(schedule)
        or len(factor_stds) != len(schedule)
        or raw["optimizer"] != "uniform"
        or not raw.get("uniform_c")
        or int(raw["n0"]) < 1
    ):
        raise RuntimeError(f"Invalid uniform_c row in {csv_path}")
    return float(raw["uniform_c"])


def _check_uniform_c_bracket(row: ResultRow) -> None:
    """A uniform-c run must land inside the coherent-rounding bracket of c^i.

    The request c is relaxed; what runs is a mixture of exact integer trees whose
    cumulative profiles satisfy r/2 < R <= 2r level by level, so the realized mean
    profile inherits that bracket. The per-level factors do not -- they range over
    roughly 0.55c to 1.82c -- which is why the check is on the profile.
    """
    factors = np.asarray(row.split_factors, dtype=float)
    if np.any(factors < 1.0 - 1e-9):
        raise RuntimeError(
            f"uniform_c N_i below one for {row.csv_path.name}, B={row.budget}"
        )
    profile = np.concatenate(([1.0], np.cumprod(factors.mean(axis=0))))
    target = float(row.uniform_c) ** np.arange(profile.size)
    ratio = profile / target
    if np.any(ratio <= 0.5 - 1e-9) or np.any(ratio > 2.0 + 1e-9):
        raise RuntimeError(
            f"uniform_c profile outside the (c/2, 2c] rounding bracket for "
            f"{row.csv_path.name}, B={row.budget}: R_i / c^i in "
            f"[{ratio.min():.4f}, {ratio.max():.4f}]"
        )


def _resolved_pilot_budget(model: dict[str, Any], budget: int) -> int:
    value = model["pilot_coefficient"] * budget ** model["pilot_exponent"]
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest)
    return math.floor(value)


def _listed_csvs(
    model: dict[str, Any],
    experiment_dir: Path,
    manifest_path: Path,
    allow_partial_schedules: bool,
) -> list[Path]:
    expected_names = {
        f"compare_results_{_schedule_name(schedule)}.csv" for schedule, _ in SCHEDULES
    }
    if allow_partial_schedules:
        listed = [
            experiment_dir / name
            for name in sorted(expected_names)
            if (experiment_dir / name).is_file()
        ]
        if not listed:
            raise RuntimeError(f"No recognized split-schedule CSVs in {experiment_dir}")
        return listed
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("runner_name") != model["runner"]:
        raise RuntimeError(f"Unexpected runner in {manifest_path}")
    if manifest.get("config_name") != model["config_name"]:
        raise RuntimeError(f"Unexpected config in {manifest_path}")
    listed = manifest.get("csv_files", [])
    if (
        len(listed) != len(SCHEDULES)
        or {Path(value).name for value in listed} != expected_names
    ):
        raise RuntimeError(
            f"{manifest_path} must list all {len(SCHEDULES)} completed "
            "split-schedule CSVs"
        )
    return [experiment_dir / Path(value).name for value in listed]


def _accept_run_count(
    n: int,
    model: dict[str, Any],
    csv_path: Path,
    allow_partial_runs: bool,
    partial_run_counts: set[int],
) -> bool:
    """False for a row with no valid runs, which partial loads skip."""
    if allow_partial_runs and n == 0:
        return False
    if n != model["n_runs"]:
        if not allow_partial_runs or not 1 <= n < model["n_runs"]:
            raise RuntimeError(
                f"{csv_path}: expected {model['n_runs']} runs, found {n}"
            )
        partial_run_counts.add(n)
    return True


def _check_loaded_schedules(
    model: dict[str, Any],
    schedule_rows: dict[tuple[float, ...], list[ResultRow]],
    experiment_dir: Path,
    manifest_path: Path,
    *,
    allow_partial_schedules: bool,
    partial_run_counts: set[int],
    skipped_zero_run_rows: int,
) -> None:
    expected_schedules = {schedule for schedule, _ in SCHEDULES}
    if not allow_partial_schedules and set(schedule_rows) != expected_schedules:
        raise RuntimeError(f"Missing split-schedule data in {manifest_path}")
    missing_schedules = expected_schedules - set(schedule_rows)
    partial = f"Loading partial {model['title']} results"
    if missing_schedules:
        missing_labels = ", ".join(
            label for schedule, label in SCHEDULES if schedule in missing_schedules
        )
        warnings.warn(
            f"{partial}; missing schedules: {missing_labels}.",
            RuntimeWarning,
            stacklevel=3,
        )
    if partial_run_counts:
        low = min(partial_run_counts)
        high = max(partial_run_counts)
        observed = str(low) if low == high else f"between {low} and {high}"
        warnings.warn(
            f"{partial}: expected {model['n_runs']} runs, found {observed}.",
            RuntimeWarning,
            stacklevel=3,
        )
    if skipped_zero_run_rows:
        warnings.warn(
            f"{partial}: skipped {skipped_zero_run_rows} rows with no valid runs.",
            RuntimeWarning,
            stacklevel=3,
        )
    if not schedule_rows:
        raise RuntimeError(f"No usable partial results in {experiment_dir}")


def _load_manifest_rows(
    outputs_root: Path,
    models: tuple[dict[str, Any], ...] = MODELS,
    *,
    allow_partial_runs: bool = False,
    allow_partial_schedules: bool = False,
) -> dict[str, dict[tuple[float, ...], list[ResultRow]]]:
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]] = {}
    for model in models:
        partial_run_counts: set[int] = set()
        skipped_zero_run_rows = 0
        experiment_dir = outputs_root / model["directory"]
        manifest_path = experiment_dir / "compare_outputs.json"
        metric = model["metric"]
        schedule_rows: dict[tuple[float, ...], list[ResultRow]] = {}
        for csv_path in _listed_csvs(
            model, experiment_dir, manifest_path, allow_partial_schedules
        ):
            schedule = _schedule_from_filename(csv_path)
            rows: list[ResultRow] = []
            with csv_path.open(newline="") as handle:
                for raw in csv.DictReader(handle):
                    n = int(raw.get(f"n_valid_{metric}", "") or raw["n_valid_ks"])
                    if not _accept_run_count(
                        n, model, csv_path, allow_partial_runs, partial_run_counts
                    ):
                        skipped_zero_run_rows += 1
                        continue
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
                adaptive_rows = [row for row in group if row.is_free_allocation]
                learned_c_rows = [row for row in group if row.is_learned_c]
                independent_rows = [row for row in group if row.is_independent]
                unknown_adaptive = [
                    row
                    for row in group
                    if row.is_adaptive
                    and not (row.is_free_allocation or row.is_learned_c)
                ]
                if unknown_adaptive:
                    raise RuntimeError(
                        f"Unexpected adaptive optimizer at B={budget}: "
                        f"{sorted({row.optimizer for row in unknown_adaptive})}"
                    )
                if (
                    len(adaptive_rows) > 1
                    or len(learned_c_rows) > 1
                    or len(independent_rows) > 1
                ):
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
                expected_loss = None if model.get("phase1") == "mmd_sibling" else "mse"
                for splitting in adaptive_rows + learned_c_rows:
                    if (
                        splitting.pilot_rule != expected_rule
                        or splitting.pilot_budget != expected_pilot
                        or splitting.loss != expected_loss
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

        _check_loaded_schedules(
            model,
            schedule_rows,
            experiment_dir,
            manifest_path,
            allow_partial_schedules=allow_partial_schedules,
            partial_run_counts=partial_run_counts,
            skipped_zero_run_rows=skipped_zero_run_rows,
        )
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
    expected_steps = dict(zip(model["budgets"], model["steps"]))
    by_schedule: dict[tuple[float, ...], list[ResultRow]] = {}
    partial_run_counts: set[int] = set()
    skipped_zero_run_rows = 0
    for csv_path in _listed_csvs(
        model, experiment_dir, manifest_path, allow_partial_schedules
    ):
        schedule = _schedule_from_filename(csv_path)
        rows: list[ResultRow] = []
        with csv_path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                if raw["mode"] != "ou_oracle":
                    raise RuntimeError(f"Unexpected oracle mode in {csv_path}")
                n = int(raw["n_valid_ks"])
                if not _accept_run_count(
                    n, model, csv_path, allow_partial_runs, partial_run_counts
                ):
                    skipped_zero_run_rows += 1
                    continue
                factors = [float(value) for value in raw["N_i"].split(",")]
                factor_stds = [float(value) for value in raw["N_i_std"].split(",")]
                if (
                    len(factors) != len(schedule)
                    or any(value < 1.0 for value in factors)
                    or len(factor_stds) != len(schedule)
                    or any(value != 0.0 for value in factor_stds)
                    or raw["optimizer"] != "finite_query_minimax"
                    or float(raw["oracle_gap"]) > 1e-6
                    or int(raw["n0"]) < 1
                ):
                    raise RuntimeError(f"Invalid OU oracle row in {csv_path}")
                rows.append(
                    ResultRow(
                        model=model,
                        schedule=schedule,
                        mode=raw["mode"],
                        budget=int(raw["B"]),
                        pilot_budget=None,
                        pilot_rule=None,
                        sampler=raw["sampler"],
                        sampling_steps=int(raw["sampling_steps"]),
                        optimizer=raw["optimizer"],
                        loss=None,
                        reuse=None,
                        mean=float(raw["mean_ks"]),
                        std=float(raw["std_ks"]),
                        n=n,
                        csv_path=csv_path,
                        raw=raw,
                    )
                )

        budgets = {row.budget for row in rows}
        expected_budgets = set(model["budgets"])
        if not budgets.issubset(expected_budgets) or (
            not allow_partial_runs and budgets != expected_budgets
        ):
            raise RuntimeError(f"Unexpected oracle budgets in {csv_path}")
        for budget in sorted(budgets):
            group = [row for row in rows if row.budget == budget]
            if (
                len(group) > 1
                or (not allow_partial_runs and len(group) != 1)
                or any(
                    row.sampler != model["sampler"]
                    or row.sampling_steps != expected_steps[budget]
                    for row in group
                )
            ):
                raise RuntimeError(f"Invalid OU oracle methods at B={budget}")
        if rows:
            by_schedule[schedule] = rows

    _check_loaded_schedules(
        model,
        by_schedule,
        experiment_dir,
        manifest_path,
        allow_partial_schedules=allow_partial_schedules,
        partial_run_counts=partial_run_counts,
        skipped_zero_run_rows=skipped_zero_run_rows,
    )
    return by_schedule


def _cifar_target_is_expected(target: dict[str, Any]) -> bool:
    return (
        target.get("kind") == "hf_ddpm_cifar10_model_samples"
        and target.get("model_id") == "google/ddpm-cifar10-32"
        and target.get("image_shape") == [3, 32, 32]
        and target.get("sample_dim") == 3_072
        and target.get("postprocess") == "clamp_0_1_flat_v1"
    )


def _ffhq_target_is_expected(target: dict[str, Any]) -> bool:
    return (
        target.get("kind") == "hf_ldm_ffhq_model_samples"
        and target.get("model_id") == "asparius/ldm-ffhq-256"
        and target.get("commit_hash") == "5f206d37fa91ccbd1a389006cbecdd40798c0c2d"
        and target.get("latent_shape") == [4, 32, 32]
        and target.get("pixel_shape") == [3, 256, 256]
        and target.get("sample_dim") == 4_096
        and target.get("latent_scale_factor") == 0.13025
        and target.get("postprocess") == "kl_latent_scaled_flat_v1"
        and target.get("to_pixels") == "kl_decode_scaled_clamp_0_1_chw_flat_v1"
    )


IMAGE_TARGET_CHECKS = {
    "ddpm_cifar10_hf": _cifar_target_is_expected,
    "ldm_ffhq": _ffhq_target_is_expected,
}


def _metric_config_is_expected(model: dict[str, Any], config: dict[str, Any]) -> bool:
    metric_config = config.get("metric_config", {})
    if model["metric"] == "ks":
        return metric_config == {"metrics": ["ks"]}
    representations = model["mmd_representations"]
    if metric_config.get("metrics") != list(representations):
        return False
    return all(
        _mmd_config_is_expected(metric_config.get(metric, {}), representation)
        for metric, representation in representations.items()
    )


def _mmd_config_is_expected(mmd: dict[str, Any], representation: str) -> bool:
    return (
        mmd.get("kind") == "target_space_random_fourier_mmd"
        and mmd.get("version") == 1
        and mmd.get("estimator") == "biased_empirical_root"
        and mmd.get("kernel") == "equal_weight_multiscale_gaussian"
        and mmd.get("standardization") == "reference_coordinate_zscore_v1"
        and mmd.get("representation") == representation
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
    image_check = IMAGE_TARGET_CHECKS.get(model["runner"])
    if image_check is not None:
        targets_match = image_check(target) and image_check(
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
    if row.model.get("timing") and (key.get("batching") or {}).get("n_parallel") != 1:
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
        and (
            (key.get("phase1") or {}).get("estimator") == "mmd_sibling"
            if row.model.get("phase1") == "mmd_sibling"
            else key.get("query_params")
            == {
                "num_queries": row.model["num_queries"],
                "k_max": row.model["k_max"],
                **EXPECTED_QUERY_PARAMS,
            }
            and key.get("crossfit_q_mlp_params") == EXPECTED_MLP_PARAMS
        )
    )


def _read_valid_records(
    path: Path, metric: str, n_runs: int, timing: bool = False
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray] | None] | None:
    """The first n_runs valid records, or None if the cache holds fewer.

    With ``timing``, also each timing field per run (NaN where a record lacks it).
    """
    values: list[float] = []
    factors: list[list[float]] = []
    timings: dict[str, list[float]] = {field: [] for field in TIMING_FIELDS}
    factor_presence: bool | None = None
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            value = record["metrics"].get(metric)
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
            for field in TIMING_FIELDS:
                timings[field].append(float(record.get(field, math.nan)))
            if len(values) == n_runs:
                break
    if len(values) != n_runs:
        return None
    factor_array = np.asarray(factors, dtype=float) if factors else None
    timing_arrays = (
        {field: np.asarray(items, dtype=float) for field, items in timings.items()}
        if timing
        else None
    )
    return np.asarray(values, dtype=float), factor_array, timing_arrays


def _check_timings(row: ResultRow) -> None:
    """Stage times must be complete, add up to total_seconds and match the CSV."""
    timings = row.timings
    name = f"{row.csv_path.name}, B={row.budget}, {row.mode} {row.optimizer or ''}".rstrip()
    expected = set(TIMING_FIELDS) if row.is_adaptive else INDEPENDENT_TIMING_FIELDS
    for field in TIMING_FIELDS:
        values = timings[field]
        csv_mean, csv_std = row.raw[f"mean_{field}"], row.raw[f"std_{field}"]
        if field not in expected:
            if not np.all(np.isnan(values)) or csv_mean or csv_std:
                raise RuntimeError(f"Unexpected {field} for {name}")
            continue
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RuntimeError(f"Invalid {field} for {name}")
        mean_matches = math.isclose(
            float(values.mean()), float(csv_mean), rel_tol=1e-11, abs_tol=1e-13
        )
        std_matches = row.n == 1 or math.isclose(
            float(values.std(ddof=1)), float(csv_std), rel_tol=1e-11, abs_tol=1e-13
        )
        if not mean_matches or not std_matches:
            raise RuntimeError(f"{field} aggregate mismatch for {name}")
    if int(row.raw["n_valid_timing"]) != row.n:
        raise RuntimeError(f"n_valid_timing != {row.n} for {name}")
    stages = sum(np.nan_to_num(timings[field]) for field in TIMING_STAGES)
    if not np.allclose(stages, timings["total_seconds"], rtol=1e-9, atol=0.0):
        raise RuntimeError(f"Stage times do not add up to total_seconds for {name}")


def _attach_and_verify_caches(
    outputs_root: Path,
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    models: tuple[dict[str, Any], ...] = MODELS,
) -> None:
    record_cache: dict[
        tuple[Path, str, int],
        tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray] | None] | None,
    ] = {}
    for model in models:
        run_root = outputs_root / model["directory"] / "runs"
        configs: list[tuple[Path, dict[str, Any]]] = []
        for config_path in sorted(run_root.glob("*/config.json")):
            with config_path.open() as handle:
                configs.append((config_path, json.load(handle)))

        for rows in all_rows[model["directory"]].values():
            for row in rows:
                matches: list[
                    tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray] | None]
                ] = []
                for config_path, config in configs:
                    if not _config_matches(row, config):
                        continue
                    runs_path = config_path.with_name("runs.jsonl")
                    if not runs_path.is_file():
                        continue
                    cache_key = (runs_path, model["metric"], row.n)
                    if cache_key not in record_cache:
                        record_cache[cache_key] = _read_valid_records(
                            runs_path, model["metric"], row.n, bool(model.get("timing"))
                        )
                    if record_cache[cache_key] is None:
                        continue
                    samples, factors, timings = record_cache[cache_key]
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
                        matches.append((samples, factors, timings))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one validated cache for {row.csv_path.name}, "
                        f"B={row.budget}, B1={row.pilot_budget}; found {len(matches)}"
                    )
                row.samples, row.split_factors, row.timings = matches[0]
                if model.get("timing"):
                    _check_timings(row)
                if row.is_uniform_c:
                    if row.split_factors is None:
                        raise RuntimeError("uniform_c cache is missing N_i")
                    _check_uniform_c_bracket(row)
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
        all_rows = _load_manifest_rows(outputs_root, (*MODELS, *TIMING_MODELS))
        _attach_and_verify_caches(outputs_root, all_rows, (*MODELS, *TIMING_MODELS))
        all_rows["ou_oracle"] = _load_ou_oracle_rows(outputs_root)
        return all_rows

    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]] = {}
    for model in (*MODELS, *TIMING_MODELS):
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


def _row_at_budget(
    rows: list[ResultRow],
    budget: int,
    mode: str,
    optimizer: str | None = None,
    uniform_c: float | None = None,
) -> ResultRow | None:
    """One row for a (budget, mode); adaptive lookups must name the optimizer."""
    matches = [
        row
        for row in rows
        if row.budget == budget
        and row.mode == mode
        and (optimizer is None or row.optimizer == optimizer)
        and (uniform_c is None or row.uniform_c == uniform_c)
    ]
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
    return _delta_reduction(
        independent.mean,
        independent.std,
        independent.n,
        splitting.mean,
        splitting.std,
        splitting.n,
    )


def _delta_reduction(
    independent_mean: float,
    independent_std: float,
    independent_n: int,
    mean: float,
    std: float,
    n: int,
) -> tuple[float, float, float]:
    """Percent reduction of ``mean`` against the independent mean, with a 90% CI."""
    if independent_mean <= 0.0:
        raise RuntimeError("Independent-path mean metric must be positive")
    observed = 100.0 * (1.0 - mean / independent_mean)
    if independent_n < 2 or n < 2:
        return observed, math.nan, math.nan
    variance = 100.0**2 * (
        mean**2 * independent_std**2 / (independent_mean**4 * independent_n)
        + std**2 / (independent_mean**2 * n)
    )
    if not math.isfinite(variance) or variance < 0.0:
        raise RuntimeError("Invalid delta-method variance for metric reduction")
    half = Z_TWO_SIDED_90 * math.sqrt(variance)
    return observed, observed - half, observed + half


def _compute_reductions(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
) -> dict[tuple[str, tuple[float, ...], int, Any], tuple[float, float, float]]:
    """Reductions against the shared fixed_N row.

    Keyed by ``"learned"`` (free allocation), ``"learned_c"`` (one learned
    branching factor) or ``("uniform", c)`` (a fixed factor).
    """
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
                splitting = _row_at_budget(rows, budget, "adaptive", "monotone")
                if splitting is not None:
                    reductions[(*key, "learned")] = _normal_reduction(
                        independent, splitting
                    )
                learned_c = _row_at_budget(
                    rows, budget, "adaptive", LEARNED_C_OPTIMIZER
                )
                if learned_c is not None:
                    reductions[(*key, "learned_c")] = _normal_reduction(
                        independent, learned_c
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
            "font.size": 11.4,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.4,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 10.2,
            "legend.fontsize": 10.2,
        }
    )


def _four_model_title(model: dict[str, Any], *, include_metric: bool = True) -> str:
    """Return a concise panel title, optionally retaining the metric identifier."""
    title = {
        "simple_ou": "OU",
        "coupled_double_well_langevin": "Langevin",
        "edm_default": "EDM-GMM",
        "ddpm_cifar10_hf_mmd": "CIFAR-10",
        "ldm_ffhq_mmd": "FFHQ",
        "ddpm_cifar10_hf_timing": "CIFAR-10",
        "ldm_ffhq_timing": "FFHQ",
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
    """Show x tick labels only on the lowest visible panel in each column.

    Only for grids that share one x range across rows; budget panels autoscale
    per model and each label their own ticks.
    """
    for column in range(axes.shape[1]):
        visible = [
            axes[row, column]
            for row in range(axes.shape[0])
            if axes[row, column].get_visible()
        ]
        for ax in visible[:-1]:
            ax.tick_params(axis="x", which="both", labelbottom=False)


def _plotted_x_values(ax: plt.Axes) -> list[int]:
    return sorted({int(round(x)) for line in ax.get_lines() for x in line.get_xdata()})


def _set_budget_ticks(ax: plt.Axes, budgets: Iterable[int]) -> None:
    """Tick every budget that was run, since log decades skip 200k/2M/5M."""
    ticks = sorted(set(budgets))
    if not ticks:
        return
    ax.set_xticks(ticks)
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(FuncFormatter(_budget_tick))


def _set_shared_axis_labels(
    fig: plt.Figure, *, xlabel: str, ylabel: str
) -> None:
    """Place shared labels close to the axes without double-counting their margins."""
    x_label_y = XLABEL_BOTTOM_IN / fig.get_figheight()
    x_label = fig.supxlabel(xlabel, x=0.52, y=x_label_y)
    y_label = fig.supylabel(ylabel, x=0.015, y=0.49)
    x_label.set_in_layout(False)
    y_label.set_in_layout(False)


def _save_figure(
    fig: plt.Figure, output_dir: Path, stem: str, *, tight: bool = True
) -> None:
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight" if tight else None,
        metadata={"Software": "paper_plots.py"},
    )
    plt.close(fig)


def _finish_four_model_figure(
    fig: plt.Figure,
    legend_handles: list[Any],
    legend_labels: list[str],
    *,
    layout: dict[str, Any],
) -> None:
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0 - LEGEND_TOP_IN / fig.get_figheight()),
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.tight_layout(**layout)


def _plot_splitting_diagram(output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
        }
    )
    height = 6.0
    fig, ax = plt.subplots(figsize=(15, height))
    fig.subplots_adjust(left=0.035, right=0.965, bottom=1.15 / height, top=0.97)
    path_color = "#2b2b2b"
    guide_color = "#c7c7c7"
    text_color = "#111111"
    label_fontsize = 28.6
    guide_bottom = -3.8
    x = [0.0, 1.6, 3.2, 4.8, 6.4, 8.0]
    level_heights = [
        [0.0],
        [0.70, -0.80],
        [2.20, 1.45, 0.85, -0.45, -1.25, -1.95],
        [2.45, 1.15, 1.05, -0.15, -1.55, -2.25],
        [3.20, 2.70, 1.95, 1.35, 1.55, 0.70, 0.25, -0.40, -1.05, -1.75, -1.95, -2.60],
        [3.00, 3.35, 1.75, 1.65, 1.15, 0.95, 0.50, -0.10, -0.85, -1.55, -2.15, -2.80],
    ]
    levels = [
        [(x_value, y) for y in heights] for x_value, heights in zip(x, level_heights)
    ]

    for x_value in x:
        ax.plot(
            [x_value, x_value],
            [guide_bottom, 4.3],
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

    def label_below(x_value: float, drop: float, label: str, fontsize: float) -> None:
        ax.annotate(
            label,
            xy=(x_value, guide_bottom),
            xytext=(0.0, -drop),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=fontsize,
            color=text_color,
        )

    time_labels = [r"$t_0=0$", r"$t_1$", r"$t_2$", r"$t_3$", r"$t_4$", r"$t_5=T$"]
    for x_value, label in zip(x, time_labels):
        label_below(x_value, 9.0, label, label_fontsize)
    for index, count in enumerate([1, 2, 6, 6, 12]):
        label_below(x[index], 45.0, rf"$R_{index}={count}$", label_fontsize)
    ax.set_xlim(-0.4, 8.4)
    ax.set_ylim(guide_bottom, 4.30)
    ax.axis("off")
    _save_figure(fig, output_dir, "splitting_diagram")


def _reduction_legend_handles(
    schedules: set[tuple[float, ...]],
    *,
    include_learned: bool,
    include_uniform_c: bool,
    include_learned_c: bool = False,
) -> list[Line2D]:
    """Colour encodes the split count; dash pattern encodes the allocation."""
    handles = [
        Line2D(
            [],
            [],
            color=SCHEDULE_COLORS[schedule],
            linewidth=LEARNED_LINEWIDTH,
            label=label,
        )
        for schedule, label in SCHEDULES
        if schedule in schedules
    ]
    if include_learned:
        handles.append(
            Line2D(
                [], [], color="#555555", linewidth=LEARNED_LINEWIDTH, label="Learned"
            )
        )
    if include_learned_c:
        handles.append(
            Line2D(
                [],
                [],
                color="#555555",
                linewidth=LEARNED_C_LINEWIDTH,
                linestyle=LEARNED_C_LINESTYLE,
                label=LEARNED_C_LABEL,
            )
        )
    if include_uniform_c:
        handles.append(
            Line2D(
                [],
                [],
                color="#555555",
                linewidth=UNIFORM_C_LINEWIDTH,
                linestyle=UNIFORM_C_LINESTYLE,
                label=r"Uniform $c$",
            )
        )
    return handles


def _reduction_series(
    reductions: dict[tuple[str, tuple[float, ...], int, Any], tuple[float, float, float]],
    directory: str,
    schedule: tuple[float, ...],
    budgets: list[int],
    kind: Any,
) -> tuple[list[int], np.ndarray]:
    points = [(budget, reductions.get((directory, schedule, budget, kind))) for budget in budgets]
    points = [(budget, value) for budget, value in points if value is not None]
    return [budget for budget, _ in points], np.asarray([value for _, value in points])


def _plot_reductions(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    reductions: dict[
        tuple[str, tuple[float, ...], int, Any], tuple[float, float, float]
    ],
    output_dir: Path,
) -> None:
    _four_model_style()
    fig, axes = plt.subplots(2, 2, figsize=FOUR_MODEL_FIGSIZE, sharey=True)
    interval_extrema: list[float] = []
    plotted_schedules: set[tuple[float, ...]] = set()
    plotted_uniform_c = False
    plotted_learned = False
    plotted_learned_c = False

    for ax, model in zip(axes.flat, PLOT_MODELS):
        directory = model["directory"]
        if directory not in all_rows:
            ax.set_visible(False)
            continue
        rows_by_schedule = all_rows[directory]
        panel_has_data = False
        for schedule, label in SCHEDULES:
            if schedule not in rows_by_schedule:
                continue
            candidate_budgets = sorted(
                {row.budget for row in rows_by_schedule[schedule]}
            )
            color = SCHEDULE_COLORS[schedule]
            budgets, values = _reduction_series(
                reductions, directory, schedule, candidate_budgets, "learned"
            )
            if budgets:
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
                        color=color,
                        alpha=INTERVAL_ALPHA,
                        linewidth=0,
                    )
                ax.plot(
                    budgets,
                    observed,
                    color=color,
                    linestyle="-",
                    linewidth=LEARNED_LINEWIDTH,
                    label=label,
                    zorder=4,
                )
                panel_has_data = True
                plotted_schedules.add(schedule)
                plotted_learned = True
            budgets, values = _reduction_series(
                reductions, directory, schedule, candidate_budgets, "learned_c"
            )
            if budgets:
                observed = values[:, 0]
                interval_extrema.extend(observed.tolist())
                ax.plot(
                    budgets,
                    observed,
                    color=color,
                    linestyle=LEARNED_C_LINESTYLE,
                    linewidth=LEARNED_C_LINEWIDTH,
                    zorder=3,
                )
                panel_has_data = True
                plotted_schedules.add(schedule)
                plotted_learned_c = True
            # Only the strongest constant c is drawn, and only if it beats
            # independent paths on average; the tables keep every c.
            uniform_curves = []
            for c in _expected_uniform_c(model, schedule):
                budgets, values = _reduction_series(
                    reductions, directory, schedule, candidate_budgets, ("uniform", c)
                )
                if budgets:
                    observed = values[:, 0].tolist()
                    uniform_curves.append((float(np.mean(observed)), budgets, observed))
            if uniform_curves:
                mean_gain, budgets, observed = max(
                    uniform_curves, key=lambda curve: curve[0]
                )
                if mean_gain >= 0.0:
                    interval_extrema.extend(observed)
                    ax.plot(
                        budgets,
                        observed,
                        color=color,
                        linestyle=UNIFORM_C_LINESTYLE,
                        linewidth=UNIFORM_C_LINEWIDTH,
                        alpha=UNIFORM_C_ALPHA,
                        zorder=2,
                    )
                    panel_has_data = True
                    plotted_schedules.add(schedule)
                    plotted_uniform_c = True
        if not panel_has_data:
            ax.set_visible(False)
            continue
        panel_budgets = _plotted_x_values(ax)
        ax.axhline(0.0, color="#555555", linewidth=REFERENCE_LINEWIDTH, linestyle=":")
        ax.set_xscale("log")
        ax.set_title(_four_model_title(model))
        ax.grid(axis="y", color="#D8D8D8", linewidth=GRID_LINEWIDTH)
        _set_budget_ticks(ax, panel_budgets)
        ax.tick_params(axis="x", which="minor", bottom=False)

    if not interval_extrema:
        plt.close(fig)
        warnings.warn(
            "No paired method and independent results; metric-gain figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    low = min(interval_extrema)
    high = max(interval_extrema)
    pad = 0.08 * max(high - low, 1.0)
    for ax in axes.flat:
        ax.set_ylim(math.floor(low - pad), math.ceil(high + pad))
        # Shorter panels make the auto locator drop to two ticks.
        ax.yaxis.set_major_locator(MultipleLocator(5))
    visible_row_count = sum(
        any(ax.get_visible() for ax in axes[row, :]) for row in range(axes.shape[0])
    )
    if visible_row_count > 1:
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
        include_uniform_c=plotted_uniform_c,
        include_learned_c=plotted_learned_c,
    )
    legend_labels = [handle.get_label() for handle in legend_handles]
    _finish_four_model_figure(
        fig,
        legend_handles,
        legend_labels,
        layout=GAIN_LAYOUT,
    )
    _save_figure(fig, output_dir, "experiment_metric_gain", tight=False)


def _collect_legend(ax: plt.Axes, handles: list[Any], labels: list[str]) -> None:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)


def _plot_mean_with_interval(
    ax: plt.Axes,
    rows: list[ResultRow],
    *,
    color: str,
    linestyle: str,
    label: str,
    error: str,
) -> None:
    budgets = [row.budget for row in rows]
    means = np.asarray([row.mean for row in rows])
    intervals = np.asarray([row.mean_ci() for row in rows])
    if np.any(means <= 0.0):
        raise RuntimeError(error)
    positive_interval = intervals[:, 0] > 0.0
    if positive_interval.any():
        ax.fill_between(
            budgets,
            intervals[:, 0],
            intervals[:, 1],
            where=positive_interval,
            color=color,
            alpha=INTERVAL_ALPHA,
            linewidth=0,
        )
    ax.plot(
        budgets,
        means,
        color=color,
        linestyle=linestyle,
        linewidth=DATA_LINEWIDTH,
        label=label,
    )


def _plot_absolute_metrics(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    output_dir: Path,
) -> None:
    _four_model_style()
    fig = plt.figure(figsize=ABSOLUTE_FIGSIZE)
    grid = fig.add_gridspec(3, 4)
    axes = [
        fig.add_subplot(grid[row, 2 * column : 2 * column + 2])
        for row in range(2)
        for column in range(2)
    ]
    axes.append(fig.add_subplot(grid[2, 1:3]))
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    plotted_any = False

    for ax, model in zip(axes, ABSOLUTE_MODELS):
        if model["directory"] not in all_rows:
            ax.set_visible(False)
            continue
        rows_by_schedule = all_rows[model["directory"]]
        panel_has_data = False
        independent_by_budget = _validated_independent_rows(rows_by_schedule)
        if independent_by_budget:
            _plot_mean_with_interval(
                ax,
                [independent_by_budget[budget] for budget in sorted(independent_by_budget)],
                color="#333333",
                linestyle="--",
                label="Independent MC",
                error="Absolute metric plot requires positive baseline values",
            )
            panel_has_data = True

        for schedule, label in SCHEDULES:
            splitting_rows = sorted(
                (
                    row
                    for row in rows_by_schedule.get(schedule, [])
                    if row.is_free_allocation
                ),
                key=lambda row: row.budget,
            )
            if splitting_rows:
                _plot_mean_with_interval(
                    ax,
                    splitting_rows,
                    color=SCHEDULE_COLORS[schedule],
                    linestyle="-",
                    label=label,
                    error="Absolute metric plot requires positive values",
                )
                panel_has_data = True

        if not panel_has_data:
            ax.set_visible(False)
            continue
        plotted_any = True

        panel_budgets = _plotted_x_values(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(_four_model_title(model))
        ax.grid(axis="y", which="both", color="#D8D8D8", linewidth=GRID_LINEWIDTH)
        _set_budget_ticks(ax, panel_budgets)
        ax.tick_params(axis="x", which="minor", bottom=False)
        _collect_legend(ax, legend_handles, legend_labels)

    if not plotted_any:
        plt.close(fig)
        warnings.warn(
            "No metric results; absolute-metric figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    _set_shared_axis_labels(
        fig,
        xlabel="Budget $B$",
        ylabel="Mean Error Metric",
    )
    _finish_four_model_figure(
        fig,
        legend_handles,
        legend_labels,
        layout=ABSOLUTE_LAYOUT,
    )
    _save_figure(fig, output_dir, "experiment_metric_absolute", tight=False)


def _allocation_curve(
    schedule: tuple[float, ...], cumulative: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Post-step coordinates of the mean cumulative allocation R(t)."""
    split_times = 1.0 - np.asarray(schedule, dtype=float)
    if np.any(np.diff(split_times) <= 0.0):
        raise RuntimeError("Split times must be strictly increasing")
    mean_at_splits = cumulative.mean(axis=0)
    times = np.concatenate(([0.0], split_times, [1.0]))
    return times, np.concatenate(([1.0], mean_at_splits, [mean_at_splits[-1]]))


def _plot_ou_oracle_allocations(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]], output_dir: Path
) -> None:
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
        learned_budgets = {row.budget for row in rows if row.is_free_allocation}
        oracle_budgets = {row.budget for row in oracle_rows if row.is_oracle}
        common_budgets = learned_budgets & oracle_budgets
        if not common_budgets:
            ax.set_visible(False)
            continue
        display_budget = max(common_budgets)
        splitting = _row_at_budget(rows, display_budget, "adaptive", "monotone")
        oracle_row = _row_at_budget(oracle_rows, display_budget, "ou_oracle")
        assert splitting is not None and oracle_row is not None
        if splitting.split_factors is None:
            raise RuntimeError("Missing OU split-factor samples")
        cumulative = np.cumprod(splitting.split_factors, axis=1)
        times, learned_mean = _allocation_curve(schedule, cumulative)
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
        _collect_legend(ax, legend_handles, legend_labels)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        warnings.warn(
            "No paired learned and OU-oracle allocations; oracle figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
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
    _save_figure(fig, output_dir, "ou_oracle_allocations")


def _mean_allocation(
    model: dict[str, Any], schedule: tuple[float, ...], row: ResultRow
) -> tuple[np.ndarray, np.ndarray]:
    if row.split_factors is None:
        raise RuntimeError("Missing split-factor samples")
    expected_shape = (row.n, len(schedule))
    if row.split_factors.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected split-factor shape for {model['directory']}: "
            f"{row.split_factors.shape}, expected {expected_shape}"
        )
    cumulative = np.cumprod(row.split_factors, axis=1)
    if np.any(cumulative < 1.0 - 1e-10) or np.any(
        np.diff(cumulative, axis=1) < -1e-10
    ):
        raise RuntimeError(f"Invalid allocation for {model['directory']}")
    return _allocation_curve(schedule, cumulative)


def _plot_allocations(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]], output_dir: Path
) -> None:
    _four_model_style()
    fig, axes = plt.subplots(2, 2, figsize=FOUR_MODEL_FIGSIZE)
    plotted_schedules: set[tuple[float, ...]] = set()
    plotted_any = False

    for ax, model in zip(axes.flat, PLOT_MODELS):
        if model["directory"] not in all_rows:
            ax.set_visible(False)
            continue
        curves: list[tuple[tuple[float, ...], tuple[np.ndarray, np.ndarray]]] = []
        for schedule, _ in SCHEDULES:
            if schedule not in all_rows[model["directory"]]:
                continue
            rows = all_rows[model["directory"]][schedule]
            splitting_rows = [row for row in rows if row.is_free_allocation]
            if not splitting_rows:
                continue
            splitting = max(splitting_rows, key=lambda row: row.budget)
            curves.append((schedule, _mean_allocation(model, schedule, splitting)))

        if not curves:
            ax.set_visible(False)
            continue
        plotted_any = True

        # Dense schedules first, so shorter ones stay visible where they overlap.
        for schedule, (times, mean) in reversed(curves):
            ax.step(
                times,
                mean,
                where="post",
                color=SCHEDULE_COLORS[schedule],
                linestyle="-",
                linewidth=DATA_LINEWIDTH,
                zorder=3,
            )
            plotted_schedules.add(schedule)
        ax.axhline(1.0, color="#555555", linewidth=REFERENCE_LINEWIDTH, linestyle=":")
        ax.set_title(_four_model_title(model, include_metric=False))
        ax.set_xlim(0.0, 1.0)
        ax.set_yscale("log", base=2)
        ax.set_ylim(bottom=0.95)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.tick_params(axis="y", which="minor", left=False)
        ax.grid(axis="y", which="major", color="#D8D8D8", linewidth=GRID_LINEWIDTH)
    if not plotted_any:
        plt.close(fig)
        warnings.warn(
            "No learned allocations; allocation figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    _hide_repeated_x_ticklabels(axes)
    _set_shared_axis_labels(
        fig,
        xlabel="Elapsed fraction of trajectory",
        ylabel=r"Learned $R_i$",
    )
    legend_handles = _reduction_legend_handles(
        plotted_schedules,
        include_learned=False,
        include_uniform_c=False,
    )
    legend_labels = [handle.get_label() for handle in legend_handles]
    _finish_four_model_figure(
        fig,
        legend_handles,
        legend_labels,
        layout=ALLOCATION_LAYOUT,
    )
    _save_figure(fig, output_dir, "experiment_allocations_four_models", tight=False)


def _timing_rows(
    model: dict[str, Any],
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
) -> list[tuple[tuple[float, ...] | None, str | None, ResultRow]]:
    """Independent first, then each (schedule, optimizer) present, in figure order."""
    rows_by_schedule = all_rows.get(model["directory"], {})
    entries: list[tuple[tuple[float, ...] | None, str | None, ResultRow]] = []
    independent = _validated_independent_rows(rows_by_schedule).get(TIMING_BUDGET)
    if independent is not None:
        entries.append((None, None, independent))
    for schedule, _ in SCHEDULES:
        for _, optimizer in TIMING_METHODS:
            row = _row_at_budget(
                rows_by_schedule.get(schedule, []), TIMING_BUDGET, "adaptive", optimizer
            )
            if row is not None:
                entries.append((schedule, optimizer, row))
    return entries


def _timing_mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    if values.size < 2:
        return mean, math.nan, math.nan
    half = Z_TWO_SIDED_90 * float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean, mean - half, mean + half


def _timing_overhead(
    independent: ResultRow, method: ResultRow
) -> tuple[float, float, float]:
    """Percent increase in mean total time over independent paths, with a 90% CI."""
    base = independent.timings["total_seconds"]
    total = method.timings["total_seconds"]
    observed, lower, upper = _delta_reduction(
        float(base.mean()),
        float(base.std(ddof=1)) if base.size > 1 else 0.0,
        base.size,
        float(total.mean()),
        float(total.std(ddof=1)) if total.size > 1 else 0.0,
        total.size,
    )
    return -observed, -upper, -lower


def _timing_label(schedule: tuple[float, ...] | None, optimizer: str | None) -> str:
    if schedule is None:
        return "Independent"
    method = dict((key, label) for label, key in TIMING_METHODS)[optimizer]
    return f"{method}, {len(schedule)} splits"


def _plot_timings(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]], output_dir: Path
) -> None:
    models = [model for model in TIMING_MODELS if model["directory"] in all_rows]
    if not models:
        warnings.warn(
            "No timing results; timing figure was not generated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    entries = {model["directory"]: _timing_rows(model, all_rows) for model in models}
    order = [(None, None)] + [
        (schedule, optimizer)
        for schedule, _ in SCHEDULES
        for _, optimizer in TIMING_METHODS
    ]
    keys = [
        key
        for key in order
        if any(key == (schedule, optimizer) for rows in entries.values() for schedule, optimizer, _ in rows)
    ]
    _four_model_style()
    height = 1.0 + 0.36 * len(keys)
    fig, axes = plt.subplots(
        1, len(models), figsize=(FOUR_MODEL_FIGSIZE[0], height), sharey=True, squeeze=False
    )
    for ax, model in zip(axes[0], models):
        by_key = {(schedule, optimizer): row for schedule, optimizer, row in entries[model["directory"]]}
        independent = by_key.get((None, None))
        right = 0.0
        for y, key in enumerate(keys):
            row = by_key.get(key)
            if row is None:
                continue
            left = 0.0
            for _, fields, color in TIMING_SEGMENTS:
                width = sum(float(np.nan_to_num(row.timings[field]).mean()) for field in fields)
                ax.barh(y, width, left=left, height=0.64, color=color, edgecolor="white", linewidth=0.8)
                left += width
            mean, lower, upper = _timing_mean_ci(row.timings["total_seconds"])
            end = mean
            if math.isfinite(lower):
                ax.errorbar(
                    mean,
                    y,
                    xerr=[[mean - lower], [upper - mean]],
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=0.9,
                    capsize=2.5,
                )
                end = upper
            text = f"{mean:.1f} s"
            if independent is not None and row is not independent:
                text += f" ({_timing_overhead(independent, row)[0]:+.1f}%)"
            ax.annotate(
                text,
                (end, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=plt.rcParams["xtick.labelsize"],
                color="#333333",
            )
            right = max(right, end)
        ax.set_xlim(0.0, 1.35 * right)
        ax.set_title(f"{_four_model_title(model, include_metric=False)}, $B$ = {_budget_tick(TIMING_BUDGET, 0)}")
        ax.set_xlabel("Mean time per run (s)")
        ax.grid(axis="x", color="#D8D8D8", linewidth=GRID_LINEWIDTH)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0, 0].set_yticks(range(len(keys)), [_timing_label(*key) for key in keys])
    axes[0, 0].invert_yaxis()
    legend_handles = [Patch(color=color, label=label) for label, _, color in TIMING_SEGMENTS]
    legend_handles.append(
        Line2D([], [], color="#333333", linewidth=0.9, marker="|", markersize=6, label="90% CI of total")
    )
    _finish_four_model_figure(
        fig,
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        layout={"rect": (0.0, 0.0, 1.0, 1.0 - MARGIN_TOP_IN / height), "w_pad": 1.2},
    )
    _save_figure(fig, output_dir, "experiment_timing", tight=False)


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
        _plot_timings(all_rows, output_dir)
    else:
        warnings.warn(
            "No complete runners found; experiment figures were not generated.",
            RuntimeWarning,
            stacklevel=2,
        )

    publication_models = (*MODELS, OU_ORACLE_MODEL, *TIMING_MODELS)
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
