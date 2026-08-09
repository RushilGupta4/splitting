#!/usr/bin/env python3
"""Validate completed experiment records and regenerate the paper assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


Z_975 = 1.96
ALLOCATION_BUDGET = 1_000_000

FOUR = (0.8, 0.6, 0.4, 0.2)
NINE = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
SCHEDULES = (
    (FOUR, "Four split points"),
    (NINE, "Nine split points"),
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
    "compile": False,
}
EXPECTED_GRID_PARAMS = {"num_queries": 1_024, "tail_eps": 0.0001}

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
        "variance": [0.01, 0.01],
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
        "n_runs": 1_000,
        "pilot_coefficient": 5.0,
        "pilot_exponent": 0.66,
        "budgets": (100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000),
        "steps": (130, 160, 220, 280, 350, 475),
        "sampler": "euler",
        "subset_sizes": [8, 16, 32, 64],
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
        "n_runs": 1_000,
        "pilot_coefficient": 5.0,
        "pilot_exponent": 0.66,
        "budgets": (100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000),
        "steps": (130, 160, 220, 280, 350, 475),
        "sampler": "euler",
        "subset_sizes": [1, 2],
        "reference_size": 2_500_000,
        "target": LANGEVIN_TARGET,
        "reference_generation": {
            "method": "sde_terminal_samples",
            "sampler": "euler",
            "sampling_steps": 10_000,
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
        "n_runs": 1_000,
        "pilot_coefficient": 10.0,
        "pilot_exponent": 0.66,
        "budgets": (100_000, 200_000, 500_000, 1_000_000, 2_000_000),
        "steps": (40, 45, 55, 63, 73),
        "sampler": "edm_stochastic",
        "subset_sizes": [8, 16, 32, 64],
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
        "n_runs": 25,
        "pilot_coefficient": 10.0,
        "pilot_exponent": 0.66,
        "budgets": (50_000, 100_000, 250_000, 500_000, 1_000_000),
        "steps": (200, 252, 340, 430, 542),
        "sampler": "ddpm",
        "subset_sizes": [8, 16, 32, 64],
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

    @property
    def is_adaptive(self) -> bool:
        return self.mode == "adaptive"

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
        default=ROOT / "outputs_final",
        help="Root containing the completed experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "plots",
        help="Destination for generated PNG figures.",
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
            f"Unexpected split schedule in {path.name}; exactly four or nine "
            "split points are required"
        ) from exc


def _resolved_pilot_budget(model: dict[str, Any], budget: int) -> int:
    value = model["pilot_coefficient"] * budget ** model["pilot_exponent"]
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest)
    return math.floor(value)


def _load_manifest_rows(
    outputs_root: Path,
) -> dict[str, dict[tuple[float, ...], list[ResultRow]]]:
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]] = {}
    expected_csv_names = {
        f"compare_results_{_schedule_name(schedule)}.csv" for schedule, _ in SCHEDULES
    }

    for model in MODELS:
        experiment_dir = outputs_root / model["directory"]
        manifest_path = experiment_dir / "compare_outputs.json"
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        if manifest.get("runner_name") != model["runner"]:
            raise RuntimeError(f"Unexpected runner in {manifest_path}")
        if manifest.get("config_name") != model["config_name"]:
            raise RuntimeError(f"Unexpected config in {manifest_path}")

        listed = manifest.get("csv_files", [])
        if len(listed) != 2 or {Path(value).name for value in listed} != expected_csv_names:
            raise RuntimeError(
                f"{manifest_path} must list the completed four- and nine-split CSVs"
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
                    n = int(metric_n or raw["n_valid_runs"])
                    if n != model["n_runs"]:
                        raise RuntimeError(
                            f"{csv_path}: expected {model['n_runs']} runs, found {n}"
                        )
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
                    if row.mode not in {"adaptive", "fixed_N"}:
                        raise RuntimeError(f"Unexpected mode {row.mode!r} in {csv_path}")
                    rows.append(row)

            budgets = sorted({row.budget for row in rows})
            if tuple(budgets) != model["budgets"]:
                raise RuntimeError(f"Unexpected budget set in {csv_path}: {budgets}")
            if len(rows) != len(budgets) * 2:
                raise RuntimeError(f"Unexpected row count in {csv_path}")

            expected_steps = dict(zip(model["budgets"], model["steps"]))
            for budget in model["budgets"]:
                group = [row for row in rows if row.budget == budget]
                if sum(row.is_adaptive for row in group) != 1 or sum(
                    not row.is_adaptive for row in group
                ) != 1:
                    raise RuntimeError(
                        f"Expected one adaptive and one independent row at B={budget}"
                    )
                independent = next(row for row in group if not row.is_adaptive)
                splitting = next(row for row in group if row.is_adaptive)
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
                if (
                    splitting.pilot_rule != expected_rule
                    or splitting.pilot_budget != expected_pilot
                    or splitting.optimizer != "monotone_cvar95"
                    or splitting.loss != "mse"
                    or splitting.reuse is not True
                ):
                    raise RuntimeError(f"Unexpected splitting rule at B={budget}")
                if independent.pilot_budget is not None or independent.pilot_rule is not None:
                    raise RuntimeError(f"Independent row has a pilot at B={budget}")
            schedule_rows[schedule] = rows

        if set(schedule_rows) != {FOUR, NINE}:
            raise RuntimeError(f"Missing four- or nine-split data in {manifest_path}")
        all_rows[model["directory"]] = schedule_rows
    return all_rows


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
        and mmd.get("kernel") == "equal_weight_multiscale_gaussian"
        and mmd.get("representation") == "quantized_image_target"
        and mmd.get("num_frequencies") == 1_024
        and mmd.get("bandwidth_multipliers")
        == [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
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
            target == model["target"] and reference.get("target_spec") == model["target"]
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
    if not row.is_adaptive:
        return spec.get("mode") == "fixed_N"

    expected_query = {
        "subset_sizes": row.model["subset_sizes"],
        "mass_min": 0.05,
        "mass_max": 0.95,
        "rank_spread": 0.4,
        "subset_seed": 0,
        "mass_bins": 16,
    }
    return (
        spec.get("mode") == "estimate_and_sample"
        and tuple(float(value) for value in spec.get("split_percentages", []))
        == row.schedule
        and spec.get("B1") == row.pilot_budget
        and spec.get("optimization_mode") == row.optimizer
        and spec.get("reuse_phase1_samples") == row.reuse
        and spec.get("sigma_estimation_mode") == "crossfit_q"
        and key.get("crossfit_q_folds") == 1
        and key.get("grid_free_params") == EXPECTED_GRID_PARAMS
        and key.get("phase1_query_params") == expected_query
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
            if record.get("extinct", False):
                continue
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
) -> None:
    record_cache: dict[
        tuple[Path, str, int], tuple[np.ndarray, np.ndarray | None]
    ] = {}
    for model in MODELS:
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
                        float(samples.std(ddof=1)), row.std, rel_tol=1e-11, abs_tol=1e-13
                    )
                    if mean_matches and std_matches:
                        matches.append((samples, factors))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one validated cache for {row.csv_path.name}, "
                        f"B={row.budget}, B1={row.pilot_budget}; found {len(matches)}"
                    )
                row.samples, row.split_factors = matches[0]
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


def _group_at_budget(rows: list[ResultRow], budget: int) -> tuple[ResultRow, ResultRow]:
    group = [row for row in rows if row.budget == budget]
    independent = next(row for row in group if not row.is_adaptive)
    splitting = next(row for row in group if row.is_adaptive)
    return independent, splitting


def _normal_reduction(
    independent: ResultRow, splitting: ResultRow
) -> tuple[float, float, float]:
    if independent.mean <= 0.0:
        raise RuntimeError("Independent-path mean metric must be positive")
    observed = 100.0 * (1.0 - splitting.mean / independent.mean)
    if independent.n < 2 or splitting.n < 2:
        return observed, math.nan, math.nan
    variance = 100.0**2 * (
        splitting.mean**2
        * independent.std**2
        / (independent.mean**4 * independent.n)
        + splitting.std**2 / (independent.mean**2 * splitting.n)
    )
    if not math.isfinite(variance) or variance < 0.0:
        raise RuntimeError("Invalid delta-method variance for metric reduction")
    half = Z_975 * math.sqrt(variance)
    return observed, observed - half, observed + half


def _compute_reductions(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
) -> dict[tuple[str, tuple[float, ...], int], tuple[float, float, float]]:
    reductions: dict[
        tuple[str, tuple[float, ...], int], tuple[float, float, float]
    ] = {}
    for model in MODELS:
        for schedule, _ in SCHEDULES:
            rows = all_rows[model["directory"]][schedule]
            for budget in sorted({row.budget for row in rows}):
                independent, splitting = _group_at_budget(rows, budget)
                reductions[(model["directory"], schedule, budget)] = _normal_reduction(
                    independent, splitting
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


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": "paper_plots.py"},
    )
    plt.close(fig)


def _plot_splitting_diagram(output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
        }
    )
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
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

    def connect(parent: tuple[float, float], child: tuple[float, float], width: float) -> None:
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
    ax.set_ylim(-5.95, 4.15)
    ax.axis("off")
    fig.savefig(
        output_dir / "splitting_diagram.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.18,
        metadata={"Software": "paper_plots.py"},
    )
    plt.close(fig)


def _plot_reductions(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]],
    reductions: dict[
        tuple[str, tuple[float, ...], int], tuple[float, float, float]
    ],
    output_dir: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.25), sharey=True)
    colors = {FOUR: "#0072B2", NINE: "#D55E00"}
    markers = {FOUR: "o", NINE: "s"}
    linestyles = {FOUR: "-", NINE: "--"}
    interval_extrema: list[float] = []

    for ax, model in zip(axes.flat, MODELS):
        rows_by_schedule = all_rows[model["directory"]]
        for schedule, label in SCHEDULES:
            budgets = sorted({row.budget for row in rows_by_schedule[schedule]})
            values = np.asarray(
                [
                    reductions[(model["directory"], schedule, budget)]
                    for budget in budgets
                ]
            )
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
                    color=colors[schedule],
                    alpha=0.14,
                    linewidth=0,
                )
            ax.plot(
                budgets,
                observed,
                color=colors[schedule],
                marker=markers[schedule],
                linestyle=linestyles[schedule],
                linewidth=1.6,
                markersize=3.8,
                label=label,
            )
        ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_title(model["plot_title"])
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55)
        ax.xaxis.set_major_formatter(FuncFormatter(_budget_tick))
        ax.tick_params(axis="x", which="minor", bottom=False)

    low = min(interval_extrema)
    high = max(interval_extrema)
    pad = 0.08 * (high - low)
    for ax in axes.flat:
        ax.set_ylim(math.floor(low - pad), math.ceil(high + pad))
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean Metric Reduction (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Budget $B$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.15, w_pad=1.0)
    _save_figure(fig, output_dir, "experiment_metric_gain")


def _plot_allocations(
    all_rows: dict[str, dict[tuple[float, ...], list[ResultRow]]], output_dir: Path
) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.25))
    colors = {FOUR: "#0072B2", NINE: "#D55E00"}
    markers = {FOUR: "o", NINE: "s"}
    linestyles = {FOUR: "-", NINE: "--"}
    elapsed = {
        FOUR: np.asarray([0.2, 0.4, 0.6, 0.8]),
        NINE: np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    }

    for ax, model in zip(axes.flat, MODELS):
        for schedule, label in SCHEDULES:
            rows = all_rows[model["directory"]][schedule]
            if ALLOCATION_BUDGET not in {row.budget for row in rows}:
                raise RuntimeError(
                    f"Missing B={ALLOCATION_BUDGET} for {model['directory']}"
                )
            _, splitting = _group_at_budget(rows, ALLOCATION_BUDGET)
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
            mean = cumulative.mean(axis=0)
            allocation_runs = cumulative.shape[0]
            if allocation_runs > 1:
                standard_error = cumulative.std(axis=0, ddof=1) / math.sqrt(
                    allocation_runs
                )
                lower = mean - Z_975 * standard_error
                upper = mean + Z_975 * standard_error
                ax.fill_between(
                    elapsed[schedule],
                    lower,
                    upper,
                    color=colors[schedule],
                    alpha=0.14,
                    linewidth=0,
                )
            ax.plot(
                elapsed[schedule],
                mean,
                color=colors[schedule],
                marker=markers[schedule],
                linestyle=linestyles[schedule],
                linewidth=1.5,
                markersize=3.5,
                label=label,
            )
        ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_title(model["plot_title"].rsplit(" (", 1)[0])
        ax.set_xlim(0.05, 0.95)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Learned $R_i$")
    for ax in axes[-1, :]:
        ax.set_xlabel("Elapsed fraction of trajectory")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.15, w_pad=1.0)
    _save_figure(fig, output_dir, "experiment_allocations_four_models")


def main() -> None:
    args = _parse_args()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _load_manifest_rows(outputs_root)
    _attach_and_verify_caches(outputs_root, all_rows)
    reductions = _compute_reductions(all_rows)
    _plot_splitting_diagram(output_dir)
    _plot_reductions(all_rows, reductions, output_dir)
    _plot_allocations(all_rows, output_dir)

    counts = ", ".join(
        f"{model['title']}: {model['n_runs']} runs" for model in MODELS
    )
    print(f"Validated {counts}; wrote publication PNGs to {output_dir}")


if __name__ == "__main__":
    main()
