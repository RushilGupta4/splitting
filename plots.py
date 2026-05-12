import argparse
import csv
import math
import os
import re
from collections import defaultdict
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

SOLVER_LINESTYLES = ("--", "-.", ":", (0, (3, 1, 1, 1)))
SOLVER_MARKERS = ("^", "D", "x", "P", "v", "*", "h")
SCHEDULE_MARKERS = ("o", "s", "^", "D", "P", "v", "*", "h", "X")
ADAPTIVE_LINESTYLES = ("-", "--", ":", "-.")
ADAPTIVE_MARKERS = ("o", "s", "D", "^", "v", "P")

PLOT_SPECS = {
    "main": {"filename_suffix": None, "title": "Mean KS vs B"},
    "best_config": {
        "filename_suffix": "best_config",
        "title": "Best adaptive config by B",
    },
    "optimal_b1": {"filename_suffix": "optimal_B1", "title": "Optimal B'"},
    "optimal_ni": {"filename_suffix": "optimal_Ni", "title": "Chosen N_i"},
    "percent_change": {
        "filename_suffix": "percent_change",
        "title": "Percent Change in KS",
    },
}
PLOT_NAMES = tuple(PLOT_SPECS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots from compare.py CSV output"
    )
    parser.add_argument("csv_file", type=str, help="Path to compare.py summary CSV")
    parser.add_argument(
        "--plots",
        type=str,
        default="all",
        help=(
            f"Comma-separated subset of: {','.join(PLOT_NAMES)}, or 'all' "
            "(default: all)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the 'main' figure (other plots derive from csv stem)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title / title prefix",
    )
    parser.add_argument(
        "--std",
        action="store_true",
        help="Show mean_ks +/- std_ks intervals on main / best_config plots",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=None,
        help="Show CI half-widths at this level on main / best_config plots",
    )
    return parser.parse_args()


def _parse_plots(raw: str):
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("--plots must include at least one plot name")
    if "all" in names:
        return list(PLOT_NAMES)
    unknown = [n for n in names if n not in PLOT_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown plot name(s): {', '.join(unknown)}. "
            f"Available: {', '.join(PLOT_NAMES)}, all"
        )
    return names


# --- CSV parsing -------------------------------------------------------------


def _parse_int(value: str):
    value = (value or "").strip()
    if not value:
        return None
    return int(value)


def _parse_float(value: str):
    value = (value or "").strip()
    if not value:
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def _parse_float_list(value: str):
    value = (value or "").strip()
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_bool(value: str):
    value = (value or "").strip().lower()
    if not value:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Unexpected boolean value: {value}")


def _parse_method_label(label: str):
    label = (label or "").strip()
    if label == "all_ones_baseline":
        return ("baseline", "", None, "", None, None)
    match = re.match(r"^(pilot_tree|independent)_(reuse|fresh)$", label)
    if match:
        return ("adaptive", match.group(1), match.group(2) == "reuse", "", None, None)
    match = re.match(r"^(.+)_([0-9]+)(?:_eta([-+0-9.eE]+))?$", label)
    if match:
        return (
            "solver",
            "",
            None,
            match.group(1),
            int(match.group(2)),
            float(match.group(3)) if match.group(3) is not None else None,
        )
    return ("solver", "", None, label, None, None)


def _load_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            if raw_row.get("error", "").strip():
                continue

            mean_ks = _parse_float(raw_row.get("mean_ks", ""))
            B = _parse_int(raw_row.get("B", ""))
            sampling_label = raw_row.get("sampling_label", "").strip()
            method_label = raw_row.get("method_label", "").strip()
            kind, sigma_estimation_mode, reuse_phase1_samples, solver, solver_steps, solver_eta = _parse_method_label(method_label)
            if mean_ks is None or B is None or not sampling_label or not method_label:
                continue
            mode = {
                "baseline": "fixed_N",
                "solver": "solver_baseline",
                "adaptive": "estimate_and_sample",
            }[kind]

            common = {
                "B": B,
                "sampling_label": sampling_label,
                "nfe_per_sample": _parse_int(raw_row.get("nfe_per_sample", "")),
                "mean_ks": mean_ks,
                "std_ks": _parse_float(raw_row.get("std_ks", "")),
                "n_valid_runs": _parse_int(raw_row.get("n_valid_runs", "")),
                "mode": mode,
                "method_label": method_label,
                "solver": solver,
                "solver_sampling_steps": solver_steps,
                "solver_eta": solver_eta,
            }
            empty_adaptive = {
                "B1": None,
                "sigma_estimation_mode": "",
                "reuse_phase1_samples": None,
                "N_i": [],
                "N_i_std": [],
                "N_i_count": None,
                "mode_key": None,
            }

            if mode in ("fixed_N", "solver_baseline"):
                rows.append({**common, **empty_adaptive})
                continue
            if mode != "estimate_and_sample":
                continue

            B1 = _parse_int(raw_row.get("B1", ""))
            if B1 is None or not sigma_estimation_mode or reuse_phase1_samples is None:
                continue

            rows.append(
                {
                    **common,
                    "B1": B1,
                    "sigma_estimation_mode": sigma_estimation_mode,
                    "reuse_phase1_samples": reuse_phase1_samples,
                    "N_i": _parse_float_list(raw_row.get("N_i", "")),
                    "N_i_std": _parse_float_list(raw_row.get("N_i_std", "")),
                    "N_i_count": len(_parse_float_list(raw_row.get("N_i", ""))),
                    "mode_key": (
                        sigma_estimation_mode,
                        reuse_phase1_samples,
                    ),
                }
            )
    return rows


# --- Keys / formatters -------------------------------------------------------


def _schedule_key(row):
    return row["sampling_label"]


def _format_schedule(schedule) -> str:
    return str(schedule)


def _series_key(row):
    kind, sigma_mode, reuse, _solver, _steps, _eta = _parse_method_label(row["method_label"])
    if kind == "baseline":
        return ("baseline",)
    if kind == "solver":
        return ("solver", row["method_label"])
    return (
        "adaptive",
        sigma_mode,
        reuse,
        row["B1"],
    )


def _format_eta(eta: float) -> str:
    return f"{eta:g}"


def _format_solver_label(method_label, solver=None, solver_sampling_steps=None, solver_eta=None):
    if solver and solver_sampling_steps is not None:
        solver_label = solver.upper() if solver == "ddim" else solver.replace("_", " ")
        label = f"{solver_label} {solver_sampling_steps}"
        if solver_eta is not None:
            label += f" eta={_format_eta(solver_eta)}"
        return label
    return method_label or "solver baseline"


def _mode_title(mode_key):
    sigma_estimation_mode, reuse_phase1_samples = mode_key
    parts = [sigma_estimation_mode, "reuse" if reuse_phase1_samples else "fresh"]
    return " ".join(parts)


def _adaptive_rows(rows):
    return [r for r in rows if r["mode"] == "estimate_and_sample"]


def _baseline_rows(rows):
    return [r for r in rows if r["mode"] == "fixed_N"]


def _solver_rows(rows):
    return [r for r in rows if r["mode"] == "solver_baseline"]


def _mode_keys(rows):
    return sorted(
        {r["mode_key"] for r in _adaptive_rows(rows) if r["mode_key"] is not None}
    )


# --- Style maps --------------------------------------------------------------


def _build_b1_color_map(rows):
    b1_values = sorted({row["B1"] for row in rows if row["B1"] not in (None, 0)})
    if not b1_values:
        return {}
    if len(b1_values) <= 10:
        cmap = plt.get_cmap("tab10")
        return {b1: cmap(i) for i, b1 in enumerate(b1_values)}
    cmap = plt.get_cmap("viridis")
    denom = max(len(b1_values) - 1, 1)
    return {b1: cmap(i / denom) for i, b1 in enumerate(b1_values)}


def _build_solver_style_map(rows):
    solver_rows_by_label = {}
    for row in _solver_rows(rows):
        solver_rows_by_label.setdefault(row["method_label"], row)
    if not solver_rows_by_label:
        return {}
    labels = sorted(solver_rows_by_label)
    cmap = plt.get_cmap("tab10") if len(labels) <= 10 else plt.get_cmap("tab20")
    return {
        method_label: (
            cmap(i % cmap.N),
            SOLVER_LINESTYLES[i % len(SOLVER_LINESTYLES)],
            SOLVER_MARKERS[i % len(SOLVER_MARKERS)],
            _format_solver_label(method_label),
        )
        for i, method_label in enumerate(labels)
    }


def _solver_style(method_label, solver_styles):
    return solver_styles.get(
        method_label,
        ("#7f7f7f", "--", "x", method_label or "solver baseline"),
    )


def _build_schedule_style_map(rows):
    schedules = sorted({_schedule_key(row) for row in rows})
    if not schedules:
        return {}
    cmap = plt.get_cmap("tab10") if len(schedules) <= 10 else plt.get_cmap("tab20")
    return {
        schedule: (
            cmap(i % cmap.N),
            SCHEDULE_MARKERS[i % len(SCHEDULE_MARKERS)],
        )
        for i, schedule in enumerate(schedules)
    }


def _build_mode_style_map(rows):
    mode_keys = _mode_keys(rows)
    if not mode_keys:
        return {}
    cmap = plt.get_cmap("tab10") if len(mode_keys) <= 10 else plt.get_cmap("tab20")
    return {
        mode_key: (
            cmap(i % cmap.N),
            ADAPTIVE_LINESTYLES[i % len(ADAPTIVE_LINESTYLES)],
            ADAPTIVE_MARKERS[i % len(ADAPTIVE_MARKERS)],
        )
        for i, mode_key in enumerate(mode_keys)
    }


def _mode_style(mode_key, mode_styles):
    return mode_styles.get(mode_key, ("#7f7f7f", "-", "o"))


# --- CI / shared plotting helpers --------------------------------------------


def _ci_multiplier(confidence_level: float, degrees_of_freedom: int):
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("--ci must be between 0 and 1, e.g. 0.95")
    quantile = 0.5 + confidence_level / 2.0
    try:
        from scipy.stats import t

        return float(t.ppf(quantile, degrees_of_freedom))
    except ImportError:
        return NormalDist().inv_cdf(quantile)


def _ci_half_width(std_ks, n_valid_runs, ci_level):
    if std_ks is None or n_valid_runs is None or n_valid_runs <= 1:
        return None
    # compare.py stores population std (ddof=0); convert to the usual SE from
    # the unbiased sample variance: sqrt(n/(n-1))*std/sqrt(n) = std/sqrt(n-1).
    return (
        _ci_multiplier(ci_level, n_valid_runs - 1)
        * std_ks
        / math.sqrt(n_valid_runs - 1)
    )


def _fill_interval(ax, x_values, y_values, half_widths, color, zorder):
    if any(width is None for width in half_widths):
        return
    lower = [max(y - w, 1e-12) for y, w in zip(y_values, half_widths)]
    upper = [y + w for y, w in zip(y_values, half_widths)]
    ax.fill_between(
        x_values, lower, upper, color=color, alpha=0.16, linewidth=0, zorder=zorder
    )


def _make_subplot_grid(count: int, max_cols: int = 2):
    ncols = min(max_cols, count)
    nrows = math.ceil(count / ncols)
    return nrows, ncols


def _apply_scalar_formatters(ax, format_x: bool = True, format_y: bool = True):
    if format_x:
        x_formatter = ScalarFormatter(useMathText=True)
        x_formatter.set_powerlimits((-2, 3))
        ax.xaxis.set_major_formatter(x_formatter)
    if format_y:
        y_formatter = ScalarFormatter(useMathText=True)
        y_formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(y_formatter)


# --- main: mean KS vs B per schedule ----------------------------------------


def _plot_schedule(ax, schedule_rows, b1_colors, solver_styles, show_std, ci_level):
    grouped = defaultdict(list)
    for row in schedule_rows:
        grouped[_series_key(row)].append(row)

    for series_key, series_rows in sorted(grouped.items()):
        points = sorted(series_rows, key=lambda r: r["B"])
        x_values = [r["B"] for r in points]
        y_values = [r["mean_ks"] for r in points]
        std_values = [r["std_ks"] for r in points]
        ci_values = (
            [_ci_half_width(r["std_ks"], r["n_valid_runs"], ci_level) for r in points]
            if ci_level is not None
            else None
        )

        if series_key[0] == "baseline":
            if show_std:
                _fill_interval(ax, x_values, y_values, std_values, "black", 2)
            if ci_values is not None:
                _fill_interval(ax, x_values, y_values, ci_values, "black", 2)
            ax.plot(
                x_values,
                y_values,
                color="black",
                linestyle="-",
                marker="o",
                linewidth=2.2,
                markersize=5,
                zorder=3,
            )
            continue

        if series_key[0] == "solver":
            method_label = series_key[1]
            color, linestyle, marker, _ = _solver_style(method_label, solver_styles)
            if show_std:
                _fill_interval(ax, x_values, y_values, std_values, color, 3)
            if ci_values is not None:
                _fill_interval(ax, x_values, y_values, ci_values, color, 3)
            ax.plot(
                x_values,
                y_values,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5,
                zorder=4,
            )
            continue

        _, sigma_estimation_mode, reuse_phase1_samples, B1 = series_key
        color = b1_colors.get(B1, "#1f77b4")
        linestyle = ":" if sigma_estimation_mode == "pilot_tree" else "-"
        marker = "s" if reuse_phase1_samples else "o"
        if show_std:
            _fill_interval(ax, x_values, y_values, std_values, color, 1)
        if ci_values is not None:
            _fill_interval(ax, x_values, y_values, ci_values, color, 1)
        ax.plot(
            x_values,
            y_values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )

    ax.grid(alpha=0.25)
    ax.set_xlabel("Total B")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(formatter)


def _make_main_legends(fig, b1_colors, rows, solver_styles, schedule_styles):
    semantic_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", linewidth=2.2),
        Line2D([0], [0], color="0.35", linestyle="-", linewidth=1.8),
        Line2D([0], [0], color="0.35", linestyle=":", linewidth=2.2),
        Line2D([0], [0], color="0.35", linestyle="None", marker="o", markersize=6),
        Line2D([0], [0], color="0.35", linestyle="None", marker="s", markersize=6),
    ]
    semantic_labels = [
        "baseline (fixed N)",
        "independent sigma estimate",
        "pilot_tree sigma estimate",
        "fresh samples",
        "reuse phase-1 samples",
    ]

    seen_solver_labels = []
    for row in rows:
        if row["mode"] != "solver_baseline":
            continue
        if row["method_label"] in seen_solver_labels:
            continue
        seen_solver_labels.append(row["method_label"])

    for method_label in sorted(seen_solver_labels):
        color, linestyle, marker, label = _solver_style(method_label, solver_styles)
        semantic_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=6,
            )
        )
        semantic_labels.append(label)

    fig.legend(
        semantic_handles,
        semantic_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )

    if schedule_styles:
        schedule_handles = []
        schedule_labels = []
        for schedule, (color, marker) in sorted(schedule_styles.items()):
            schedule_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="None",
                    marker=marker,
                    markersize=6,
                )
            )
            schedule_labels.append(_format_schedule(schedule))
        fig.legend(
            schedule_handles,
            schedule_labels,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            title="Sampling config",
        )

    if not b1_colors:
        return

    b1_handles = [
        Line2D([0], [0], color=color, linestyle="-", linewidth=2.0)
        for _, color in sorted(b1_colors.items())
    ]
    b1_labels = [f"B1={b1}" for b1 in sorted(b1_colors)]
    fig.legend(
        b1_handles,
        b1_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(5, max(1, len(b1_labels))),
        frameon=False,
        title="Adaptive budgets",
    )


def _plot_main(rows, b1_colors, solver_styles, args, output_path):
    solver_rows = _solver_rows(rows)
    non_solver_rows = [r for r in rows if r["mode"] != "solver_baseline"]
    rows_for_panels = non_solver_rows or rows
    schedules = sorted({_schedule_key(row) for row in rows_for_panels})
    schedule_groups = defaultdict(list)
    for row in rows_for_panels:
        schedule_groups[_schedule_key(row)].append(row)
    schedule_styles = _build_schedule_style_map(non_solver_rows)

    n_schedules = len(schedules)
    ncols = min(2, n_schedules)
    nrows = math.ceil(n_schedules / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10 * ncols, 7 * nrows),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, schedule in zip(axes_flat, schedules):
        _plot_schedule(
            ax,
            schedule_groups[schedule] + solver_rows,
            b1_colors,
            solver_styles,
            args.std,
            args.ci,
        )
        ax.set_title(_format_schedule(schedule))

    for ax in axes_flat[n_schedules:]:
        ax.set_visible(False)

    for row_axes in axes:
        row_axes[0].set_ylabel("Mean KS")
        row_axes[0].set_yscale("log")

    if args.title:
        fig.suptitle(args.title)

    _make_main_legends(fig, b1_colors, rows, solver_styles, schedule_styles)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return n_schedules


# --- best_config -------------------------------------------------------------


def _best_adaptive_rows(rows):
    grouped = defaultdict(list)
    for row in _adaptive_rows(rows):
        grouped[(row["B"], _schedule_key(row))].append(row)
    selected = []
    for candidates in grouped.values():
        selected.append(
            min(
                candidates,
                key=lambda r: (
                    r["mean_ks"],
                    r["method_label"],
                    r["B1"] if r["B1"] is not None else -1,
                ),
            )
        )
    return sorted(selected, key=lambda r: (_schedule_key(r), r["B"]))


def _best_by_group(rows, *, key_fn, score_fn):
    """Group rows by key_fn(row), keeping the row that minimizes score_fn(row)."""
    out = {}
    for row in rows:
        k = key_fn(row)
        if k not in out or score_fn(row) < score_fn(out[k]):
            out[k] = row
    return out


def _best_solver_rows(rows):
    selected = _best_by_group(
        _solver_rows(rows),
        key_fn=lambda r: (r["B"], r["method_label"]),
        score_fn=lambda r: (r["mean_ks"], r.get("nfe_per_sample") or 0),
    )
    return sorted(selected.values(), key=lambda r: (r["method_label"], r["B"]))


def _best_adaptive_improvement_points(rows):
    score = lambda r: (r["mean_ks"], r["method_label"])
    by_b = lambda r: r["B"]
    best_adaptive = _best_by_group(
        (r for r in rows if r["mode"] == "estimate_and_sample"),
        key_fn=by_b, score_fn=score,
    )
    best_baseline = _best_by_group(
        (r for r in rows if r["mode"] in {"fixed_N", "solver_baseline"}),
        key_fn=by_b, score_fn=score,
    )
    x_values, y_values = [], []
    for budget in sorted(set(best_adaptive) & set(best_baseline)):
        baseline = best_baseline[budget]
        adaptive = best_adaptive[budget]
        if baseline["mean_ks"] == 0:
            continue
        x_values.append(budget)
        y_values.append(
            100.0 * (baseline["mean_ks"] - adaptive["mean_ks"]) / baseline["mean_ks"]
        )
    return x_values, y_values


def _best_config_series_key(row):
    schedule = _schedule_key(row)
    if row["mode"] == "fixed_N":
        return ("fixed_N", schedule)
    if row["mode"] == "solver_baseline":
        return ("solver", row["method_label"])
    return ("best_adaptive", schedule)


def _format_best_config_label(series_key):
    kind = series_key[0]
    if kind == "fixed_N":
        schedule_label = _format_schedule(series_key[1])
        return f"fixed N, {schedule_label}"
    if kind == "solver":
        _, method_label = series_key
        return _format_solver_label(method_label)
    schedule_label = _format_schedule(series_key[1])
    return f"best adaptive, {schedule_label}"


def _plot_best_config(rows, output_path, show_std, ci_level, title=None):
    selected_rows = _baseline_rows(rows)
    selected_rows.extend(_best_solver_rows(rows))
    selected_rows.extend(_best_adaptive_rows(rows))
    if not selected_rows:
        return False

    schedule_styles = _build_schedule_style_map(selected_rows)
    solver_styles = _build_solver_style_map(selected_rows)
    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[_best_config_series_key(row)].append(row)

    fig, (ax, improvement_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    for series_key, series_rows in sorted(grouped.items()):
        points = sorted(series_rows, key=lambda r: r["B"])
        x_values = [r["B"] for r in points]
        y_values = [r["mean_ks"] for r in points]
        std_values = [r["std_ks"] for r in points]
        ci_values = (
            [_ci_half_width(r["std_ks"], r["n_valid_runs"], ci_level) for r in points]
            if ci_level is not None
            else None
        )

        kind = series_key[0]
        linestyle = "-"
        marker = "o"
        linewidth = 2.0
        zorder = 3
        alpha = 0.95

        if kind == "solver":
            method_label = series_key[1]
            color, linestyle, marker, _ = _solver_style(method_label, solver_styles)
            linewidth = 1.9
            zorder = 4
        else:
            schedule = series_key[1]
            color, schedule_marker = schedule_styles.get(schedule, ("#1f77b4", "o"))
            marker = schedule_marker
            if kind == "best_adaptive":
                linestyle = ":"
                marker = "*"
                linewidth = 2.4
                zorder = 5
            elif kind == "fixed_N":
                linestyle = "-"
                linewidth = 2.2
                zorder = 4

        if show_std:
            _fill_interval(ax, x_values, y_values, std_values, color, zorder - 1)
        if ci_values is not None:
            _fill_interval(ax, x_values, y_values, ci_values, color, zorder - 1)

        ax.plot(
            x_values,
            y_values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            markersize=6 if kind != "best_adaptive" else 8,
            alpha=alpha,
            zorder=zorder,
            label=_format_best_config_label(series_key),
        )

    ax.grid(alpha=0.25)
    ax.set_ylabel("Mean KS")
    ax.set_yscale("log")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_title(title or PLOT_SPECS["best_config"]["title"])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    improvement_x, improvement_y = _best_adaptive_improvement_points(rows)
    if improvement_x:
        improvement_ax.plot(
            improvement_x,
            improvement_y,
            color="black",
            linestyle="-",
            marker="o",
            linewidth=2.0,
            markersize=5,
            label="best adaptive vs best baseline",
        )
        improvement_ax.legend(frameon=False, loc="best")
    else:
        improvement_ax.text(
            0.5,
            0.5,
            "no matched adaptive/baseline data",
            ha="center",
            va="center",
            transform=improvement_ax.transAxes,
        )
    improvement_ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    improvement_ax.grid(alpha=0.25)
    improvement_ax.set_xlabel("Total B")
    improvement_ax.set_ylabel("KS improvement (%)")
    improvement_ax.xaxis.set_major_formatter(formatter)
    fig.tight_layout(rect=(0, 0, 0.78, 1))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


# --- optimal_b1 / optimal_ni / percent_change --------------------------------


def _make_mode_legend(fig, mode_keys, mode_styles):
    handles = []
    labels = []
    for mode_key in mode_keys:
        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=5,
            )
        )
        labels.append(_mode_title(mode_key))
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )


def _plot_optimal_b1_panel(ax, selected_rows, mode_keys, mode_styles):
    plotted_any = False
    for mode_key in mode_keys:
        points = [r for r in selected_rows if r["mode_key"] == mode_key]
        points.sort(key=lambda r: r["B"])
        if not points:
            continue
        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        ax.plot(
            [r["B"] for r in points],
            [r["B1"] for r in points],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )
        plotted_any = True
    if not plotted_any:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)


def _plot_optimal_b1(rows, title_prefix, mode_styles):
    rows = _adaptive_rows(rows)
    mode_keys = _mode_keys(rows)
    schedules = sorted({_schedule_key(row) for row in rows})
    selected = _best_by_group(
        rows,
        key_fn=lambda r: (_schedule_key(r), r["mode_key"], r["B"]),
        score_fn=lambda r: (r["mean_ks"], r["B1"]),
    )
    selected_by_schedule = defaultdict(list)
    for (schedule, _, _), row in selected.items():
        selected_by_schedule[schedule].append(row)

    nrows, ncols = _make_subplot_grid(len(schedules), max_cols=2)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 4.8 * nrows),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.flatten()
    for ax, schedule in zip(axes_flat, schedules):
        _plot_optimal_b1_panel(
            ax, selected_by_schedule.get(schedule, []), mode_keys, mode_styles
        )
        ax.set_title(_format_schedule(schedule))
        ax.set_xlabel("B")
        _apply_scalar_formatters(ax, format_x=True, format_y=True)
    for ax in axes_flat[len(schedules) :]:
        ax.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("Optimal B'")
    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS['optimal_b1']['title']}")
    _make_mode_legend(fig, mode_keys, mode_styles)
    return fig


def _plot_optimal_ni_panel(ax, selected_rows, max_levels, mode_keys, mode_styles):
    plotted_any = False
    for mode_key in mode_keys:
        matching_rows = [r for r in selected_rows if r["mode_key"] == mode_key]
        if not matching_rows:
            continue
        row = min(matching_rows, key=lambda r: (r["mean_ks"], r["B1"]))
        if not row["N_i"]:
            continue
        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        x_values = list(range(1, len(row["N_i"]) + 1))
        ni_std = row.get("N_i_std") or []
        if len(ni_std) == len(row["N_i"]):
            lower = [max(0.0, m - s) for m, s in zip(row["N_i"], ni_std)]
            upper = [m + s for m, s in zip(row["N_i"], ni_std)]
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.16, linewidth=0)
        ax.plot(
            x_values,
            row["N_i"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )
        plotted_any = True
    if not plotted_any:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    if max_levels > 0:
        ax.set_xticks(list(range(1, max_levels + 1)))
    ax.grid(alpha=0.25)
    _apply_scalar_formatters(ax, format_x=False, format_y=True)


def _plot_optimal_ni(rows, title_prefix, mode_styles):
    rows = _adaptive_rows(rows)
    mode_keys = _mode_keys(rows)
    schedules = sorted({_schedule_key(row) for row in rows})
    budgets = sorted({r["B"] for r in rows})
    selected = _best_by_group(
        rows,
        key_fn=lambda r: (_schedule_key(r), r["mode_key"], r["B"]),
        score_fn=lambda r: (r["mean_ks"], r["B1"]),
    )
    selected_by_schedule_and_budget = defaultdict(list)
    for (schedule, _, B), row in selected.items():
        selected_by_schedule_and_budget[(schedule, B)].append(row)
    max_levels = max((len(r["N_i"]) for r in selected.values()), default=0)

    fig, axes = plt.subplots(
        len(schedules),
        len(budgets),
        figsize=(4.8 * len(budgets), 4.0 * len(schedules)),
        squeeze=False,
    )
    for row_idx, schedule in enumerate(schedules):
        for col_idx, budget in enumerate(budgets):
            ax = axes[row_idx][col_idx]
            panel_rows = selected_by_schedule_and_budget.get((schedule, budget), [])
            _plot_optimal_ni_panel(ax, panel_rows, max_levels, mode_keys, mode_styles)
            if row_idx == 0:
                ax.set_title(f"B={budget}")
            if col_idx == 0:
                ax.set_ylabel(f"Chosen N_i\n{_format_schedule(schedule)}")
            if row_idx == len(schedules) - 1:
                ax.set_xlabel("Split level i")
    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS['optimal_ni']['title']}")
    _make_mode_legend(fig, mode_keys, mode_styles)
    return fig


def _percent_change_points(schedule, baseline_by_group, candidate_by_budget):
    budgets = sorted(
        B
        for group_schedule, B in baseline_by_group
        if group_schedule == schedule and B in candidate_by_budget
    )
    x_values = []
    y_values = []
    for budget in budgets:
        baseline = baseline_by_group[(schedule, budget)]
        candidate = candidate_by_budget[budget]
        if baseline["mean_ks"] == 0:
            continue
        x_values.append(budget)
        y_values.append(
            100.0 * (baseline["mean_ks"] - candidate["mean_ks"]) / baseline["mean_ks"]
        )
    return x_values, y_values


def _plot_percent_change_panel(
    ax,
    schedule,
    baseline_by_group,
    best_adaptive_by_group,
    solver_by_group,
    solver_styles,
):
    plotted_any = False
    adaptive_candidates = {
        B: row
        for (group_schedule, B), row in best_adaptive_by_group.items()
        if group_schedule == schedule
    }
    x_values, y_values = _percent_change_points(
        schedule, baseline_by_group, adaptive_candidates
    )
    if x_values:
        ax.plot(
            x_values,
            y_values,
            color="#2ca02c",
            linestyle="-",
            marker="o",
            linewidth=1.8,
            markersize=5,
            label="best adaptive",
        )
        for x_value, y_value in zip(x_values, y_values):
            ax.annotate(
                f"{y_value:.1f}%",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="#2ca02c",
            )
        plotted_any = True

    solver_labels = sorted(
        {
            method_label
            for _, method_label in solver_by_group
        }
    )
    for method_label in solver_labels:
        solver_candidates = {
            B: row
            for (B, group_method_label), row in solver_by_group.items()
            if group_method_label == method_label
        }
        x_values, y_values = _percent_change_points(
            schedule, baseline_by_group, solver_candidates
        )
        if not x_values:
            continue
        color, linestyle, marker, label = _solver_style(method_label, solver_styles)
        ax.plot(
            x_values,
            y_values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            label=label,
        )
        plotted_any = True

    if plotted_any:
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)
    _apply_scalar_formatters(ax, format_x=True, format_y=True)


def _plot_percent_change(rows, title_prefix, solver_styles):
    non_solver_rows = [r for r in rows if r["mode"] != "solver_baseline"]
    schedules = sorted({_schedule_key(row) for row in non_solver_rows})
    if not schedules:
        return plt.figure(figsize=(7.2, 4.8))
    schedule_budget = lambda r: (_schedule_key(r), r["B"])
    by_mean_ks = lambda r: r["mean_ks"]
    baseline_by_group = _best_by_group(
        _baseline_rows(rows), key_fn=schedule_budget, score_fn=by_mean_ks
    )
    best_adaptive_by_group = _best_by_group(
        _adaptive_rows(rows),
        key_fn=schedule_budget,
        score_fn=lambda r: (r["mean_ks"], r["B1"]),
    )
    solver_by_group = _best_by_group(
        _solver_rows(rows),
        key_fn=lambda r: (r["B"], r["method_label"]),
        score_fn=by_mean_ks,
    )

    nrows, ncols = _make_subplot_grid(len(schedules), max_cols=2)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 4.8 * nrows),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.flatten()
    for ax, schedule in zip(axes_flat, schedules):
        _plot_percent_change_panel(
            ax,
            schedule,
            baseline_by_group,
            best_adaptive_by_group,
            solver_by_group,
            solver_styles,
        )
        ax.set_title(_format_schedule(schedule))
        ax.set_xlabel("B")
    for ax in axes_flat[len(schedules) :]:
        ax.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("KS change vs baseline (%)")
    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS['percent_change']['title']}")
    return fig


# --- output paths / save -----------------------------------------------------


def _default_output_path(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    if not base:
        return csv_path + ".png"
    return base + ".png"


def _suffixed_output_path(csv_path: str, suffix: str) -> str:
    base, _ = os.path.splitext(csv_path)
    if not base:
        base = csv_path
    return f"{base}_{suffix}.png"


def _save_figure(fig, output_path: str, has_title: bool):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    top = 0.92 if has_title else 0.95
    fig.tight_layout(rect=(0, 0, 1, top))
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    plot_names = _parse_plots(args.plots)

    rows = _load_rows(args.csv_file)
    if not rows:
        raise ValueError(
            "No valid rows found in CSV after filtering failed/invalid records"
        )

    b1_colors = _build_b1_color_map(rows)
    solver_styles = _build_solver_style_map(rows)
    mode_styles = _build_mode_style_map(rows)

    output_paths = []
    for plot_name in plot_names:
        if plot_name == "main":
            output_path = args.output or _default_output_path(args.csv_file)
            _plot_main(rows, b1_colors, solver_styles, args, output_path)
            output_paths.append(output_path)
            continue

        if plot_name == "best_config":
            output_path = _suffixed_output_path(
                args.csv_file, PLOT_SPECS["best_config"]["filename_suffix"]
            )
            title = (
                f"{args.title} - best adaptive config by B"
                if args.title
                else PLOT_SPECS["best_config"]["title"]
            )
            if _plot_best_config(rows, output_path, args.std, args.ci, title):
                output_paths.append(output_path)
            continue

        if plot_name == "optimal_b1":
            if not _adaptive_rows(rows):
                continue
            fig = _plot_optimal_b1(rows, args.title, mode_styles)
        elif plot_name == "optimal_ni":
            if not _adaptive_rows(rows):
                continue
            fig = _plot_optimal_ni(rows, args.title, mode_styles)
        elif plot_name == "percent_change":
            fig = _plot_percent_change(rows, args.title, solver_styles)
        else:
            continue

        output_path = _suffixed_output_path(
            args.csv_file, PLOT_SPECS[plot_name]["filename_suffix"]
        )
        _save_figure(fig, output_path, has_title=bool(args.title))
        output_paths.append(output_path)

    print(f"Loaded {len(rows)} valid rows")
    for path in output_paths:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
