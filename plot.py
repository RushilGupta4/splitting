import argparse
import csv
import math
import os
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot compare.py CSV output as mean KS vs total B by inference schedule"
    )
    parser.add_argument("csv_file", type=str, help="Path to compare.py summary CSV")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output PNG (default: same as csv_file with .png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title",
    )
    parser.add_argument(
        "--std",
        action="store_true",
        help="Show mean_ks +/- std_ks intervals when std_ks is available",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=None,
        help="Show confidence interval for mean_ks at this level, e.g. 0.95",
    )
    return parser.parse_args()


def _default_output_path(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    if not base:
        return csv_path + ".png"
    return base + ".png"


def _best_config_output_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    if not base:
        return output_path + "_best_config.png"
    return f"{base}_best_config{ext or '.png'}"


def _parse_int(value: str):
    value = value.strip()
    if not value:
        return None
    return int(value)


def _parse_float(value: str):
    value = value.strip()
    if not value:
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def _parse_bool(value: str):
    value = value.strip().lower()
    if not value:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Unexpected boolean value: {value}")


def _load_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            if raw_row.get("error", "").strip():
                continue

            mean_ks = _parse_float(raw_row.get("mean_ks", ""))
            B = _parse_int(raw_row.get("B", ""))
            sampling_steps = _parse_int(raw_row.get("sampling_steps", ""))
            eta = _parse_float(raw_row.get("eta", ""))
            if mean_ks is None or B is None or sampling_steps is None or eta is None:
                continue

            rows.append(
                {
                    "B": B,
                    "B1": _parse_int(raw_row.get("B1", "")),
                    "sampling_steps": sampling_steps,
                    "eta": eta,
                    "mean_ks": mean_ks,
                    "std_ks": _parse_float(raw_row.get("std_ks", "")),
                    "n_valid_runs": _parse_int(raw_row.get("n_valid_runs", "")),
                    "mode": raw_row.get("mode", "").strip(),
                    "method_label": raw_row.get("method_label", "").strip(),
                    "solver": raw_row.get("solver", "").strip(),
                    "solver_sampling_steps": _parse_int(
                        raw_row.get("solver_sampling_steps", "")
                    ),
                    "solver_eta": _parse_float(raw_row.get("solver_eta", "")),
                    "solver_nfe": _parse_int(raw_row.get("solver_nfe", "")),
                    "sigma_estimation_mode": raw_row.get(
                        "sigma_estimation_mode", ""
                    ).strip(),
                    "bias_type": raw_row.get("bias_type", "").strip(),
                    "reuse_phase1_samples": _parse_bool(
                        raw_row.get("reuse_phase1_samples", "")
                    ),
                }
            )
    return rows


def _format_eta(eta: float) -> str:
    return f"{eta:g}"


def _series_key(row):
    if row["mode"] == "fixed_N":
        return ("baseline",)
    if row["mode"] == "solver_baseline":
        return (
            "solver",
            row["method_label"],
            row["solver"],
            row["solver_sampling_steps"],
            row["solver_eta"],
        )
    return (
        "adaptive",
        row["sigma_estimation_mode"],
        row["bias_type"],
        row["reuse_phase1_samples"],
        row["B1"],
    )


def _schedule_key(row):
    return (row["sampling_steps"], row["eta"])


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


def _format_solver_label(series_key):
    _, method_label, solver, solver_sampling_steps, solver_eta = series_key
    if solver and solver_sampling_steps is not None:
        solver_label = solver.upper() if solver == "ddim" else solver.replace("_", " ")
        label = f"{solver_label} {solver_sampling_steps}"
        if solver_eta is not None:
            label += f" eta={_format_eta(solver_eta)}"
        return label
    return method_label or "solver baseline"


def _build_solver_style_map(rows):
    solver_keys = sorted(
        {_series_key(row) for row in rows if row["mode"] == "solver_baseline"}
    )
    if not solver_keys:
        return {}

    cmap = plt.get_cmap("tab10") if len(solver_keys) <= 10 else plt.get_cmap("tab20")
    return {
        key: (
            cmap(i % cmap.N),
            SOLVER_LINESTYLES[i % len(SOLVER_LINESTYLES)],
            SOLVER_MARKERS[i % len(SOLVER_MARKERS)],
            _format_solver_label(key),
        )
        for i, key in enumerate(solver_keys)
    }


def _solver_style_from_key(series_key, solver_styles):
    return solver_styles.get(
        series_key,
        ("#7f7f7f", "--", "x", _format_solver_label(series_key)),
    )


def _bias_alpha(bias_type: str):
    return 0.55 if bias_type == "biased" else 0.95


def _ci_multiplier(confidence_level: float, degrees_of_freedom: int):
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("--ci must be between 0 and 1, e.g. 0.95")
    quantile = 0.5 + confidence_level / 2.0
    try:
        from scipy.stats import t

        return float(t.ppf(quantile, degrees_of_freedom))
    except ImportError:
        return NormalDist().inv_cdf(quantile)


def _fill_interval(ax, x_values, y_values, half_widths, color, zorder):
    if any(width is None for width in half_widths):
        return

    lower = [max(y - width, 1e-12) for y, width in zip(y_values, half_widths)]
    upper = [y + width for y, width in zip(y_values, half_widths)]
    ax.fill_between(
        x_values,
        lower,
        upper,
        color=color,
        alpha=0.16,
        linewidth=0,
        zorder=zorder,
    )


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


def _plot_schedule(ax, schedule_rows, b1_colors, solver_styles, show_std, ci_level):
    grouped = defaultdict(list)
    for row in schedule_rows:
        grouped[_series_key(row)].append(row)

    for series_key, series_rows in sorted(grouped.items()):
        points = sorted(series_rows, key=lambda row: row["B"])
        x_values = [row["B"] for row in points]
        y_values = [row["mean_ks"] for row in points]
        std_values = [row["std_ks"] for row in points]
        ci_values = (
            [
                _ci_half_width(row["std_ks"], row["n_valid_runs"], ci_level)
                for row in points
            ]
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
            color, linestyle, marker, _ = _solver_style_from_key(
                series_key, solver_styles
            )
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

        _, sigma_estimation_mode, bias_type, reuse_phase1_samples, B1 = series_key
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
            alpha=_bias_alpha(bias_type),
        )

    ax.grid(alpha=0.25)
    ax.set_xlabel("Total B")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(formatter)


def _format_schedule_label(schedule):
    sampling_steps, eta = schedule
    return f"steps={sampling_steps}, eta={_format_eta(eta)}"


def _build_schedule_style_map(rows):
    schedules = sorted({_schedule_key(row) for row in rows})
    if not schedules:
        return {}

    step_values = sorted({schedule[0] for schedule in schedules})
    cmap = plt.get_cmap("tab10") if len(step_values) <= 10 else plt.get_cmap("tab20")
    step_colors = {step: cmap(i % cmap.N) for i, step in enumerate(step_values)}
    return {
        schedule: (
            step_colors[schedule[0]],
            SCHEDULE_MARKERS[i % len(SCHEDULE_MARKERS)],
        )
        for i, schedule in enumerate(schedules)
    }


def _best_adaptive_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["mode"] != "estimate_and_sample":
            continue
        grouped[(row["B"], row["sampling_steps"], row["eta"])].append(row)

    selected = []
    for candidates in grouped.values():
        selected.append(
            min(
                candidates,
                key=lambda row: (
                    row["mean_ks"],
                    row["method_label"],
                    row["B1"] if row["B1"] is not None else -1,
                ),
            )
        )
    return sorted(selected, key=lambda row: (_schedule_key(row), row["B"]))


def _best_by_budget(rows, modes):
    best_by_budget = {}
    for row in rows:
        if row["mode"] not in modes:
            continue
        best = best_by_budget.get(row["B"])
        if best is None or (row["mean_ks"], row["method_label"]) < (
            best["mean_ks"],
            best["method_label"],
        ):
            best_by_budget[row["B"]] = row
    return best_by_budget


def _best_adaptive_improvement_points(rows):
    best_adaptive = _best_by_budget(rows, {"estimate_and_sample"})
    best_baseline = _best_by_budget(rows, {"fixed_N", "solver_baseline"})
    budgets = sorted(set(best_adaptive) & set(best_baseline))
    x_values = []
    y_values = []
    for budget in budgets:
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
        return (
            "solver",
            schedule,
            row["method_label"],
            row["solver"],
            row["solver_sampling_steps"],
            row["solver_eta"],
        )
    return ("best_adaptive", schedule)


def _format_best_config_label(series_key):
    kind = series_key[0]
    schedule_label = _format_schedule_label(series_key[1])
    if kind == "fixed_N":
        return f"fixed N, {schedule_label}"
    if kind == "solver":
        solver_key = ("solver",) + series_key[2:]
        return f"{_format_solver_label(solver_key)}, {schedule_label}"
    return f"best adaptive, {schedule_label}"


def _plot_best_config(rows, output_path, show_std, ci_level, title=None):
    selected_rows = [
        row for row in rows if row["mode"] in {"fixed_N", "solver_baseline"}
    ]
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
        points = sorted(series_rows, key=lambda row: row["B"])
        x_values = [row["B"] for row in points]
        y_values = [row["mean_ks"] for row in points]
        std_values = [row["std_ks"] for row in points]
        ci_values = (
            [
                _ci_half_width(row["std_ks"], row["n_valid_runs"], ci_level)
                for row in points
            ]
            if ci_level is not None
            else None
        )

        kind = series_key[0]
        schedule = series_key[1]
        color, schedule_marker = schedule_styles.get(schedule, ("#1f77b4", "o"))
        linestyle = "-"
        marker = schedule_marker
        linewidth = 2.0
        zorder = 3
        alpha = 0.95

        if kind == "solver":
            solver_key = ("solver",) + series_key[2:]
            _, _, marker, _ = _solver_style_from_key(solver_key, solver_styles)
            linestyle = "--"
            linewidth = 1.9
            zorder = 4
        elif kind == "best_adaptive":
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
    ax.set_title(title or "Best adaptive config by B")
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
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _make_legends(fig, b1_colors, rows, solver_styles):
    semantic_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", linewidth=2.2),
        Line2D([0], [0], color="0.35", linestyle="-", linewidth=1.8),
        Line2D([0], [0], color="0.35", linestyle=":", linewidth=2.2),
        Line2D([0], [0], color="0.35", linestyle="None", marker="o", markersize=6),
        Line2D([0], [0], color="0.35", linestyle="None", marker="s", markersize=6),
        Line2D([0], [0], color="0.35", linestyle="-", linewidth=2.0, alpha=0.55),
        Line2D([0], [0], color="0.35", linestyle="-", linewidth=2.0, alpha=0.95),
    ]
    semantic_labels = [
        "baseline (fixed N)",
        "independent sigma estimate",
        "pilot_tree sigma estimate",
        "fresh samples",
        "reuse phase-1 samples",
        "biased estimator",
        "unbiased estimator",
    ]

    solver_keys = []
    seen_solver_labels = set()
    for row in rows:
        if row["mode"] != "solver_baseline":
            continue
        key = _series_key(row)
        if key[1] in seen_solver_labels:
            continue
        seen_solver_labels.add(key[1])
        solver_keys.append(key)

    for solver_key in sorted(solver_keys):
        color, linestyle, marker, label = _solver_style_from_key(
            solver_key, solver_styles
        )
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


def main():
    args = parse_args()

    rows = _load_rows(args.csv_file)
    if not rows:
        raise ValueError(
            "No valid rows found in CSV after filtering failed/invalid records"
        )

    schedules = sorted({_schedule_key(row) for row in rows})
    schedule_groups = defaultdict(list)
    for row in rows:
        schedule_groups[_schedule_key(row)].append(row)

    b1_colors = _build_b1_color_map(rows)
    solver_styles = _build_solver_style_map(rows)

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
            schedule_groups[schedule],
            b1_colors,
            solver_styles,
            args.std,
            args.ci,
        )
        sampling_steps, eta = schedule
        ax.set_title(f"steps={sampling_steps}, eta={_format_eta(eta)}")

    for ax in axes_flat[n_schedules:]:
        ax.set_visible(False)

    for row_axes in axes:
        row_axes[0].set_ylabel("Mean KS")
        row_axes[0].set_yscale("log")

    if args.title:
        fig.suptitle(args.title)

    _make_legends(fig, b1_colors, rows, solver_styles)

    output_path = args.output or _default_output_path(args.csv_file)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    best_config_output_path = _best_config_output_path(output_path)
    best_config_title = (
        f"{args.title} - best adaptive config by B"
        if args.title
        else "Best adaptive config by B"
    )
    wrote_best_config = _plot_best_config(
        rows,
        best_config_output_path,
        args.std,
        args.ci,
        best_config_title,
    )

    print(f"Saved plot to {output_path}")
    if wrote_best_config:
        print(f"Saved best-config plot to {best_config_output_path}")
    print(f"Plotted {len(rows)} valid rows across {n_schedules} inference schedules")


if __name__ == "__main__":
    main()
