import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

PLOT_SPECS = {
    "optimal_b1": {
        "filename_suffix": "optimal_B1",
        "title": "Optimal B'",
    },
    "optimal_ni": {
        "filename_suffix": "optimal_Ni",
        "title": "Chosen N_i",
    },
    "percent_change": {
        "filename_suffix": "percent_change",
        "title": "Percent Change in KS",
    },
}

ADAPTIVE_LINESTYLES = ("-", "--", ":", "-.")
ADAPTIVE_MARKERS = ("o", "s", "D", "^", "v", "P")
SOLVER_LINESTYLES = ("--", "-.", ":", (0, (3, 1, 1, 1)))
SOLVER_MARKERS = ("^", "D", "x", "P", "v", "*", "h")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate summary plots from compare.py CSV output"
    )
    parser.add_argument("csv_file", type=str, help="Path to compare.py summary CSV")
    parser.add_argument(
        "--plots",
        type=str,
        default=",".join(PLOT_SPECS),
        help=(
            "Comma-separated plot names to generate "
            f"(available: {', '.join(PLOT_SPECS)})"
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title prefix",
    )
    return parser.parse_args()


def _parse_plots(raw: str):
    plot_names = [item.strip() for item in raw.split(",") if item.strip()]
    if not plot_names:
        raise ValueError("--plots must include at least one plot name")

    unknown = [name for name in plot_names if name not in PLOT_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown plot name(s): {', '.join(unknown)}. "
            f"Available plots: {', '.join(PLOT_SPECS)}"
        )
    return plot_names


def _output_stem(csv_path: str) -> str:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return stem or "stats"


def _default_output_path(csv_path: str, plot_name: str) -> str:
    directory = os.path.dirname(csv_path) or "."
    filename_suffix = PLOT_SPECS[plot_name]["filename_suffix"]
    return os.path.join(directory, f"{_output_stem(csv_path)}_{filename_suffix}.png")


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


def _parse_float_list(value: str):
    value = value.strip()
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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
            mode = raw_row.get("mode", "").strip()

            if mean_ks is None or B is None or sampling_steps is None or eta is None:
                continue

            if mode == "fixed_N":
                rows.append(
                    {
                        "B": B,
                        "B1": None,
                        "sampling_steps": sampling_steps,
                        "eta": eta,
                        "mean_ks": mean_ks,
                        "mode": mode,
                        "method_label": raw_row.get("method_label", "").strip(),
                        "solver": "",
                        "solver_sampling_steps": None,
                        "solver_eta": None,
                        "solver_nfe": None,
                        "sigma_estimation_mode": "",
                        "bias_type": "",
                        "reuse_phase1_samples": None,
                        "N_i": [],
                        "N_i_std": [],
                        "N_i_count": None,
                        "mode_key": None,
                    }
                )
                continue

            if mode == "solver_baseline":
                method_label = raw_row.get("method_label", "").strip()
                rows.append(
                    {
                        "B": B,
                        "B1": None,
                        "sampling_steps": sampling_steps,
                        "eta": eta,
                        "mean_ks": mean_ks,
                        "mode": mode,
                        "method_label": method_label,
                        "solver": raw_row.get("solver", "").strip(),
                        "solver_sampling_steps": _parse_int(
                            raw_row.get("solver_sampling_steps", "")
                        ),
                        "solver_eta": _parse_float(raw_row.get("solver_eta", "")),
                        "solver_nfe": _parse_int(raw_row.get("solver_nfe", "")),
                        "sigma_estimation_mode": "",
                        "bias_type": "",
                        "reuse_phase1_samples": None,
                        "N_i": [],
                        "N_i_std": [],
                        "N_i_count": None,
                        "mode_key": ("solver", method_label),
                    }
                )
                continue

            if mode != "estimate_and_sample":
                continue

            B1 = _parse_int(raw_row.get("B1", ""))
            sigma_estimation_mode = raw_row.get("sigma_estimation_mode", "").strip()
            bias_type = raw_row.get("bias_type", "").strip()
            reuse_raw = raw_row.get("reuse_phase1_samples")
            if reuse_raw is None:
                reuse_raw = raw_row.get("reuse_pilot_samples", "")
            reuse_phase1_samples = _parse_bool(reuse_raw or "")

            if B1 is None or not sigma_estimation_mode or reuse_phase1_samples is None:
                continue

            row = {
                "B": B,
                "B1": B1,
                "sampling_steps": sampling_steps,
                "eta": eta,
                "mean_ks": mean_ks,
                "mode": mode,
                "method_label": raw_row.get("method_label", "").strip(),
                "solver": "",
                "solver_sampling_steps": None,
                "solver_eta": None,
                "solver_nfe": None,
                "sigma_estimation_mode": sigma_estimation_mode,
                "bias_type": bias_type,
                "reuse_phase1_samples": reuse_phase1_samples,
                "N_i": _parse_float_list(raw_row.get("N_i", "")),
                "N_i_std": _parse_float_list(raw_row.get("N_i_std", "")),
                "N_i_count": _parse_int(raw_row.get("N_i_count", "")),
            }
            row["mode_key"] = _mode_key(row)
            rows.append(row)
    return rows


def _format_eta(eta: float) -> str:
    return f"{eta:g}"


def _format_schedule(schedule) -> str:
    sampling_steps, eta = schedule
    return f"steps={sampling_steps}, eta={_format_eta(eta)}"


def _schedule_key(row):
    return (row["sampling_steps"], row["eta"])


def _mode_key(row):
    return (
        row["sigma_estimation_mode"],
        row.get("bias_type", ""),
        row["reuse_phase1_samples"],
    )


def _adaptive_rows(rows):
    return [row for row in rows if row["mode"] == "estimate_and_sample"]


def _baseline_rows(rows):
    return [row for row in rows if row["mode"] == "fixed_N"]


def _solver_rows(rows):
    return [row for row in rows if row["mode"] == "solver_baseline"]


def _mode_title(mode_key):
    if len(mode_key) == 2:
        sigma_estimation_mode, reuse_phase1_samples = mode_key
        bias_type = ""
    else:
        sigma_estimation_mode, bias_type, reuse_phase1_samples = mode_key
    reuse_label = "reuse" if reuse_phase1_samples else "fresh"
    parts = [sigma_estimation_mode]
    if bias_type:
        parts.append(bias_type)
    parts.append(reuse_label)
    return " ".join(parts)


def _mode_keys(rows):
    return sorted({row["mode_key"] for row in _adaptive_rows(rows)})


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


def _format_solver_label(row):
    solver = row.get("solver")
    solver_sampling_steps = row.get("solver_sampling_steps")
    solver_eta = row.get("solver_eta")
    if solver and solver_sampling_steps is not None:
        solver_label = solver.upper() if solver == "ddim" else solver.replace("_", " ")
        label = f"{solver_label} {solver_sampling_steps}"
        if solver_eta is not None:
            label += f" eta={_format_eta(solver_eta)}"
        return label
    return row.get("method_label") or "solver baseline"


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
            _format_solver_label(solver_rows_by_label[method_label]),
        )
        for i, method_label in enumerate(labels)
    }


def _solver_style(method_label: str, solver_styles):
    return solver_styles.get(
        method_label,
        ("#7f7f7f", "--", "x", method_label or "solver baseline"),
    )


def _optimal_rows(rows):
    best_by_group = {}
    for row in rows:
        group_key = (_schedule_key(row), row["mode_key"], row["B"])
        best = best_by_group.get(group_key)
        if best is None or (row["mean_ks"], row["B1"]) < (
            best["mean_ks"],
            best["B1"],
        ):
            best_by_group[group_key] = row
    return best_by_group


def _selected_rows_by_schedule(selected_rows):
    grouped = defaultdict(list)
    for (schedule, _, _), row in selected_rows.items():
        grouped[schedule].append(row)
    return grouped


def _selected_rows_by_schedule_and_budget(selected_rows):
    grouped = defaultdict(list)
    for (schedule, _, B), row in selected_rows.items():
        grouped[(schedule, B)].append(row)
    return grouped


def _best_adaptive_by_schedule_and_budget(rows):
    best_by_group = {}
    for row in rows:
        group_key = (_schedule_key(row), row["B"])
        best = best_by_group.get(group_key)
        if best is None or (row["mean_ks"], row["B1"]) < (best["mean_ks"], best["B1"]):
            best_by_group[group_key] = row
    return best_by_group


def _baseline_by_schedule_and_budget(rows):
    baseline_by_group = {}
    for row in rows:
        group_key = (_schedule_key(row), row["B"])
        best = baseline_by_group.get(group_key)
        if best is None or row["mean_ks"] < best["mean_ks"]:
            baseline_by_group[group_key] = row
    return baseline_by_group


def _solver_by_schedule_budget_and_label(rows):
    solver_by_group = {}
    for row in rows:
        group_key = (_schedule_key(row), row["B"], row["method_label"])
        best = solver_by_group.get(group_key)
        if best is None or row["mean_ks"] < best["mean_ks"]:
            solver_by_group[group_key] = row
    return solver_by_group


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


def _make_legend(fig, mode_keys, mode_styles):
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
        points = [row for row in selected_rows if row["mode_key"] == mode_key]
        points.sort(key=lambda row: row["B"])
        if not points:
            continue

        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        x_values = [row["B"] for row in points]
        y_values = [row["B1"] for row in points]
        ax.plot(
            x_values,
            y_values,
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


def _plot_optimal_b1(rows, title_prefix: str | None, mode_styles, solver_styles):
    rows = _adaptive_rows(rows)
    mode_keys = _mode_keys(rows)
    schedules = sorted({_schedule_key(row) for row in rows})
    selected = _optimal_rows(rows)
    selected_by_schedule = _selected_rows_by_schedule(selected)

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
            ax,
            selected_by_schedule.get(schedule, []),
            mode_keys,
            mode_styles,
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

    _make_legend(fig, mode_keys, mode_styles)
    return fig


def _plot_optimal_ni_panel(ax, selected_rows, max_levels: int, mode_keys, mode_styles):
    plotted_any = False
    for mode_key in mode_keys:
        matching_rows = [row for row in selected_rows if row["mode_key"] == mode_key]
        if not matching_rows:
            continue

        row = min(matching_rows, key=lambda item: (item["mean_ks"], item["B1"]))
        if not row["N_i"]:
            continue

        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        x_values = list(range(1, len(row["N_i"]) + 1))
        ni_std = row.get("N_i_std") or []
        if len(ni_std) == len(row["N_i"]):
            lower = [max(0.0, mean - std) for mean, std in zip(row["N_i"], ni_std)]
            upper = [mean + std for mean, std in zip(row["N_i"], ni_std)]
            ax.fill_between(
                x_values,
                lower,
                upper,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
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


def _plot_optimal_ni(rows, title_prefix: str | None, mode_styles, solver_styles):
    rows = _adaptive_rows(rows)
    mode_keys = _mode_keys(rows)
    schedules = sorted({_schedule_key(row) for row in rows})
    budgets = sorted({row["B"] for row in rows})
    selected = _optimal_rows(rows)
    selected_by_schedule_and_budget = _selected_rows_by_schedule_and_budget(selected)
    max_levels = max((len(row["N_i"]) for row in selected.values()), default=0)

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
            _plot_optimal_ni_panel(
                ax,
                panel_rows,
                max_levels,
                mode_keys,
                mode_styles,
            )

            if row_idx == 0:
                ax.set_title(f"B={budget}")
            if col_idx == 0:
                ax.set_ylabel(f"Chosen N_i\n{_format_schedule(schedule)}")
            if row_idx == len(schedules) - 1:
                ax.set_xlabel("Split level i")

    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS['optimal_ni']['title']}")

    _make_legend(fig, mode_keys, mode_styles)
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
        percent_change = (
            100.0 * (baseline["mean_ks"] - candidate["mean_ks"]) / baseline["mean_ks"]
        )
        y_values.append(percent_change)
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
            for group_schedule, _, method_label in solver_by_group
            if group_schedule == schedule
        }
    )
    for method_label in solver_labels:
        solver_candidates = {
            B: row
            for (group_schedule, B, group_method_label), row in solver_by_group.items()
            if group_schedule == schedule and group_method_label == method_label
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


def _plot_percent_change(rows, title_prefix: str | None, mode_styles, solver_styles):
    schedules = sorted({_schedule_key(row) for row in rows})
    baseline_by_group = _baseline_by_schedule_and_budget(_baseline_rows(rows))
    best_adaptive_by_group = _best_adaptive_by_schedule_and_budget(_adaptive_rows(rows))
    solver_by_group = _solver_by_schedule_budget_and_label(_solver_rows(rows))

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
        raise ValueError("No valid rows found in CSV after filtering")
    mode_styles = _build_mode_style_map(rows)
    solver_styles = _build_solver_style_map(rows)

    plotters = {
        "optimal_b1": _plot_optimal_b1,
        "optimal_ni": _plot_optimal_ni,
        "percent_change": _plot_percent_change,
    }

    output_paths = []
    for plot_name in plot_names:
        fig = plotters[plot_name](rows, args.title, mode_styles, solver_styles)
        output_path = _default_output_path(args.csv_file, plot_name)
        _save_figure(fig, output_path, has_title=bool(args.title))
        output_paths.append(output_path)

    print(f"Loaded {len(rows)} valid rows")
    for output_path in output_paths:
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
