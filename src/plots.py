import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

SOLVER_LINESTYLES = ("--", "-.", ":", (0, (3, 1, 1, 1)))
SOLVER_MARKERS = ("^", "D", "x", "P", "v", "*", "h")
ADAPTIVE_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "h")
LAYOUT_PADS = {"w_pad": 0.16, "h_pad": 0.16, "wspace": 0.08, "hspace": 0.09}

CSV_FIELDS = {
    "mode",
    "sampler",
    "sampling_steps",
    "sampling_params",
    "step_schedule",
    "B",
    "B1",
    "sigma_mode",
    "reuse",
    "free_B1",
    "optimizer",
    "solver",
    "solver_steps",
    "solver_params",
    "nfe_per_sample",
    "mean_ks",
    "std_ks",
    "n_valid_runs",
    "N_i",
    "N_i_std",
}

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


@dataclass(frozen=True)
class Row:
    mode: str
    sampler: str
    sampling_steps: int | None
    sampling_params: dict[str, Any]
    step_schedule: str
    B: int
    B1: int | None
    sigma_mode: str
    reuse: bool | None
    free_B1: bool
    optimizer: str
    solver: str
    solver_steps: int | None
    solver_params: dict[str, Any]
    nfe_per_sample: float | None
    mean_ks: float
    std_ks: float | None
    n_valid_runs: int | None
    N_i: list[float]
    N_i_std: list[float]


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
    intervals = parser.add_mutually_exclusive_group()
    intervals.add_argument(
        "--std",
        action="store_true",
        help="Show mean_ks +/- std_ks intervals on main / best_config plots",
    )
    intervals.add_argument(
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


def _parse_json_dict(value: str):
    value = (value or "").strip()
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {value!r}")
    return parsed


def _load_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = sorted((CSV_FIELDS - {"optimizer"}) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                "CSV is missing structured field(s): "
                f"{', '.join(missing)}; rerun compare.py"
            )
        for raw_row in reader:
            if raw_row.get("error", "").strip():
                continue
            mean_ks = _parse_float(raw_row.get("mean_ks", ""))
            B = _parse_int(raw_row.get("B", ""))
            if mean_ks is None or B is None:
                continue
            mode = (raw_row.get("mode") or "").strip()
            if mode == "estimate_and_sample":
                mode = "adaptive"
            if mode not in {"fixed_N", "solver_baseline", "adaptive"}:
                continue

            reuse = _parse_bool(raw_row.get("reuse", ""))
            free_b1 = _parse_bool(raw_row.get("free_B1", ""))
            optimizer = (raw_row.get("optimizer") or "").strip()
            if mode == "adaptive" and not optimizer:
                optimizer = "monotone"
            rows.append(
                Row(
                    mode=mode,
                    sampler=(raw_row.get("sampler") or "").strip(),
                    sampling_steps=_parse_int(raw_row.get("sampling_steps", "")),
                    sampling_params=_parse_json_dict(raw_row.get("sampling_params", "")),
                    step_schedule=(raw_row.get("step_schedule") or "default").strip(),
                    B=B,
                    B1=_parse_int(raw_row.get("B1", "")),
                    sigma_mode=(raw_row.get("sigma_mode") or "").strip(),
                    reuse=reuse,
                    free_B1=bool(free_b1),
                    optimizer=optimizer,
                    solver=(raw_row.get("solver") or "").strip(),
                    solver_steps=_parse_int(raw_row.get("solver_steps", "")),
                    solver_params=_parse_json_dict(raw_row.get("solver_params", "")),
                    nfe_per_sample=_parse_float(raw_row.get("nfe_per_sample", "")),
                    mean_ks=mean_ks,
                    std_ks=_parse_float(raw_row.get("std_ks", "")),
                    n_valid_runs=_parse_int(raw_row.get("n_valid_runs", "")),
                    N_i=_parse_float_list(raw_row.get("N_i", "")),
                    N_i_std=_parse_float_list(raw_row.get("N_i_std", "")),
                )
            )
    return rows


def _json_key(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, list):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _format_count(value) -> str:
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return str(value)


def _params_label(params: dict[str, Any]) -> str:
    if not params:
        return ""
    return ", ".join(f"{key}={_format_value(params[key])}" for key in sorted(params))


def _compact_params_label(params: dict[str, Any]) -> str:
    if not params:
        return ""
    sampler_params = params.get("sampler_params")
    if isinstance(sampler_params, dict):
        for key in ("stochastic_churn_rate", "S_churn"):
            if key in sampler_params:
                return f"churn={_format_value(sampler_params[key])}"
    if "eta" in params:
        return f"eta={_format_value(params['eta'])}"
    key = sorted(params)[0]
    return f"{key}={_format_value(params[key])}"


def _pretty_name(name: str) -> str:
    return name.upper() if name == "ddim" else str(name).replace("_", " ")


def _short_sigma_mode(name: str) -> str:
    if name == "pilot_tree":
        return "pilot"
    if name == "independent":
        return "ind"
    return str(name).replace("_", " ")


def _sampling_key(row: Row):
    return (row.sampler, _json_key(row.sampling_params))


def _schedule_key(row: Row):
    return (_sampling_key(row), row.step_schedule)


def _solver_family_key(row: Row):
    return (row.solver, _json_key(row.solver_params))


def _format_sampling_config_key(key) -> str:
    sampler, params_json = key
    params = json.loads(params_json) if params_json else {}
    parts = [_pretty_name(sampler) if sampler else "sampling config"]
    params_text = _params_label(params)
    if params_text:
        parts.append(params_text)
    return ", ".join(parts)


def _format_sampling_config_short(key) -> str:
    sampler, params_json = key
    label = _pretty_name(sampler) if sampler else "sampling config"
    params = json.loads(params_json) if params_json else {}
    params_text = _compact_params_label(params)
    if params_text:
        label += f", {params_text}"
    return label


def _format_schedule_key(key) -> str:
    sampling_key, step_schedule = key
    config_label = _format_sampling_config_key(sampling_key)
    if step_schedule:
        return f"{config_label}, {step_schedule}"
    return config_label


def _format_schedule_short(key) -> str:
    sampling_key, step_schedule = key
    config_label = _format_sampling_config_short(sampling_key)
    if step_schedule:
        return f"{config_label}, {step_schedule}"
    return config_label


def _method_key(row: Row):
    if row.mode == "fixed_N":
        return ("fixed_N",)
    if row.mode == "solver_baseline":
        return ("solver", *_solver_family_key(row))
    return (
        "adaptive",
        row.sigma_mode,
        bool(row.reuse),
        bool(row.free_B1),
        row.optimizer or "monotone",
    )


def _method_display_label(key) -> str:
    if key[0] == "fixed_N":
        return "fixed"
    if key[0] == "solver":
        _, solver, params_json = key
        params = json.loads(params_json) if params_json else {}
        label = _pretty_name(solver)
        params_text = _compact_params_label(params)
        if params_text:
            label += f" {params_text}"
        return label
    return _mode_title(key[1:])


def _row_stable_label(row: Row) -> str:
    return "|".join(
        [
            row.mode,
            _method_display_label(_method_key(row)),
            _format_schedule_key(_schedule_key(row)) if row.mode != "solver_baseline" else row.step_schedule,
            str(row.B1 if row.B1 is not None else ""),
        ]
    )


def _adaptive_rows(rows):
    return [row for row in rows if row.mode == "adaptive"]


def _baseline_rows(rows):
    return [row for row in rows if row.mode == "fixed_N"]


def _all_baseline_rows(rows):
    return [row for row in rows if row.mode in {"fixed_N", "solver_baseline"}]


def _solver_rows(rows):
    return [row for row in rows if row.mode == "solver_baseline"]


def _mode_key(row: Row):
    return (
        row.sigma_mode,
        bool(row.reuse),
        bool(row.free_B1),
        row.optimizer or "monotone",
    )


def _mode_keys(rows):
    return sorted({_mode_key(row) for row in _adaptive_rows(rows)})


def _mode_title(mode_key):
    sigma_mode, reuse, free_b1, optimizer = mode_key
    label = f"{_short_sigma_mode(sigma_mode)} {'reuse' if reuse else 'fresh'}"
    if free_b1:
        label += " free"
    if optimizer:
        label += f" {optimizer}"
    return label


def _baseline_series_key(row: Row):
    if row.mode == "fixed_N":
        return ("fixed_N", _schedule_key(row))
    if row.mode == "solver_baseline":
        return ("solver", _solver_family_key(row), row.step_schedule)
    raise ValueError(f"Not a baseline row: {row.mode!r}")


def _baseline_method_key(series_key):
    if series_key[0] == "fixed_N":
        return ("fixed_N",)
    if series_key[0] == "solver":
        return ("solver", *series_key[1])
    raise ValueError(f"Unknown baseline series key: {series_key!r}")


def _baseline_series_label(series_key) -> str:
    if series_key[0] == "fixed_N":
        return f"fixed, {_format_schedule_short(series_key[1])}"
    if series_key[0] == "solver":
        label = _method_display_label(("solver", *series_key[1]))
        if series_key[2]:
            label += f", {series_key[2]}"
        return label
    raise ValueError(f"Unknown baseline series key: {series_key!r}")


def _main_series_key(row: Row):
    if row.mode == "adaptive":
        return ("adaptive", _method_key(row), row.B1)
    return _baseline_series_key(row)


def best_row(rows):
    candidates = list(rows)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            row.mean_ks,
            row.B1 if row.B1 is not None else -1,
            _row_stable_label(row),
        ),
    )


def _best_by_group(rows, *, key_fn, score_fn=None):
    out = {}
    for row in rows:
        key = key_fn(row)
        if score_fn is None:
            score = (
                row.mean_ks,
                row.B1 if row.B1 is not None else -1,
                _row_stable_label(row),
            )
        else:
            score = score_fn(row)
        if key not in out or score < out[key][0]:
            out[key] = (score, row)
    return {key: row for key, (_, row) in out.items()}


def _build_method_style_map(rows):
    keys = sorted({_method_key(row) for row in rows})
    cmap = plt.get_cmap("tab10") if len(keys) <= 10 else plt.get_cmap("tab20")
    return {key: cmap(i % cmap.N) for i, key in enumerate(keys)}


def _build_b1_marker_map(rows):
    b1_values = sorted({row.B1 for row in _adaptive_rows(rows) if row.B1 is not None})
    return {
        b1: ADAPTIVE_MARKERS[i % len(ADAPTIVE_MARKERS)]
        for i, b1 in enumerate(b1_values)
    }


def _build_mode_style_map(rows):
    mode_keys = _mode_keys(rows)
    cmap = plt.get_cmap("tab10") if len(mode_keys) <= 10 else plt.get_cmap("tab20")
    return {
        mode_key: (
            cmap(i % cmap.N),
            "--" if mode_key[0] == "independent" else ":" if mode_key[0] == "pilot_tree" else "-",
            ADAPTIVE_MARKERS[i % len(ADAPTIVE_MARKERS)],
        )
        for i, mode_key in enumerate(mode_keys)
    }


def _mode_style(mode_key, mode_styles):
    return mode_styles.get(mode_key, ("#7f7f7f", "-", "o"))


def _build_solver_style_map(rows):
    solver_keys = sorted({_solver_family_key(row) for row in _solver_rows(rows)})
    cmap = plt.get_cmap("tab10") if len(solver_keys) <= 10 else plt.get_cmap("tab20")
    return {
        key: (
            cmap(i % cmap.N),
            SOLVER_LINESTYLES[i % len(SOLVER_LINESTYLES)],
            SOLVER_MARKERS[i % len(SOLVER_MARKERS)],
            _method_display_label(("solver", *key)),
        )
        for i, key in enumerate(solver_keys)
    }


def _solver_style(solver_key, solver_styles):
    return solver_styles.get(solver_key, ("#7f7f7f", "--", "x", "solver baseline"))


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
    return _ci_multiplier(ci_level, n_valid_runs - 1) * std_ks / math.sqrt(n_valid_runs)


def _fill_interval(ax, x_values, y_values, half_widths, color, zorder):
    if any(width is None for width in half_widths):
        return
    lower = [max(y - w, 1e-12) for y, w in zip(y_values, half_widths)]
    upper = [y + w for y, w in zip(y_values, half_widths)]
    ax.fill_between(
        x_values, lower, upper, color=color, alpha=0.16, linewidth=0, zorder=zorder
    )


def _make_subplot_grid(count: int, max_cols: int = 2):
    ncols = min(max_cols, max(1, count))
    nrows = math.ceil(max(1, count) / ncols)
    return nrows, ncols


def _legend_rows(label_count: int, ncol: int) -> int:
    return math.ceil(int(label_count) / max(1, int(ncol))) if label_count else 0


def _with_top_legend_height(figsize, label_count: int, ncol: int):
    width, height = figsize
    return width, height + 0.45 * _legend_rows(label_count, ncol)


def _main_legend_count(rows, b1_markers) -> int:
    return (
        len({_baseline_series_key(row) for row in _all_baseline_rows(rows)})
        + len({_method_key(row) for row in _adaptive_rows(rows)})
        + len(b1_markers)
    )


def _apply_scalar_formatters(ax, format_x: bool = True, format_y: bool = True):
    if format_x:
        x_formatter = ScalarFormatter(useMathText=True)
        x_formatter.set_powerlimits((-2, 3))
        ax.xaxis.set_major_formatter(x_formatter)
    if format_y:
        y_formatter = ScalarFormatter(useMathText=True)
        y_formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(y_formatter)


def _series_points(rows):
    return sorted(rows, key=lambda row: row.B)


def _plot_series(ax, rows, *, color, linestyle, marker, label=None, show_std=False, ci_level=None, zorder=3):
    points = _series_points(rows)
    if not points:
        return False
    x_values = [row.B for row in points]
    y_values = [row.mean_ks for row in points]
    if show_std:
        _fill_interval(
            ax,
            x_values,
            y_values,
            [row.std_ks for row in points],
            color,
            zorder - 1,
        )
    if ci_level is not None:
        _fill_interval(
            ax,
            x_values,
            y_values,
            [_ci_half_width(row.std_ks, row.n_valid_runs, ci_level) for row in points],
            color,
            zorder - 1,
        )
    ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=2.0,
        markersize=5.5,
        label=label,
        zorder=zorder,
    )
    return True


def _main_facet_keys(rows):
    non_solver_keys = sorted({_schedule_key(row) for row in rows if row.mode != "solver_baseline"})
    if non_solver_keys:
        return non_solver_keys
    return [(("", "{}"), row.step_schedule) for row in _solver_rows(rows)]


def _plot_main_panel(ax, rows, facet_key, method_styles, b1_markers, show_std, ci_level):
    panel_rows = [
        row
        for row in rows
        if (row.mode in {"adaptive", "fixed_N"} and _schedule_key(row) == facet_key)
        or row.mode == "solver_baseline"
    ]
    grouped = defaultdict(list)
    for row in panel_rows:
        series_key = _main_series_key(row)
        grouped[series_key].append(row)

    plotted_any = False
    for series_key, series_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        kind = series_key[0]
        method_key = series_key[1] if kind == "adaptive" else _baseline_method_key(series_key)
        color = method_styles.get(method_key, "#1f77b4")
        linestyle = "-"
        marker = "o"
        zorder = 3
        if kind == "solver":
            family = series_key[1]
            solver_idx = sorted({key[1:] for key in method_styles if key[0] == "solver"}).index(family)
            linestyle = SOLVER_LINESTYLES[solver_idx % len(SOLVER_LINESTYLES)]
            marker = SOLVER_MARKERS[solver_idx % len(SOLVER_MARKERS)]
            zorder = 5
        elif kind == "adaptive":
            b1 = series_key[2]
            marker = b1_markers.get(b1, "o")
            linestyle = "--" if method_key[1] == "independent" else ":" if method_key[1] == "pilot_tree" else "-"
            zorder = 2
        else:
            zorder = 4
        plotted_any |= _plot_series(
            ax,
            series_rows,
            color=color,
            linestyle=linestyle,
            marker=marker,
            show_std=show_std,
            ci_level=ci_level,
            zorder=zorder,
        )

    if not plotted_any:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total B")
    _apply_scalar_formatters(ax, format_x=True, format_y=True)


def _make_main_legend(fig, rows, method_styles, b1_markers):
    handles = []
    labels = []
    solver_families = sorted({key[1:] for key in method_styles if key[0] == "solver"})

    baseline_series_keys = sorted(
        {_baseline_series_key(row) for row in _all_baseline_rows(rows)}, key=str
    )
    for series_key in baseline_series_keys:
        method_key = _baseline_method_key(series_key)
        color = method_styles.get(method_key, "#1f77b4")
        marker = "o"
        linestyle = "-"
        if series_key[0] == "solver":
            family = series_key[1]
            solver_idx = solver_families.index(family) if family in solver_families else 0
            marker = SOLVER_MARKERS[solver_idx % len(SOLVER_MARKERS)]
            linestyle = SOLVER_LINESTYLES[solver_idx % len(SOLVER_LINESTYLES)]
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle, marker=marker, linewidth=2.0))
        labels.append(_baseline_series_label(series_key))

    adaptive_method_keys = sorted({_method_key(row) for row in _adaptive_rows(rows)}, key=str)
    for method_key in adaptive_method_keys:
        color = method_styles.get(method_key, "#1f77b4")
        linestyle = "--" if method_key[1] == "independent" else ":" if method_key[1] == "pilot_tree" else "-"
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle, marker="o", linewidth=2.0))
        labels.append(_method_display_label(method_key))

    for b1, marker in sorted(b1_markers.items()):
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                linestyle="None",
                marker=marker,
                markersize=6,
            )
        )
        labels.append(f"B1={_format_count(b1)}")
    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=min(4, max(1, len(labels))),
            frameon=False,
            borderaxespad=1.0,
            columnspacing=1.4,
            labelspacing=0.8,
        )


def _plot_main(rows, args):
    facet_keys = _main_facet_keys(rows)
    sampling_keys = sorted({key[0] for key in facet_keys})
    schedules = sorted({key[1] for key in facet_keys})
    method_styles = _build_method_style_map(rows)
    b1_markers = _build_b1_marker_map(rows)
    legend_count = _main_legend_count(rows, b1_markers)
    legend_ncol = min(4, max(1, legend_count))

    if len(sampling_keys) > 1 and len(schedules) > 1:
        figsize = _with_top_legend_height(
            (6.2 * len(schedules), 4.6 * len(sampling_keys)),
            legend_count,
            legend_ncol,
        )
        fig, axes = plt.subplots(
            len(sampling_keys),
            len(schedules),
            figsize=figsize,
            squeeze=False,
            sharey=True,
            constrained_layout=True,
        )
        for row_idx, sampling_key in enumerate(sampling_keys):
            for col_idx, schedule in enumerate(schedules):
                ax = axes[row_idx][col_idx]
                facet_key = (sampling_key, schedule)
                if facet_key not in facet_keys:
                    ax.set_visible(False)
                    continue
                _plot_main_panel(ax, rows, facet_key, method_styles, b1_markers, args.std, args.ci)
                if row_idx == 0:
                    ax.set_title(schedule)
                if col_idx == 0:
                    ax.set_ylabel(f"Mean KS\n{_format_sampling_config_short(sampling_key)}")
    else:
        nrows, ncols = _make_subplot_grid(len(facet_keys), max_cols=2)
        figsize = _with_top_legend_height(
            (7.0 * ncols, 4.8 * nrows),
            legend_count,
            legend_ncol,
        )
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            sharey=True,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()
        for ax, facet_key in zip(axes_flat, facet_keys):
            _plot_main_panel(ax, rows, facet_key, method_styles, b1_markers, args.std, args.ci)
            ax.set_title(_format_schedule_short(facet_key))
            ax.set_ylabel("Mean KS")
        for ax in axes_flat[len(facet_keys) :]:
            ax.set_visible(False)

    if args.title:
        fig.suptitle(args.title)
    _make_main_legend(fig, rows, method_styles, b1_markers)
    return fig


def _best_adaptive_rows(rows, *, include_free=False):
    candidates = [row for row in _adaptive_rows(rows) if include_free or not row.free_B1]
    selected = _best_by_group(candidates, key_fn=lambda row: (_schedule_key(row), row.B))
    return sorted(selected.values(), key=lambda row: (_format_schedule_key(_schedule_key(row)), row.B))


def _best_solver_rows(rows):
    selected = _best_by_group(
        _solver_rows(rows),
        key_fn=lambda row: (_solver_family_key(row), row.step_schedule, row.B),
    )
    return sorted(selected.values(), key=lambda row: (_solver_family_key(row), row.step_schedule, row.B))


def _best_config_series_key(row: Row):
    if row.mode == "fixed_N":
        return ("fixed_N", _schedule_key(row))
    if row.mode == "solver_baseline":
        return ("solver", _solver_family_key(row), row.step_schedule)
    return ("best_adaptive", _schedule_key(row))


def _format_best_config_label(series_key):
    kind = series_key[0]
    if kind == "fixed_N":
        return f"fixed, {_format_schedule_short(series_key[1])}"
    if kind == "solver":
        return f"{_method_display_label(('solver', *series_key[1]))}, {series_key[2]}"
    return f"best, {_format_schedule_short(series_key[1])}"


def _best_adaptive_improvement_points(rows):
    best_adaptive = _best_by_group(
        (row for row in _adaptive_rows(rows) if not row.free_B1),
        key_fn=lambda row: row.B,
    )
    best_baseline = _best_by_group(
        (row for row in rows if row.mode in {"fixed_N", "solver_baseline"}),
        key_fn=lambda row: row.B,
    )
    x_values, y_values = [], []
    for budget in sorted(set(best_adaptive) & set(best_baseline)):
        baseline = best_baseline[budget]
        adaptive = best_adaptive[budget]
        if baseline.mean_ks == 0:
            continue
        x_values.append(budget)
        y_values.append(100.0 * (baseline.mean_ks - adaptive.mean_ks) / baseline.mean_ks)
    return x_values, y_values


def _plot_best_config(rows, show_std, ci_level, title=None):
    selected_rows = []
    selected_rows.extend(_baseline_rows(rows))
    selected_rows.extend(_best_solver_rows(rows))
    selected_rows.extend(_best_adaptive_rows(rows))
    if not selected_rows:
        return None

    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[_best_config_series_key(row)].append(row)
    method_styles = _build_method_style_map(selected_rows)
    solver_styles = _build_solver_style_map(selected_rows)
    figsize = (13.0 + 0.25 * min(len(grouped), 8), 10.0)

    fig, (ax, improvement_ax) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        constrained_layout=True,
    )
    for series_key, series_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        kind = series_key[0]
        if kind == "solver":
            color, linestyle, marker, _ = _solver_style(series_key[1], solver_styles)
        elif kind == "best_adaptive":
            color = method_styles.get(_method_key(series_rows[0]), "#2ca02c")
            linestyle, marker = ":", "*"
        else:
            color = method_styles.get(("fixed_N",), "#1f77b4")
            linestyle, marker = "-", "o"
        _plot_series(
            ax,
            series_rows,
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=_format_best_config_label(series_key),
            show_std=show_std,
            ci_level=ci_level,
        )

    ax.grid(alpha=0.25)
    ax.set_ylabel("Mean KS")
    ax.set_yscale("log")
    ax.set_title(title or PLOT_SPECS["best_config"]["title"])
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        borderaxespad=0.8,
        labelspacing=0.6,
    )
    _apply_scalar_formatters(ax, format_x=True, format_y=True)

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
            label="adaptive vs baseline",
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
    _apply_scalar_formatters(improvement_ax, format_x=True, format_y=True)
    return fig


def _make_mode_legend(fig, mode_keys, mode_styles):
    handles = []
    labels = []
    for mode_key in mode_keys:
        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        handles.append(
            Line2D([0], [0], color=color, linestyle=linestyle, marker=marker, linewidth=1.8, markersize=5)
        )
        labels.append(_mode_title(mode_key))
    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=min(3, max(1, len(labels))),
            frameon=False,
            borderaxespad=1.0,
            columnspacing=1.4,
            labelspacing=0.8,
        )


def _plot_optimal_b1_panel(ax, selected_rows, mode_keys, mode_styles):
    plotted_any = False
    for mode_key in mode_keys:
        points = [row for row in selected_rows if _mode_key(row) == mode_key]
        points.sort(key=lambda row: row.B)
        if not points:
            continue
        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        ax.plot(
            [row.B for row in points],
            [row.B1 for row in points],
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
    _apply_scalar_formatters(ax, format_x=True, format_y=True)


def _plot_optimal_b1(rows, title_prefix, mode_styles):
    rows = _adaptive_rows(rows)
    mode_keys = _mode_keys(rows)
    schedules = sorted({_schedule_key(row) for row in rows}, key=_format_schedule_key)
    selected = _best_by_group(
        rows,
        key_fn=lambda row: (_schedule_key(row), _mode_key(row), row.B),
    )
    selected_by_schedule = defaultdict(list)
    for (schedule, _, _), row in selected.items():
        selected_by_schedule[schedule].append(row)

    nrows, ncols = _make_subplot_grid(len(schedules), max_cols=2)
    legend_count = len(mode_keys)
    legend_ncol = min(3, max(1, legend_count))
    figsize = _with_top_legend_height(
        (7.2 * ncols, 4.8 * nrows),
        legend_count,
        legend_ncol,
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()
    for ax, schedule in zip(axes_flat, schedules):
        _plot_optimal_b1_panel(ax, selected_by_schedule.get(schedule, []), mode_keys, mode_styles)
        ax.set_title(_format_schedule_short(schedule))
        ax.set_xlabel("B")
    for ax in axes_flat[len(schedules) :]:
        ax.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("Optimal B'")
    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS['optimal_b1']['title']}")
    _make_mode_legend(fig, mode_keys, mode_styles)
    return fig


def _plot_optimal_ni_panel(
    ax, selected_rows, max_levels, mode_keys, mode_styles, ci_level=None
):
    plotted_any = False
    for mode_key in mode_keys:
        row = best_row(row for row in selected_rows if _mode_key(row) == mode_key)
        if row is None or not row.N_i:
            continue
        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        x_values = list(range(1, len(row.N_i) + 1))
        if ci_level is not None and len(row.N_i_std) == len(row.N_i):
            _fill_interval(
                ax,
                x_values,
                row.N_i,
                [
                    _ci_half_width(std, row.n_valid_runs, ci_level)
                    for std in row.N_i_std
                ],
                color,
                1,
            )
        ax.plot(
            x_values,
            row.N_i,
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


def _plot_optimal_ni(rows, title_prefix, mode_styles, ci_level=None):
    rows = _adaptive_rows(rows)
    mode_keys = _mode_keys(rows)
    schedules = sorted({_schedule_key(row) for row in rows}, key=_format_schedule_key)
    budgets = sorted({row.B for row in rows})
    selected = _best_by_group(
        rows,
        key_fn=lambda row: (_schedule_key(row), _mode_key(row), row.B),
    )
    selected_by_schedule_and_budget = defaultdict(list)
    for (schedule, _, budget), row in selected.items():
        selected_by_schedule_and_budget[(schedule, budget)].append(row)
    max_levels = max((len(row.N_i) for row in selected.values()), default=0)
    legend_count = len(mode_keys)
    legend_ncol = min(3, max(1, legend_count))
    figsize = _with_top_legend_height(
        (4.8 * max(1, len(budgets)), 4.0 * max(1, len(schedules))),
        legend_count,
        legend_ncol,
    )

    fig, axes = plt.subplots(
        max(1, len(schedules)),
        max(1, len(budgets)),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, schedule in enumerate(schedules):
        for col_idx, budget in enumerate(budgets):
            ax = axes[row_idx][col_idx]
            panel_rows = selected_by_schedule_and_budget.get((schedule, budget), [])
            _plot_optimal_ni_panel(
                ax, panel_rows, max_levels, mode_keys, mode_styles, ci_level
            )
            if row_idx == 0:
                ax.set_title(f"B={_format_count(budget)}")
            if col_idx == 0:
                ax.set_ylabel(f"Chosen N_i\n{_format_schedule_short(schedule)}")
            if row_idx == len(schedules) - 1:
                ax.set_xlabel("Split level i")
    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS['optimal_ni']['title']}")
    _make_mode_legend(fig, mode_keys, mode_styles)
    return fig


def _percent_change_points(baseline_by_budget, candidate_by_budget):
    x_values = []
    y_values = []
    for budget in sorted(set(baseline_by_budget) & set(candidate_by_budget)):
        baseline = baseline_by_budget[budget]
        candidate = candidate_by_budget[budget]
        if baseline.mean_ks == 0:
            continue
        x_values.append(budget)
        y_values.append(100.0 * (baseline.mean_ks - candidate.mean_ks) / baseline.mean_ks)
    return x_values, y_values


def _plot_percent_line(ax, baseline_by_budget, candidate_by_budget, *, color, linestyle, marker, label, annotation_dy=None):
    x_values, y_values = _percent_change_points(baseline_by_budget, candidate_by_budget)
    if not x_values:
        return False
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
    if annotation_dy is not None:
        for x_value, y_value in zip(x_values, y_values):
            ax.annotate(
                f"{y_value:.1f}%",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, annotation_dy),
                ha="center",
                fontsize=8,
                color=color,
            )
    return True


def _plot_percent_change_panel(
    ax,
    schedule,
    baseline_by_group,
    adaptive_by_group,
    solver_by_group,
    solver_styles,
    mode_styles,
):
    baseline_by_budget = {
        budget: row
        for (group_schedule, budget), row in baseline_by_group.items()
        if group_schedule == schedule
    }
    plotted_any = False

    adaptive_mode_keys = sorted(
        {
            mode_key
            for (group_schedule, mode_key, _) in adaptive_by_group
            if group_schedule == schedule
        }
    )
    for mode_key in adaptive_mode_keys:
        adaptive_by_budget = {
            budget: row
            for (group_schedule, group_mode_key, budget), row in adaptive_by_group.items()
            if group_schedule == schedule and group_mode_key == mode_key
        }
        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        plotted_any |= _plot_percent_line(
            ax,
            baseline_by_budget,
            adaptive_by_budget,
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=_mode_title(mode_key),
        )

    solver_series_keys = sorted(
        {(group_step_schedule, solver_key) for (group_step_schedule, _, solver_key) in solver_by_group}
    )
    for solver_step_schedule, solver_key in solver_series_keys:
        solver_by_budget = {
            budget: row
            for (row_step_schedule, budget, group_solver_key), row in solver_by_group.items()
            if row_step_schedule == solver_step_schedule
            and group_solver_key == solver_key
        }
        color, linestyle, marker, label = _solver_style(solver_key, solver_styles)
        if solver_step_schedule:
            label = f"{label}, {solver_step_schedule}"
        plotted_any |= _plot_percent_line(
            ax,
            baseline_by_budget,
            solver_by_budget,
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=label,
        )

    if plotted_any:
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.legend(frameon=False, fontsize=8, borderaxespad=0.6, labelspacing=0.5)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)
    ax.set_xscale("log")
    _apply_scalar_formatters(ax, format_x=True, format_y=True)


def _plot_percent_change(rows, title_prefix, solver_styles, mode_styles):
    schedules = sorted({_schedule_key(row) for row in rows if row.mode != "solver_baseline"}, key=_format_schedule_key)
    if not schedules:
        return plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    baseline_by_group = _best_by_group(
        _baseline_rows(rows), key_fn=lambda row: (_schedule_key(row), row.B)
    )
    adaptive_by_group = _best_by_group(
        _adaptive_rows(rows),
        key_fn=lambda row: (_schedule_key(row), _mode_key(row), row.B),
    )
    solver_by_group = _best_by_group(
        _solver_rows(rows),
        key_fn=lambda row: (row.step_schedule, row.B, _solver_family_key(row)),
    )

    nrows, ncols = _make_subplot_grid(len(schedules), max_cols=2)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 4.8 * nrows),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()
    for ax, schedule in zip(axes_flat, schedules):
        _plot_percent_change_panel(
            ax,
            schedule,
            baseline_by_group,
            adaptive_by_group,
            solver_by_group,
            solver_styles,
            mode_styles,
        )
        ax.set_title(_format_schedule_short(schedule))
        ax.set_xlabel("B")
    for ax in axes_flat[len(schedules) :]:
        ax.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("KS change vs baseline (%)")
    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS['percent_change']['title']}")
    return fig


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


def _dynamic_layout_pads(fig):
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    legend_count = len(fig.legends)
    legend_entries = sum(len(legend.texts) for legend in fig.legends)
    axis_count = len(visible_axes)

    pads = dict(LAYOUT_PADS)
    pads["w_pad"] = max(pads["w_pad"], 0.18)
    pads["h_pad"] = min(
        0.52,
        max(pads["h_pad"], 0.18 + 0.04 * legend_count + 0.01 * legend_entries),
    )
    if axis_count > 1:
        spacing_boost = min(0.08, 0.015 * axis_count)
        pads["wspace"] = max(pads["wspace"], 0.10 + spacing_boost)
        pads["hspace"] = max(pads["hspace"], 0.10 + spacing_boost)
    return pads


def _savefig_pad_inches(fig) -> float:
    return 0.08 + 0.02 * min(len(fig.legends), 3)


def _save_figure(fig, output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.set_constrained_layout(True)
    fig.set_constrained_layout_pads(**_dynamic_layout_pads(fig))
    fig.canvas.draw()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
        pad_inches=_savefig_pad_inches(fig),
    )
    plt.close(fig)


def main():
    args = parse_args()
    plot_names = _parse_plots(args.plots)

    rows = _load_rows(args.csv_file)
    if not rows:
        raise ValueError("No valid rows found in CSV after filtering failed/invalid records")

    solver_styles = _build_solver_style_map(rows)
    mode_styles = _build_mode_style_map(rows)

    output_paths = []
    for plot_name in plot_names:
        if plot_name == "main":
            output_path = args.output or _default_output_path(args.csv_file)
            fig = _plot_main(rows, args)
        elif plot_name == "best_config":
            output_path = _suffixed_output_path(
                args.csv_file, PLOT_SPECS["best_config"]["filename_suffix"]
            )
            title = (
                f"{args.title} - best adaptive config by B"
                if args.title
                else PLOT_SPECS["best_config"]["title"]
            )
            fig = _plot_best_config(rows, args.std, args.ci, title)
            if fig is None:
                continue
        elif plot_name == "optimal_b1":
            if not _adaptive_rows(rows):
                continue
            output_path = _suffixed_output_path(
                args.csv_file, PLOT_SPECS[plot_name]["filename_suffix"]
            )
            fig = _plot_optimal_b1(rows, args.title, mode_styles)
        elif plot_name == "optimal_ni":
            if not _adaptive_rows(rows):
                continue
            output_path = _suffixed_output_path(
                args.csv_file, PLOT_SPECS[plot_name]["filename_suffix"]
            )
            fig = _plot_optimal_ni(rows, args.title, mode_styles, args.ci)
        elif plot_name == "percent_change":
            output_path = _suffixed_output_path(
                args.csv_file, PLOT_SPECS[plot_name]["filename_suffix"]
            )
            fig = _plot_percent_change(rows, args.title, solver_styles, mode_styles)
        else:
            continue

        _save_figure(fig, output_path)
        output_paths.append(output_path)

    print(f"Loaded {len(rows)} valid rows")
    for path in output_paths:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
