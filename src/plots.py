import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

SOLVER_LINESTYLES = ("--", "-.", ":", (0, (3, 1, 1, 1)))
SOLVER_MARKERS = ("^", "D", "x", "P", "v", "*", "h")
ADAPTIVE_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "h")

STRUCTURAL_CSV_FIELDS = {
    "mode",
    "sampler",
    "sampling_steps",
    "sampling_params",
    "step_schedule",
    "B",
    "B1",
    "crossfit_q_folds",
    "crossfit_q_mlp_loss",
    "reuse",
    "optimizer",
    "solver",
    "solver_steps",
    "solver_params",
    "nfe_per_sample",
    "N_i",
    "N_i_std",
}
MMD_SPACE_LABELS = {
    "mmd_latent": "Latent MMD",
    "mmd_pixel": "Pixel MMD",
    "mmd_inception": "Inception MMD",
    "mmd_dino": "DINOv2 MMD",
}
METRIC_ORDER = ("mmd", *MMD_SPACE_LABELS, "ks")

PLOT_SPECS = {
    "main": {"filename_suffix": None, "title": "Metrics vs B"},
    "optimal_b1": {"filename_suffix": "optimal_B1", "title": "Optimal B'"},
    "optimal_ni": {"filename_suffix": "optimal_Ni", "title": "Chosen N_i"},
    "cumulative_splits": {
        "filename_suffix": "cumulative_splits",
        "title": "Cumulative splits R_i",
    },
    "percent_change": {
        "filename_suffix": "percent_change",
        "title": "Percent Change by Metric",
    },
}
PLOT_NAMES = tuple(PLOT_SPECS)
DEFAULT_PLOTS = ("main", "percent_change", "cumulative_splits")


@dataclass(frozen=True)
class Row:
    mode: str
    sampler: str
    sampling_params: dict[str, Any]
    step_schedule: str
    B: int
    B1: int | None
    B1_spec: str
    crossfit_q_folds: int
    crossfit_q_mlp_loss: str
    reuse: bool | None
    optimizer: str
    solver: str
    solver_steps: int | None
    solver_params: dict[str, Any]
    metrics: dict[str, dict[str, float | int | None]]
    N_i: list[float]
    N_i_std: list[float]
    sampling_steps: int | None = None
    R_i: list[float] | None = None


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
        help="Show mean +/- std intervals on the main plot",
    )
    intervals.add_argument(
        "--ci",
        type=float,
        default=None,
        help="Show CI half-widths at this level on the main and optimal_ni plots",
    )
    return parser.parse_args()


def _parse_plots(raw: str):
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("--plots must include at least one plot name")
    if "all" in names:
        return list(DEFAULT_PLOTS)
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


def _metric_sort_key(metric: str):
    try:
        return (METRIC_ORDER.index(metric), metric)
    except ValueError:
        return (len(METRIC_ORDER), metric)


def _detect_metrics(fieldnames) -> list[str]:
    fields = set(fieldnames or [])
    return [metric for metric in METRIC_ORDER if f"mean_{metric}" in fields]


def _parse_metric_values(raw_row: dict[str, str], metrics: list[str]):
    values: dict[str, dict[str, float | int | None]] = {}
    for metric in metrics:
        mean_value = _parse_float(raw_row.get(f"mean_{metric}", ""))
        if mean_value is None:
            continue
        n_valid = _parse_int(raw_row.get(f"n_valid_{metric}", ""))
        values[metric] = {
            "mean": mean_value,
            "std": _parse_float(raw_row.get(f"std_{metric}", "")),
            "n_valid": n_valid,
        }
    return values


def _available_metrics(rows) -> list[str]:
    metrics = {metric for row in rows for metric in row.metrics}
    return sorted(metrics, key=_metric_sort_key)


def _metric_value(row: Row, metric: str, field: str = "mean"):
    return (row.metrics.get(metric) or {}).get(field)


def _metric_label(metric: str):
    if metric == "mmd":
        return "MMD"
    if metric == "ks":
        return "KS"
    if metric in MMD_SPACE_LABELS:
        return MMD_SPACE_LABELS[metric]
    return str(metric).replace("_", " ").upper()


def _metric_score(row: Row, metric: str):
    value = _metric_value(row, metric)
    if value is None:
        value = float("inf")
    return (
        value,
        row.B1 if row.B1 is not None else -1,
        _row_stable_label(row),
    )


def _load_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = sorted(STRUCTURAL_CSV_FIELDS - set(fieldnames))
        if missing:
            raise ValueError(
                "CSV is missing structured field(s): "
                f"{', '.join(missing)}; rerun compare.py"
            )
        metrics = _detect_metrics(fieldnames)
        if not metrics:
            raise ValueError("CSV does not contain any metric columns")
        for raw_row in reader:
            if raw_row.get("error", "").strip():
                continue
            B = _parse_int(raw_row.get("B", ""))
            if B is None:
                continue
            metric_values = _parse_metric_values(raw_row, metrics)
            if not metric_values:
                continue
            mode = (raw_row.get("mode") or "").strip()
            if mode not in {"fixed_N", "solver_baseline", "adaptive"}:
                continue
            rows.append(
                Row(
                    mode=mode,
                    sampler=(raw_row.get("sampler") or "").strip(),
                    sampling_params=_parse_json_dict(
                        raw_row.get("sampling_params", "")
                    ),
                    step_schedule=(raw_row.get("step_schedule") or "default").strip(),
                    B=B,
                    B1=_parse_int(raw_row.get("B1", "")),
                    B1_spec=(raw_row.get("B1_spec") or "").strip(),
                    crossfit_q_folds=_parse_int(raw_row.get("crossfit_q_folds", "")) or 1,
                    crossfit_q_mlp_loss=(raw_row.get("crossfit_q_mlp_loss") or "").strip(),
                    reuse=_parse_bool(raw_row.get("reuse", "")),
                    optimizer=(raw_row.get("optimizer") or "").strip(),
                    solver=(raw_row.get("solver") or "").strip(),
                    solver_steps=_parse_int(raw_row.get("solver_steps", "")),
                    solver_params=_parse_json_dict(raw_row.get("solver_params", "")),
                    metrics=metric_values,
                    N_i=_parse_float_list(raw_row.get("N_i", "")),
                    N_i_std=_parse_float_list(raw_row.get("N_i_std", "")),
                    sampling_steps=_parse_int(raw_row.get("sampling_steps", "")),
                )
            )
    return rows


def _json_key(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return _json_key(value)
    return str(value)


def _format_count(value) -> str:
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return str(value)


def _format_b1_series_label(series_key) -> str:
    mode, payload = str(series_key).split(":", 1)
    if mode == "ratio":
        return f"B1/B={payload}"
    if mode == "absolute":
        return f"B1={_format_count(payload)}"
    if mode == "power":
        coefficient, exponent = payload.split(",", 1)
        return f"B1={coefficient}\u00b7B^{exponent}"
    raise ValueError(f"Unknown B1 series key: {series_key!r}")


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


def _sampling_key(row: Row):
    return (row.sampler, _json_key(row.sampling_params))


def _schedule_key(row: Row):
    return (_sampling_key(row), row.step_schedule)


def _solver_family_key(row: Row):
    return (row.solver, row.solver_steps, _json_key(row.solver_params))


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


def _main_panel_title(rows, facet_key) -> str:
    title = _format_schedule_short(facet_key).strip()
    if title and title != "sampling config":
        return title
    matching_rows = [
        row
        for row in rows
        if row.mode in {"adaptive", "fixed_N"} and _schedule_key(row) == facet_key
    ]
    if matching_rows:
        return _format_schedule_short(_schedule_key(matching_rows[0]))
    return title or "sampling config"


def _set_main_panel_title(ax, title: str):
    ax.text(
        0.5,
        1.015,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.titlesize"],
        fontweight=plt.rcParams["axes.titleweight"],
        clip_on=False,
    )


def _method_key(row: Row):
    if row.mode == "fixed_N":
        return ("fixed_N",)
    if row.mode == "solver_baseline":
        return ("solver", *_solver_family_key(row))
    return (
        "adaptive",
        row.crossfit_q_mlp_loss,
        bool(row.reuse),
        row.optimizer,
        int(row.crossfit_q_folds),
    )


def _method_display_label(key) -> str:
    if key[0] == "fixed_N":
        return "fixed"
    if key[0] == "solver":
        _, solver, solver_steps, params_json = key
        params = json.loads(params_json) if params_json else {}
        label = _pretty_name(solver)
        if solver_steps is not None:
            label += f" {solver_steps}"
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
            (
                _format_schedule_key(_schedule_key(row))
                if row.mode != "solver_baseline"
                else row.step_schedule
            ),
            (
                _format_b1_series_label(row.B1_spec)
                if row.mode == "adaptive"
                else ""
            ),
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
        row.crossfit_q_mlp_loss,
        bool(row.reuse),
        row.optimizer,
        int(row.crossfit_q_folds),
    )


def _mode_keys(rows):
    return sorted({_mode_key(row) for row in _adaptive_rows(rows)})


def _mode_title(mode_key):
    crossfit_q_mlp_loss, reuse, optimizer, crossfit_q_folds = mode_key
    label = "reuse" if reuse else "fresh"
    if crossfit_q_mlp_loss:
        label += f" {crossfit_q_mlp_loss}"
    if optimizer:
        label += f" {optimizer}"
    return f"{label} K={int(crossfit_q_folds)}"


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
        return ("adaptive", _method_key(row), row.B1_spec)
    return _baseline_series_key(row)


def best_row(rows, metric: str = "ks"):
    candidates = [row for row in rows if _metric_value(row, metric) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _metric_score(row, metric))


def _best_by_group(rows, *, key_fn, score_fn=None, metric: str = "ks"):
    out = {}
    for row in rows:
        key = key_fn(row)
        if score_fn is None:
            if _metric_value(row, metric) is None:
                continue
            score = _metric_score(row, metric)
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
    b1_values = sorted({row.B1_spec for row in _adaptive_rows(rows)})
    return {
        b1: ADAPTIVE_MARKERS[i % len(ADAPTIVE_MARKERS)]
        for i, b1 in enumerate(b1_values)
    }


def _build_mode_style_map(rows):
    mode_keys = _mode_keys(rows)
    cmap = plt.get_cmap("tab10") if len(mode_keys) <= 10 else plt.get_cmap("tab20")
    return {
        mode_key: (cmap(i % cmap.N), "-", ADAPTIVE_MARKERS[i % len(ADAPTIVE_MARKERS)])
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


def _solver_series_keys(rows):
    return sorted(
        {_baseline_series_key(row) for row in _solver_rows(rows)},
        key=str,
    )


def _main_solver_series_style(series_key, rows, method_styles):
    solver_series_keys = _solver_series_keys(rows)
    solver_idx = (
        solver_series_keys.index(series_key) if series_key in solver_series_keys else 0
    )
    method_key = _baseline_method_key(series_key)
    return (
        method_styles.get(method_key, "#1f77b4"),
        SOLVER_LINESTYLES[solver_idx % len(SOLVER_LINESTYLES)],
        SOLVER_MARKERS[solver_idx % len(SOLVER_MARKERS)],
    )


def _ci_multiplier(confidence_level: float, degrees_of_freedom: int):
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("--ci must be between 0 and 1, e.g. 0.95")
    quantile = 0.5 + confidence_level / 2.0
    from scipy.stats import t

    return float(t.ppf(quantile, degrees_of_freedom))


def _ci_half_width(std_ks, n_valid_ks, ci_level):
    if std_ks is None or n_valid_ks is None or n_valid_ks <= 1:
        return None
    return _ci_multiplier(ci_level, n_valid_ks - 1) * std_ks / math.sqrt(n_valid_ks)


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


def _with_top_legend_height(figsize, label_count: int, ncol: int):
    width, height = figsize
    legend_rows = math.ceil(int(label_count) / max(1, int(ncol))) if label_count else 0
    return width, height + 0.45 * legend_rows


def _legend_line(color, linestyle, marker, **kwargs):
    return Line2D([0], [0], color=color, linestyle=linestyle, marker=marker, **kwargs)


def _main_legend_ncol(labels) -> int:
    if any(len(label) > 28 for label in labels):
        return min(2, max(1, len(labels)))
    return min(4, max(1, len(labels)))


def _apply_scalar_formatters(ax, format_x: bool = True, format_y: bool = True):
    if format_x:
        x_formatter = ScalarFormatter(useMathText=True)
        x_formatter.set_powerlimits((-2, 3))
        ax.xaxis.set_major_formatter(x_formatter)
    if format_y:
        y_formatter = ScalarFormatter(useMathText=True)
        y_formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(y_formatter)


def _plot_series(
    ax,
    rows,
    *,
    metric: str,
    color,
    linestyle,
    marker,
    label=None,
    show_std=False,
    ci_level=None,
    zorder=3,
):
    points = [
        row
        for row in sorted(rows, key=lambda row: row.B)
        if _metric_value(row, metric) is not None
    ]
    if not points:
        return False
    x_values = [row.B for row in points]
    y_values = [_metric_value(row, metric) for row in points]
    if show_std:
        _fill_interval(
            ax,
            x_values,
            y_values,
            [_metric_value(row, metric, "std") for row in points],
            color,
            zorder - 1,
        )
    if ci_level is not None:
        _fill_interval(
            ax,
            x_values,
            y_values,
            [
                _ci_half_width(
                    _metric_value(row, metric, "std"),
                    _metric_value(row, metric, "n_valid"),
                    ci_level,
                )
                for row in points
            ],
            color,
            zorder - 1,
        )
    ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=1.0,
        markersize=3,
        label=label,
        zorder=zorder,
    )
    return True


def _main_facet_keys(rows):
    non_solver_keys = sorted(
        {_schedule_key(row) for row in rows if row.mode != "solver_baseline"}
    )
    if non_solver_keys:
        return non_solver_keys
    return [(("", "{}"), row.step_schedule) for row in _solver_rows(rows)]


def _plot_main_panel(
    ax, rows, facet_key, metric, method_styles, b1_markers, show_std, ci_level
):
    panel_rows = [
        row
        for row in rows
        if (
            row.mode in {"adaptive", "fixed_N"}
            and _schedule_key(row) == facet_key
        )
        or row.mode == "solver_baseline"
    ]
    grouped = defaultdict(list)
    for row in panel_rows:
        series_key = _main_series_key(row)
        grouped[series_key].append(row)

    plotted_any = False
    for series_key, series_rows in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        kind = series_key[0]
        method_key = (
            series_key[1] if kind == "adaptive" else _baseline_method_key(series_key)
        )
        color = method_styles.get(method_key, "#1f77b4")
        linestyle = "-"
        marker = "o"
        zorder = 3
        if kind == "solver":
            color, linestyle, marker = _main_solver_series_style(
                series_key, rows, method_styles
            )
            zorder = 5
        elif kind == "adaptive":
            marker = b1_markers.get(series_key[2], "o")
            zorder = 2
        else:
            zorder = 4
        plotted_any |= _plot_series(
            ax,
            series_rows,
            metric=metric,
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
    metric_values = [_metric_value(row, metric) for row in panel_rows]
    metric_values = [value for value in metric_values if value is not None]
    if metric_values and all(value > 0 for value in metric_values):
        ax.set_yscale("log")
    ax.set_xlabel("Total B")
    _apply_scalar_formatters(ax, format_x=True, format_y=True)


def _main_legend_entries(rows, method_styles, b1_markers):
    handles = []
    labels = []
    baseline_series_keys = sorted(
        {_baseline_series_key(row) for row in _all_baseline_rows(rows)}, key=str
    )
    for series_key in baseline_series_keys:
        color = method_styles.get(_baseline_method_key(series_key), "#1f77b4")
        linestyle, marker = "-", "o"
        if series_key[0] == "solver":
            color, linestyle, marker = _main_solver_series_style(
                series_key, rows, method_styles
            )
        handles.append(_legend_line(color, linestyle, marker, linewidth=2.0))
        labels.append(_baseline_series_label(series_key))
    for method_key in sorted({_method_key(row) for row in _adaptive_rows(rows)}, key=str):
        color = method_styles.get(method_key, "#1f77b4")
        handles.append(_legend_line(color, "-", "o", linewidth=2.0))
        labels.append(_method_display_label(method_key))
    for b1_key, marker in sorted(b1_markers.items()):
        handles.append(_legend_line("0.35", "None", marker, markersize=6))
        labels.append(_format_b1_series_label(b1_key))
    return handles, labels


def _make_main_legend(fig, handles, labels, legend_ncol):
    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=legend_ncol,
            frameon=False,
            borderaxespad=1.0,
            columnspacing=1.4,
            labelspacing=0.8,
        )


def _plot_main(rows, args):
    metrics = _available_metrics(rows)
    facet_keys = _main_facet_keys(rows)
    sampling_keys = sorted({key[0] for key in facet_keys})
    schedules = sorted({key[1] for key in facet_keys})
    method_styles = _build_method_style_map(rows)
    b1_markers = _build_b1_marker_map(rows)
    legend_handles, legend_labels = _main_legend_entries(rows, method_styles, b1_markers)
    legend_count = len(legend_labels)
    legend_ncol = _main_legend_ncol(legend_labels)

    if len(sampling_keys) > 1 and len(schedules) > 1:
        row_keys = [(metric, sampling_key) for metric in metrics for sampling_key in sampling_keys]
        figsize = _with_top_legend_height(
            (8.5 * len(schedules), 6.0 * len(row_keys)),
            legend_count,
            legend_ncol,
        )
        fig, axes = plt.subplots(
            len(row_keys),
            len(schedules),
            figsize=figsize,
            squeeze=False,
            sharey="row",
            constrained_layout=True,
        )
        for row_idx, (metric, sampling_key) in enumerate(row_keys):
            for col_idx, schedule in enumerate(schedules):
                ax = axes[row_idx][col_idx]
                facet_key = (sampling_key, schedule)
                if facet_key not in facet_keys:
                    ax.set_visible(False)
                    continue
                _plot_main_panel(
                    ax,
                    rows,
                    facet_key,
                    metric,
                    method_styles,
                    b1_markers,
                    args.std,
                    args.ci,
                )
                if row_idx == 0:
                    _set_main_panel_title(
                        ax, schedule or _main_panel_title(rows, facet_key)
                    )
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{_metric_label(metric)}\n{_format_sampling_config_short(sampling_key)}"
                    )
    else:
        panels = [(metric, facet_key) for metric in metrics for facet_key in facet_keys]
        nrows, ncols = _make_subplot_grid(len(panels), max_cols=2)
        figsize = _with_top_legend_height(
            (9.5 * ncols, 6.5 * nrows),
            legend_count,
            legend_ncol,
        )
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            sharey=False,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()
        for ax, (metric, facet_key) in zip(axes_flat, panels):
            _plot_main_panel(
                ax,
                rows,
                facet_key,
                metric,
                method_styles,
                b1_markers,
                args.std,
                args.ci,
            )
            title = _main_panel_title(rows, facet_key)
            if len(metrics) > 1:
                title = f"{title} - {_metric_label(metric)}"
            _set_main_panel_title(ax, title)
            ax.set_ylabel(_metric_label(metric))
        for ax in axes_flat[len(panels) :]:
            ax.set_visible(False)

    if args.title:
        fig.suptitle(args.title)
    _make_main_legend(fig, legend_handles, legend_labels, legend_ncol)
    return fig


def _make_mode_legend(fig, mode_keys, mode_styles):
    handles = []
    labels = []
    for mode_key in mode_keys:
        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        handles.append(
            _legend_line(color, linestyle, marker, linewidth=1.8, markersize=5)
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


def _best_per_mode(rows, metric):
    return _best_by_group(
        rows,
        key_fn=lambda row: (_schedule_key(row), _mode_key(row), row.B),
        metric=metric,
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
    ax.set_xscale("log")
    ax.set_yscale("log")
    _apply_scalar_formatters(ax, format_x=True, format_y=True)


def _plot_optimal_b1(rows, title_prefix, mode_styles):
    rows = _adaptive_rows(rows)
    metrics = _available_metrics(rows)
    mode_keys = _mode_keys(rows)
    schedules = sorted({_schedule_key(row) for row in rows}, key=_format_schedule_key)
    panels = [(metric, schedule) for metric in metrics for schedule in schedules]

    nrows, ncols = _make_subplot_grid(len(panels), max_cols=2)
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
    selected_by_metric_and_schedule = {}
    for metric in metrics:
        selected_by_schedule = defaultdict(list)
        for (schedule, _, _), row in _best_per_mode(rows, metric).items():
            selected_by_schedule[schedule].append(row)
        selected_by_metric_and_schedule[metric] = selected_by_schedule
    for ax, (metric, schedule) in zip(axes_flat, panels):
        _plot_optimal_b1_panel(
            ax,
            selected_by_metric_and_schedule.get(metric, {}).get(schedule, []),
            mode_keys,
            mode_styles,
        )
        ax.set_title(f"{_format_schedule_short(schedule)} - {_metric_label(metric)}")
        ax.set_xlabel("B")
    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("Optimal B'")
    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS['optimal_b1']['title']}")
    _make_mode_legend(fig, mode_keys, mode_styles)
    return fig


def _cumulative_products(values):
    products = []
    running_product = 1.0
    for value in values:
        running_product *= value
        products.append(running_product)
    return products


def _load_cumulative_allocations(csv_path, rows):
    """Average cumulative allocations from the runs underlying each CSV row."""
    path = Path(csv_path)
    prefix = "compare_results_"
    if not path.stem.startswith(prefix):
        raise ValueError(f"Cannot resolve split schedule from CSV path: {csv_path}")
    schedule = [float(value) for value in path.stem[len(prefix):].split("_")]
    configs = []
    for config_path in sorted((path.parent / "runs").glob("*/config.json")):
        with config_path.open() as handle:
            configs.append((config_path.parent, json.load(handle)["cache_key"]))

    resolved = []
    for row in rows:
        if row.mode != "adaptive" or not row.N_i:
            resolved.append(row)
            continue
        candidates = []
        for run_dir, key in configs:
            spec = key.get("spec", {})
            sampling = spec.get("runner_sampling_config", {})
            if (
                spec.get("mode") != "estimate_and_sample"
                or spec.get("B") != row.B
                or spec.get("B1") != row.B1
                or spec.get("split_percentages") != schedule
                or spec.get("optimization_mode") != row.optimizer
                or spec.get("reuse_phase1_samples") != row.reuse
                or sampling.get("sampler") != row.sampler
                or sampling.get("sampling_steps") != row.sampling_steps
                or any(sampling.get(k) != v for k, v in row.sampling_params.items())
                or int(key.get("crossfit_q_folds", 1)) != row.crossfit_q_folds
                or (
                    row.crossfit_q_mlp_loss
                    and (key.get("crossfit_q_mlp_params") or {}).get("loss")
                    != row.crossfit_q_mlp_loss
                )
            ):
                continue
            n = max(int(values.get("n_valid") or 0) for values in row.metrics.values())
            with (run_dir / "runs.jsonl").open() as handle:
                records = [json.loads(line) for line in handle if line.strip()][:n]
            factors = [record.get("N_i", []) for record in records]
            if len(factors) != n or not n:
                continue
            if any(len(values) != len(row.N_i) for values in factors):
                continue
            if any(
                not math.isfinite(v) or v < 1.0 - 1e-8
                for values in factors for v in values
            ):
                raise ValueError(f"Invalid cached split factors in {run_dir}")
            means = [sum(values) / n for values in zip(*factors)]
            if not all(
                math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-8)
                for a, b in zip(means, row.N_i)
            ):
                continue
            cumulative = [_cumulative_products(values) for values in factors]
            candidates.append([sum(values) / n for values in zip(*cumulative)])
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one matching allocation cache for {csv_path}, "
                f"B={row.B}, B1={row.B1}, optimizer={row.optimizer}; "
                f"found {len(candidates)}. Cumulative plots require per-run N_i."
            )
        resolved.append(replace(row, R_i=candidates[0]))
    return resolved


def _plot_split_levels_panel(
    ax, selected_rows, max_levels, mode_keys, mode_styles, metric, ci_level, cumulative
):
    plotted_any = False
    for mode_key in mode_keys:
        row = best_row(
            (row for row in selected_rows if _mode_key(row) == mode_key),
            metric=metric,
        )
        if row is None or not row.N_i:
            continue
        color, linestyle, marker = _mode_style(mode_key, mode_styles)
        x_values = list(range(1, len(row.N_i) + 1))
        if (
            not cumulative
            and ci_level is not None
            and len(row.N_i_std) == len(row.N_i)
        ):
            n_valid = _metric_value(row, metric, "n_valid")
            _fill_interval(
                ax,
                x_values,
                row.N_i,
                [_ci_half_width(std, n_valid, ci_level) for std in row.N_i_std],
                color,
                1,
            )
        ax.plot(
            x_values,
            row.R_i if cumulative else row.N_i,
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


def _plot_split_levels(rows, title_prefix, mode_styles, plot_name, ci_level=None):
    """Mean split factors or mean per-run cumulative allocations."""
    cumulative = plot_name == "cumulative_splits"
    if cumulative and any(row.R_i is None for row in _adaptive_rows(rows) if row.N_i):
        raise ValueError("Cumulative plots require per-run allocations from the cache")
    ylabel = "Cumulative splits $R_i$" if cumulative else "Chosen N_i"
    rows = _adaptive_rows(rows)
    metrics = _available_metrics(rows)
    mode_keys = _mode_keys(rows)
    schedules = sorted({_schedule_key(row) for row in rows}, key=_format_schedule_key)
    budgets = sorted({row.B for row in rows})
    selected_by_metric = {}
    selected_values = []
    for metric in metrics:
        selected = defaultdict(list)
        for (schedule, _, budget), row in _best_per_mode(rows, metric).items():
            selected[(schedule, budget)].append(row)
            selected_values.append(row)
        selected_by_metric[metric] = selected
    max_levels = max((len(row.N_i) for row in selected_values), default=0)
    legend_count = len(mode_keys)
    legend_ncol = min(3, max(1, legend_count))
    figsize = _with_top_legend_height(
        (4.8 * max(1, len(budgets)), 4.0 * max(1, len(schedules) * len(metrics))),
        legend_count,
        legend_ncol,
    )

    fig, axes = plt.subplots(
        max(1, len(schedules) * len(metrics)),
        max(1, len(budgets)),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    row_keys = [(metric, schedule) for metric in metrics for schedule in schedules]
    for row_idx, (metric, schedule) in enumerate(row_keys):
        for col_idx, budget in enumerate(budgets):
            ax = axes[row_idx][col_idx]
            _plot_split_levels_panel(
                ax,
                selected_by_metric.get(metric, {}).get((schedule, budget), []),
                max_levels,
                mode_keys,
                mode_styles,
                metric,
                ci_level,
                cumulative,
            )
            if row_idx == 0:
                ax.set_title(f"B={_format_count(budget)}")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{ylabel}\n{_format_schedule_short(schedule)}\n{_metric_label(metric)}"
                )
            if row_idx == len(row_keys) - 1:
                ax.set_xlabel("Split level i")
    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS[plot_name]['title']}")
    _make_mode_legend(fig, mode_keys, mode_styles)
    return fig


def _plot_percent_line(
    ax,
    baseline_by_budget,
    candidate_by_budget,
    legend_handles,
    *,
    color,
    linestyle,
    marker,
    label,
    metric,
):
    x_values = []
    y_values = []
    for budget in sorted(set(baseline_by_budget) & set(candidate_by_budget)):
        baseline_value = _metric_value(baseline_by_budget[budget], metric)
        candidate_value = _metric_value(candidate_by_budget[budget], metric)
        if baseline_value in {None, 0} or candidate_value is None:
            continue
        x_values.append(budget)
        y_values.append(100.0 * (baseline_value - candidate_value) / baseline_value)
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
    legend_handles.setdefault(
        label, _legend_line(color, linestyle, marker, linewidth=2.0)
    )
    return True


def _plot_percent_change_panel(
    ax,
    schedule,
    metric,
    baseline_by_group,
    adaptive_by_group,
    solver_by_group,
    method_styles,
    b1_markers,
    solver_styles,
    legend_handles,
):
    baseline_by_budget = {
        budget: row
        for (group_schedule, budget), row in baseline_by_group.items()
        if group_schedule == schedule
    }
    plotted_any = False

    adaptive_series_keys = sorted(
        {
            (method_key, b1)
            for (group_schedule, method_key, b1, _) in adaptive_by_group
            if group_schedule == schedule
        },
        key=str,
    )
    for method_key, b1 in adaptive_series_keys:
        adaptive_by_budget = {
            budget: row
            for (group_schedule, group_method_key, group_b1, budget), row
            in adaptive_by_group.items()
            if (group_schedule, group_method_key, group_b1) == (schedule, method_key, b1)
        }
        plotted_any |= _plot_percent_line(
            ax,
            baseline_by_budget,
            adaptive_by_budget,
            legend_handles,
            color=method_styles.get(method_key, "#1f77b4"),
            linestyle="-",
            marker=b1_markers.get(b1, "o"),
            label=f"{_method_display_label(method_key)}, {_format_b1_series_label(b1)}",
            metric=metric,
        )

    solver_series_keys = sorted(
        {
            (group_step_schedule, solver_key)
            for (group_step_schedule, _, solver_key) in solver_by_group
        }
    )
    for solver_step_schedule, solver_key in solver_series_keys:
        solver_by_budget = {
            budget: row
            for (row_step_schedule, budget, group_solver_key), row
            in solver_by_group.items()
            if (row_step_schedule, group_solver_key) == (solver_step_schedule, solver_key)
        }
        color, linestyle, marker, label = _solver_style(solver_key, solver_styles)
        if solver_step_schedule:
            label = f"{label}, {solver_step_schedule}"
        plotted_any |= _plot_percent_line(
            ax,
            baseline_by_budget,
            solver_by_budget,
            legend_handles,
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=label,
            metric=metric,
        )

    if plotted_any:
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)
    ax.set_xscale("log")
    _apply_scalar_formatters(ax, format_x=True, format_y=True)


def _plot_percent_change(rows, title_prefix, solver_styles, method_styles, b1_markers):
    metrics = _available_metrics(rows)
    schedules = sorted(
        {_schedule_key(row) for row in rows if row.mode != "solver_baseline"},
        key=_format_schedule_key,
    )
    if not schedules:
        return plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    panels = [(metric, schedule) for metric in metrics for schedule in schedules]
    nrows, ncols = _make_subplot_grid(len(panels), max_cols=2)
    groups_by_metric = {}
    for metric in metrics:
        groups_by_metric[metric] = (
            _best_by_group(
                _baseline_rows(rows),
                key_fn=lambda row: (_schedule_key(row), row.B),
                metric=metric,
            ),
            _best_by_group(
                _adaptive_rows(rows),
                key_fn=lambda row: (
                    _schedule_key(row),
                    _method_key(row),
                    row.B1_spec,
                    row.B,
                ),
                metric=metric,
            ),
            _best_by_group(
                _solver_rows(rows),
                key_fn=lambda row: (row.step_schedule, row.B, _solver_family_key(row)),
                metric=metric,
            ),
        )
    legend_handles: dict[str, Line2D] = {}
    label_count = len(
        {
            f"{_method_display_label(_method_key(row))}, "
            f"{_format_b1_series_label(row.B1_spec)}"
            for row in _adaptive_rows(rows)
        }
    ) + len(_solver_series_keys(rows))
    legend_ncol = _main_legend_ncol([""] * label_count) if label_count else 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=_with_top_legend_height(
            (7.2 * ncols, 4.8 * nrows), label_count, legend_ncol
        ),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()
    for ax, (metric, schedule) in zip(axes_flat, panels):
        baseline_by_group, adaptive_by_group, solver_by_group = groups_by_metric[metric]
        _plot_percent_change_panel(
            ax,
            schedule,
            metric,
            baseline_by_group,
            adaptive_by_group,
            solver_by_group,
            method_styles,
            b1_markers,
            solver_styles,
            legend_handles,
        )
        ax.set_title(f"{_format_schedule_short(schedule)} - {_metric_label(metric)}")
        ax.set_xlabel("B")
    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("Change vs baseline (%)")
    if title_prefix:
        fig.suptitle(f"{title_prefix}: {PLOT_SPECS['percent_change']['title']}")
    if legend_handles:
        labels = sorted(legend_handles)
        fig.legend(
            [legend_handles[label] for label in labels],
            labels,
            loc="outside upper center",
            ncol=_main_legend_ncol(labels),
            frameon=False,
            borderaxespad=1.0,
            columnspacing=1.4,
            labelspacing=0.8,
        )
    return fig


def _output_path(csv_path: str, suffix: str | None = None) -> str:
    base = os.path.splitext(csv_path)[0] or csv_path
    return f"{base}_{suffix}.png" if suffix else f"{base}.png"


def _dynamic_layout_pads(fig):
    axis_count = sum(ax.get_visible() for ax in fig.axes)
    legend_entries = sum(len(legend.texts) for legend in fig.legends)
    spacing = 0.10 + min(0.08, 0.015 * axis_count) if axis_count > 1 else None
    return {
        "w_pad": 0.18,
        "h_pad": min(0.52, 0.18 + 0.04 * len(fig.legends) + 0.01 * legend_entries),
        "wspace": 0.08 if spacing is None else spacing,
        "hspace": 0.09 if spacing is None else spacing,
    }


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
        dpi=240,
        pad_inches=_savefig_pad_inches(fig),
    )
    plt.close(fig)


def main():
    args = parse_args()
    plot_names = _parse_plots(args.plots)

    rows = _load_rows(args.csv_file)
    if not rows:
        raise ValueError(
            "No valid rows found in CSV after filtering failed/invalid records"
        )
    if "cumulative_splits" in plot_names:
        rows = _load_cumulative_allocations(args.csv_file, rows)

    solver_styles = _build_solver_style_map(rows)
    mode_styles = _build_mode_style_map(rows)

    output_paths = []
    for plot_name in plot_names:
        if plot_name == "main":
            fig = _plot_main(rows, args)
        elif plot_name == "percent_change":
            fig = _plot_percent_change(
                rows,
                args.title,
                solver_styles,
                _build_method_style_map(rows),
                _build_b1_marker_map(rows),
            )
        elif not _adaptive_rows(rows):
            continue
        elif plot_name == "optimal_b1":
            fig = _plot_optimal_b1(rows, args.title, mode_styles)
        else:
            ci_level = args.ci if plot_name == "optimal_ni" else None
            fig = _plot_split_levels(rows, args.title, mode_styles, plot_name, ci_level)
        if plot_name == "main":
            output_path = args.output or _output_path(args.csv_file)
        else:
            output_path = _output_path(
                args.csv_file, PLOT_SPECS[plot_name]["filename_suffix"]
            )
        _save_figure(fig, output_path)
        output_paths.append(output_path)

    print(f"Loaded {len(rows)} valid rows")
    for path in output_paths:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
