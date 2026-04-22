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
    return parser.parse_args()


def _default_output_path(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    if not base:
        return csv_path + ".png"
    return base + ".png"


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
                    "mode": raw_row.get("mode", "").strip(),
                    "method_label": raw_row.get("method_label", "").strip(),
                    "sigma_estimation_mode": raw_row.get(
                        "sigma_estimation_mode", ""
                    ).strip(),
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
    return (
        "adaptive",
        row["sigma_estimation_mode"],
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


def _plot_schedule(ax, schedule_rows, b1_colors):
    grouped = defaultdict(list)
    for row in schedule_rows:
        grouped[_series_key(row)].append(row)

    for series_key, series_rows in sorted(grouped.items()):
        points = sorted(series_rows, key=lambda row: row["B"])
        x_values = [row["B"] for row in points]
        y_values = [row["mean_ks"] for row in points]

        if series_key[0] == "baseline":
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

        _, sigma_estimation_mode, reuse_phase1_samples, B1 = series_key
        color = b1_colors.get(B1, "#1f77b4")
        linestyle = ":" if sigma_estimation_mode == "pilot_tree" else "-"
        marker = "s" if reuse_phase1_samples else "o"
        ax.plot(
            x_values,
            y_values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            alpha=0.95,
        )

    ax.grid(alpha=0.25)
    ax.set_xlabel("Total B")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(formatter)


def _make_legends(fig, b1_colors):
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
    fig.legend(
        semantic_handles,
        semantic_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
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
        _plot_schedule(ax, schedule_groups[schedule], b1_colors)
        sampling_steps, eta = schedule
        ax.set_title(f"steps={sampling_steps}, eta={_format_eta(eta)}")

    for ax in axes_flat[n_schedules:]:
        ax.set_visible(False)

    for row_axes in axes:
        row_axes[0].set_ylabel("Mean KS")
        row_axes[0].set_yscale("log")

    if args.title:
        fig.suptitle(args.title)

    _make_legends(fig, b1_colors)

    output_path = args.output or _default_output_path(args.csv_file)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")
    print(f"Plotted {len(rows)} valid rows across {n_schedules} inference schedules")


if __name__ == "__main__":
    main()
