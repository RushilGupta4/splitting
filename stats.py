import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


MODE_ROWS = [
    ("pilot_tree", False),
    ("pilot_tree", True),
    ("independent", False),
    ("independent", True),
    None,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the optimal B1 per adaptive mode from compare.py CSV output"
    )
    parser.add_argument("csv_file", type=str, help="Path to compare.py summary CSV")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output PNG (default: same as csv_file with _optimal_B1.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title",
    )
    return parser.parse_args()


def _default_output_path(csv_path: str) -> str:
    base, ext = os.path.splitext(csv_path)
    if not base:
        return csv_path + "_optimal_B1.png"
    if not ext:
        return csv_path + "_optimal_B1.png"
    return base + "_optimal_B1.png"


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

            mode = raw_row.get("mode", "").strip()
            if mode != "estimate_and_sample":
                continue

            mean_ks = _parse_float(raw_row.get("mean_ks", ""))
            B = _parse_int(raw_row.get("B", ""))
            B1 = _parse_int(raw_row.get("B1", ""))
            sampling_steps = _parse_int(raw_row.get("sampling_steps", ""))
            eta = _parse_float(raw_row.get("eta", ""))
            sigma_estimation_mode = raw_row.get("sigma_estimation_mode", "").strip()
            reuse_raw = raw_row.get("reuse_phase1_samples")
            if reuse_raw is None:
                reuse_raw = raw_row.get("reuse_pilot_samples", "")
            reuse_phase1_samples = _parse_bool(reuse_raw or "")

            if (
                mean_ks is None
                or B is None
                or B1 is None
                or sampling_steps is None
                or eta is None
                or not sigma_estimation_mode
                or reuse_phase1_samples is None
            ):
                continue

            rows.append(
                {
                    "B": B,
                    "B1": B1,
                    "sampling_steps": sampling_steps,
                    "eta": eta,
                    "mean_ks": mean_ks,
                    "sigma_estimation_mode": sigma_estimation_mode,
                    "reuse_phase1_samples": reuse_phase1_samples,
                }
            )
    return rows


def _format_eta(eta: float) -> str:
    return f"{eta:g}"


def _schedule_key(row):
    return (row["sampling_steps"], row["eta"])


def _mode_title(mode_key):
    if mode_key is None:
        return "best overall"
    sigma_estimation_mode, reuse_phase1_samples = mode_key
    reuse_label = "reuse" if reuse_phase1_samples else "fresh"
    return f"{sigma_estimation_mode} {reuse_label}"


def _optimal_rows(rows):
    best_by_group = {}
    for row in rows:
        mode_key = (row["sigma_estimation_mode"], row["reuse_phase1_samples"])
        group_key = (_schedule_key(row), mode_key, row["B"])
        best = best_by_group.get(group_key)
        if best is None or (row["mean_ks"], row["B1"]) < (best["mean_ks"], best["B1"]):
            best_by_group[group_key] = row
    return best_by_group


def _optimal_rows_overall(rows):
    best_by_group = {}
    for row in rows:
        group_key = (_schedule_key(row), row["B"])
        best = best_by_group.get(group_key)
        if best is None or (row["mean_ks"], row["B1"]) < (best["mean_ks"], best["B1"]):
            best_by_group[group_key] = row
    return best_by_group


def _plot_mode_schedule(ax, selected_rows, mode_key):
    if mode_key is None:
        points = list(selected_rows)
    else:
        points = [
            row
            for row in selected_rows
            if (row["sigma_estimation_mode"], row["reuse_phase1_samples"]) == mode_key
        ]
    points.sort(key=lambda row: row["B"])

    if not points:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.grid(alpha=0.2)
        return

    x_values = [row["B"] for row in points]
    y_values = [row["B1"] for row in points]
    ax.plot(
        x_values,
        y_values,
        color="#1f77b4",
        linestyle="-",
        marker="o",
        linewidth=1.8,
        markersize=5,
    )
    ax.grid(alpha=0.25)


def main():
    args = parse_args()

    rows = _load_rows(args.csv_file)
    if not rows:
        raise ValueError("No valid adaptive rows found in CSV after filtering")

    schedules = sorted({_schedule_key(row) for row in rows})
    selected = _optimal_rows(rows)
    selected_overall = _optimal_rows_overall(rows)
    selected_by_schedule = defaultdict(list)
    selected_overall_by_schedule = defaultdict(list)
    for (schedule, _, _), row in selected.items():
        selected_by_schedule[schedule].append(row)
    for (schedule, _), row in selected_overall.items():
        selected_overall_by_schedule[schedule].append(row)

    nrows = len(MODE_ROWS)
    ncols = len(schedules)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 3.8 * nrows),
        squeeze=False,
        sharex="col",
        sharey=True,
    )

    for col_idx, schedule in enumerate(schedules):
        schedule_rows = selected_by_schedule.get(schedule, [])
        overall_rows = selected_overall_by_schedule.get(schedule, [])
        for row_idx, mode_key in enumerate(MODE_ROWS):
            ax = axes[row_idx][col_idx]
            plot_rows = overall_rows if mode_key is None else schedule_rows
            _plot_mode_schedule(ax, plot_rows, mode_key)
            if row_idx == 0:
                sampling_steps, eta = schedule
                ax.set_title(f"steps={sampling_steps}, eta={_format_eta(eta)}")
            if col_idx == 0:
                ax.set_ylabel(f"Optimal B'\n{_mode_title(mode_key)}")

            x_formatter = ScalarFormatter(useMathText=True)
            x_formatter.set_powerlimits((-2, 3))
            y_formatter = ScalarFormatter(useMathText=True)
            y_formatter.set_powerlimits((-2, 3))
            ax.xaxis.set_major_formatter(x_formatter)
            ax.yaxis.set_major_formatter(y_formatter)

    for ax in axes[-1]:
        ax.set_xlabel("B")

    if args.title:
        fig.suptitle(args.title)

    output_path = args.output or _default_output_path(args.csv_file)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    top = 0.94 if args.title else 0.97
    fig.tight_layout(rect=(0, 0, 1, top))
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")
    print(
        f"Plotted {len(rows)} valid adaptive rows across {len(schedules)} step/eta schedules"
    )


if __name__ == "__main__":
    main()
