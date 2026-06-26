import numpy as np
from runners.sde1d.sampling import SDE1D_SAMPLERS

START_SPLITS, END_SPLITS = 4, 4
SDE1D_SPLITS = [
    np.round(np.arange(j, 0, -1) / (j + 1), 2).tolist()
    for j in range(START_SPLITS, END_SPLITS + 1)
]


def make_sde1d_x_grid(start: float, end: float, gap: float) -> list[float]:
    start = float(start)
    end = float(end)
    gap = float(gap)
    if gap <= 0.0:
        raise ValueError("gap must be positive")
    if end < start:
        raise ValueError("end must be at least start")
    count = int(round((end - start) / gap)) + 1
    return np.linspace(start, end, count).round(5).tolist()


def _sampling_configs() -> list[dict[str, str]]:
    return [{"sampler": sampler} for sampler in SDE1D_SAMPLERS]


_EULER_CALIBRATED = {
    10_000: 60,
    20_000: 75,
    50_000: 100,
    100_000: 130,
    200_000: 160,
    500_000: 220,
    1_000_000: 280,
}

_MILSTEIN_CALIBRATED = {
    10_000: 35,
    20_000: 40,
    50_000: 50,
    100_000: 55,
    200_000: 65,
    500_000: 75,
    1_000_000: 90,
    2_000_000: 100,
    5_000_000: 120,
    10_000_000: 140,
}

SDE1D_SAMPLING_BASE = {
    "sampling_defaults": {
        "terminal_time": 1.0,
        "reference_sampler": "milstein",
        "reference_steps": 50_000,
    },
    "sampling_configs": [
        # {"sampler": "euler"},
        {"sampler": "milstein"},
    ],
    "step_schedules": {
        "calibrated": {
            "euler": _EULER_CALIBRATED,
            "milstein": _MILSTEIN_CALIBRATED,
        },
    },
    "B_list": [
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
        500_000,
        # 1_000_000,
        # 2_000_000,
        # 5_000_000,
        # 10_000_000,
    ],
    "B1_list": [
        1_000,
        2_000,
        5_000,
        10_000,
        # 25_000,
        # 50_000,
        # 75_000,
        # 100_000,
        # 150_000,
        # 200_000,
        # 250_000,
        # 500_000,
    ],
    "free_B1_list": [100_000, 1_000_000],
    "baselines": ["fixed_N"],
}

SDE1D_METADATA_BASE = {
    "sigma_modes": [
        "pilot_tree",
        "independent",
    ],
    "pilot_m": 2.0,
    "independent_n2": 3,
    "reuse_flags": [True],
    "optimization_modes": ["monotone_fw"],
    "split_percentages_list": [list(split) for split in SDE1D_SPLITS],
    "num_base_samples": 1_000_000,
    "n_runs": 25,
    "n_parallel": 25,
}


__all__ = ["SDE1D_METADATA_BASE", "SDE1D_SAMPLING_BASE", "make_sde1d_x_grid"]
