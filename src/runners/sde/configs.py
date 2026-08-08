import numpy as np

from runners.common_configs import crossfit_q_config
from runners.sde.sampling import SDE_SAMPLERS

SPLIT_LISTS = [4, 9]
SDE_SPLITS = [np.round(np.arange(j, 0, -1) / (j + 1), 2).tolist() for j in SPLIT_LISTS]


def _sampling_configs() -> list[dict[str, str]]:
    return [{"sampler": sampler} for sampler in SDE_SAMPLERS]


_EULER_CALIBRATED = {
    10_000: 60,
    20_000: 75,
    50_000: 100,
    100_000: 130,
    200_000: 160,
    500_000: 220,
    1_000_000: 280,
    2_000_000: 350,
    5_000_000: 475,
    10_000_000: 600,
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

SDE_SAMPLING_BASE = {
    "runner_defaults": {
        "dimension": 2,
    },
    "sampling_defaults": {
        "terminal_time": 1.0,
    },
    "reference_generation_config": {
        "method": "sde_terminal_samples",
        "sampler": "euler",
        "sampling_steps": 20_000,
        "terminal_time": 1.0,
    },
    "sampling_configs": [
        {"sampler": "euler"},
        # {"sampler": "milstein"},
    ],
    "step_schedules": {
        "calibrated": {
            "euler": _EULER_CALIBRATED,
            "milstein": _MILSTEIN_CALIBRATED,
        },
    },
    "B_list": [
        100_000,
        200_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        # 10_000_000,
    ],
    "B1_list": [
        # 0.01,
        # 0.02,
        # 0.05,
        "5,0.66",
        # "10,0.66",
        # "500,0.20",
        # "500,0.1",
        # 5_000,
        # 10_000,
        # 25_000,
        # 50_000,
        # 75_000,
        # 100_000,
        # 150_000,
        # 200_000,
        # 250_000,
    ],
    "baselines": ["fixed_N"],
}

SDE_METADATA_BASE = {
    **crossfit_q_config(),
    "split_percentages_list": [list(split) for split in SDE_SPLITS],
    "optimization_modes": ["monotone_cvar95"],
    "crossfit_q_mlp_losses": ["mse"],
    "num_base_samples": 2_500_000,
    "n_runs": 25,
    "n_parallel": 25,
}

SDE_MMD_SELECTION = {
    "metrics": ["mmd"],
    "primary_metric": "mmd",
}

__all__ = ["SDE_MMD_SELECTION", "SDE_METADATA_BASE", "SDE_SAMPLING_BASE"]
