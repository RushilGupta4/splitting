X_GRID = [round(-1.7 + 0.1 * i, 1) for i in range(35)]

_N500_1000 = {
    1_000_000: 500,
    2_500_000: 500,
    5_000_000: 500,
    10_000_000: 1000,
    20_000_000: 1000,
    50_000_000: 1000,
}

_SPLITS = [
    [0.5],
    [0.66, 0.33],
    [0.75, 0.5, 0.25],
]

_SAMPLING_BASE = {
    "sampling_configs": [
        {"sampler": "ddim", "eta": 1.0},
    ],
    "step_schedules": {
        "n500_1000": {
            "ddim": _N500_1000,
            "dpmpp_2m": _N500_1000,
        },
    },
    "baseline_step_schedules": {
        "n500_1000": {
            "dpmpp_2m": _N500_1000,
        },
    },
    "B_list": [
        1_000_000,
        2_500_000,
        5_000_000,
        10_000_000,
        20_000_000,
        50_000_000,
    ],
    "B1_list": [
        250_000,
        500_000,
        1_000_000,
        2_500_000,
        5_000_000,
    ],
    "free_B1_list": [
        250_000,
        500_000,
        1_000_000,
        2_500_000,
        5_000_000,
    ],
    "baselines": ["fixed_N", "dpmpp_2m"],
}

_METADATA_DEFAULTS = {
    "comparison_mode": "true_samples",
    "split_percentages_list": _SPLITS,
    "pilot_m": 2.0,
    "independent_n2": 5,
    "num_base_samples": 1_000_000,
    "n_runs": 200,
    "reuse_flags": [True],
    "optimization_modes": ["monotone"],
}

CONFIGS = {
    "default": {
        **_SAMPLING_BASE,
        **_METADATA_DEFAULTS,
        "description": "All methods, true_samples reference",
        "sigma_modes": ["pilot_tree", "independent"],
        "x_grid": X_GRID,
    },
    "joint_only": {
        **_SAMPLING_BASE,
        **_METADATA_DEFAULTS,
        "description": "Pilot-tree only, true_samples reference",
        "sigma_modes": ["pilot_tree"],
        "x_grid": X_GRID,
    },
    "independent_only": {
        **_SAMPLING_BASE,
        **_METADATA_DEFAULTS,
        "description": "Independent-mode only, true_samples reference",
        "sigma_modes": ["independent"],
        "x_grid": X_GRID,
    },
}
