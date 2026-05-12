X_GRID = [
    -1.7,
    -1.6,
    -1.5,
    -1.4,
    -1.3,
    -1.2,
    -1.1,
    -1.0,
    -0.9,
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
]

_SAMPLING_CONFIGS = [
    {"sampling_steps": 1000, "eta": 1.0},
    {"sampling_steps": 500, "eta": 1.0},
]

_SPLITS = [
    [0.5],
    [0.66, 0.33],
    [0.75, 0.5, 0.25],
    [0.8, 0.6, 0.4, 0.2],
]

_COMMON = {
    "sampling_configs": _SAMPLING_CONFIGS,
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
    "baselines": ["fixed_N", "dpmpp_2m_40"],
    "split_percentages_list": _SPLITS,
    "observable_config": X_GRID,
    "independent_n2": 5,
    "num_base_samples": 1_000_000,
    "n_runs": 200,
}

CONFIGS = {
    "default": {
        **_COMMON,
        "description": "All methods, true_samples reference",
        "comparison_mode": "true_samples",
        "sigma_modes": ["pilot_tree", "independent"],
        "reuse_flags": [True],
    },
    "joint_only": {
        **_COMMON,
        "description": "Pilot-tree only, true_samples reference",
        "comparison_mode": "true_samples",
        "sigma_modes": ["pilot_tree"],
        "reuse_flags": [True],
    },
    "independent_only": {
        **_COMMON,
        "description": "Independent-mode only, true_samples reference",
        "comparison_mode": "true_samples",
        "sigma_modes": ["independent"],
        "reuse_flags": [True],
    },
}
