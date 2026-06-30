from runners.common_configs import crossfit_q_config

_N40_SCALE = {
    25_000: 40,
    50_000: 50,
    100_000: 63,
    250_000: 85,
}
_N40_FIXED = {
    25_000: 40,
    50_000: 40,
    100_000: 40,
    250_000: 40,
}

_SPLITS = [
    [0.75, 0.5, 0.25],
]

_SAMPLING_BASE = {
    "sampling_configs": [
        {"sampler": "ddim", "eta": 1.0},
    ],
    "step_schedules": {
        "n40_scale": {
            "ddim": _N40_SCALE,
        },
    },
    "baseline_step_schedules": {
        "n40_scale": {
            "dpmpp_2m": _N40_SCALE,
        },
    },
    "B_list": [
        25_000,
        50_000,
        # 100_000,
        # 250_000,
    ],
    "B1_list": [
        5_000,
        10_000,
        # 25_000,
    ],
    "baselines": [
        "fixed_N",
        # "dpmpp_2m",
    ],
}

_METADATA_DEFAULTS = {
    "comparison_mode": "true_samples",
    "reference_generation_config": {
        "method": "hf_ddpm_scheduler",
        "T": 1000,
        "sampling_steps": 1000,
    },
    "split_percentages_list": _SPLITS,
    "num_base_samples": 25_000,
    "n_runs": 200,
}

CONFIGS = {
    "default": {
        **_SAMPLING_BASE,
        **_METADATA_DEFAULTS,
        **crossfit_q_config(),
        "description": "All methods, true_samples reference",
    },
}
