from runners.common_configs import crossfit_q_config

_MODEL_REFERENCE_SAMPLE_COUNT = 20_000

SUPPORTED_B = [
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
]
_N100_FIXED = {b: 100 for b in SUPPORTED_B}
_N500_FIXED = {b: 500 for b in SUPPORTED_B}
_N1000_FIXED = {b: 1000 for b in SUPPORTED_B}
_N100_SCALE = {50_000: 100, 100_000: 126, 250_000: 170, 500_000: 215}
_N200_SCALE = {50_000: 200, 100_000: 252, 250_000: 340, 500_000: 430, 1_000_000: 542}

_SPLITS = [
    # [0.75, 0.5, 0.25],
    # [0.05],
    [0.8, 0.6, 0.4, 0.2],
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
]

_MMD_METRIC_PARAMS = {
    "batch_size": 2500,
}

_SAMPLING_BASE = {
    "sampling_configs": [
        {"sampler": "ddpm", "timestep_spacing": "trailing"},
    ],
    "step_schedules": {
        # "n100_scale": {
        #     "ddpm": _N100_SCALE,
        # },
        "n200_scale": {
            "ddpm": _N200_SCALE,
        },
        # "n100_fixed": {
        #     "ddpm": _N100_FIXED,
        # },
        # "n500_fixed": {
        #     "ddpm": _N500_FIXED,
        # },
        # "n1000_fixed": {
        #     "ddpm": _N1000_FIXED,
        # },
    },
    "baseline_step_schedules": {},
    "B_list": [
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        # 2_000_000,
        # 5_000_000,
        # 10_000_000,
    ],
    "B1_list": [
        # 0.01,
        # 0.02,
        # 0.05,
        # "5,0.66",
        "10,0.66",
        # 10_000,
        # 25_000,
        # 50_000,
    ],
    "baselines": [
        "fixed_N",
        # "ddim_40_eta0",
        # "ddim_100_eta0",
    ],
}

_MODEL_REFERENCE = {
    "comparison_mode": "true_samples",
    "reference_generation_config": {
        "method": "hf_ddpm_scheduler",
        "sampler": "ddpm",
        "T": 1000,
        "sampling_steps": 1000,
        "timestep_spacing": "trailing",
        "seed": 0,
    },
}

_METADATA_DEFAULTS = {
    **_MODEL_REFERENCE,
    "metrics": ["mmd"],
    "primary_metric": "mmd",
    "metric_params": {
        "mmd": _MMD_METRIC_PARAMS,
    },
    "split_percentages_list": _SPLITS,
    "optimization_modes": ["monotone_cvar95"],
    "crossfit_q_mlp_losses": ["mse"],
    "num_base_samples": _MODEL_REFERENCE_SAMPLE_COUNT,
    "max_sampling_batch_size": 2500,
    "n_runs": 100,
}

_MMD_DEFAULTS = {
    **_METADATA_DEFAULTS,
    "metrics": ["mmd"],
    "primary_metric": "mmd",
    "metric_params": {
        "mmd": _MMD_METRIC_PARAMS,
    },
}

CONFIGS = {
    "default": {
        **_SAMPLING_BASE,
        **_METADATA_DEFAULTS,
        **crossfit_q_config(),
        "description": "CIFAR-10 HF DDPM splitting against the configured DDPM reference",
    },
    "mmd": {
        **_SAMPLING_BASE,
        **_MMD_DEFAULTS,
        **crossfit_q_config(),
        "description": "CIFAR-10 HF DDPM splitting with MMD against the configured DDPM reference",
    },
}
