from runners.common_configs import (
    OPTIMIZATION_MODES,
    crossfit_q_config,
    split_schedules,
)

_MODEL_REFERENCE_SAMPLE_COUNT = 20_000
_N200_SCALE = {50_000: 200, 100_000: 252, 200_000: 317, 500_000: 430, 1_000_000: 542}
_SPLITS = split_schedules()

_MMD_METRIC_PARAMS = {
    "batch_size": 2500,
}

_SAMPLING_BASE = {
    "sampling_configs": [
        {"sampler": "ddpm", "timestep_spacing": "trailing"},
    ],
    "step_schedules": {
        "n200_scale": {
            "ddpm": _N200_SCALE,
        },
    },
    "baseline_step_schedules": {},
    "B_list": [
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
    ],
    "B1_list": [
        "10,0.66",
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
    "optimization_modes": OPTIMIZATION_MODES,
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
