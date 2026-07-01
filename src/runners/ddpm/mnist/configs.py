from runners.common_configs import crossfit_q_config

_N40_SCALE = {
    25_000: 40,
    50_000: 50,
    100_000: 63,
    250_000: 85,
    500_000: 110,
}

_N100_SCALE = {
    25_000: 100,
    50_000: 125,
    100_000: 160,
    250_000: 215,
    500_000: 270,
}

_SPLITS = [
    [0.75, 0.5, 0.25],
]

_SAMPLING_BASE = {
    "sampling_configs": [
        {"sampler": "ddim", "eta": 1.0},
    ],
    "step_schedules": {
        "n100_scale": {
            "ddim": _N100_SCALE,
        },
    },
    "baseline_step_schedules": {
        "n40_scale": {
            "ddim": _N40_SCALE,
        },
    },
    "B_list": [
        # 25_000,
        50_000,
        # 100_000,
        # 250_000,
        # 500_000,
    ],
    "B1_list": [
        5_000,
        10_000,
        # 25_000,
        # 50_000,
    ],
    "baselines": [
        "fixed_N",
        "ddim_eta0",
        "ddim_50_eta0",
    ],
}

_METADATA_DEFAULTS = {
    "comparison_mode": "mnist_dataset_samples",
    "reference_generation_config": {
        "method": "torchvision_mnist",
        "split": "train",
        "data_root": "data",
        "download": True,
        "seed": 0,
        "selection": "seeded_without_replacement",
        "transform": "to_tensor_flat_0_1_v1",
    },
    "split_percentages_list": _SPLITS,
    "optimization_modes": ["monotone", "monotone_cvar95"],
    "crossfit_q_mlp_losses": ["bce", "mse"],
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
