from runners.common_configs import crossfit_q_config
from runners.edm.configs import _DPMPP_2S_PARAMS, _EDM_PARAMS

_SAMPLING_DEFAULTS = {
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "rho": 3.0,
}


_N40_SCALE = {
    100_000: 40,
    200_000: 45,
    500_000: 55,
    1_000_000: 63,
    2_000_000: 73,
    5_000_000: 87,
}

_SAMPLING_BASE = {
    "sampling_defaults": _SAMPLING_DEFAULTS,
    "step_schedules": {
        "n40_scale": {
            "edm_stochastic": _N40_SCALE,
            "dpmpp_2s": _N40_SCALE,
        },
    },
    "baseline_step_schedules": {
        "n40_scale": {
            "edm_stochastic": _N40_SCALE,
            "dpmpp_2s": _N40_SCALE,
        },
    },
    "sampling_configs": [
        *[
            {
                "sampler": "edm_stochastic",
                "sampler_params": {
                    **_EDM_PARAMS,
                    "S_churn": i,
                },
            }
            for i in [40]
        ],
    ],
    "B_list": [
        100_000,
        200_000,
        500_000,
        1_000_000,
        2_000_000,
        # 5_000_000,
    ],
    "B1_list": [
        # 0.01,
        # 0.02,
        # 0.05,
        "10,0.66",
        # "500,0.20",
        # "500,0.1",
        # 10_000,
        # 25_000,
        # 50_000,
        # 75_000,
    ],
    "baselines": [
        "fixed_N",
        # "edm_stochastic_churn0",
        # "dpmpp_2s_churn2.5",
    ],
}

_DEFAULT_CONFIG = {
    **_SAMPLING_BASE,
    **crossfit_q_config(),
    "description": "Adaptive splitting vs deterministic EDM baselines on EDM (2D GMM)",
    "comparison_mode": "true_samples",
    "reference_generation_config": {
        "method": "target_samples",
    },
    "split_percentages_list": [
        [0.8, 0.6, 0.4, 0.2],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    ],
    "optimization_modes": ["monotone_cvar95"],
    "crossfit_q_mlp_losses": ["mse"],
    "num_base_samples": 5_000_000,
    "n_runs": 50,
}

CONFIGS = {
    "default": _DEFAULT_CONFIG,
    "mmd": {
        **_DEFAULT_CONFIG,
        "metrics": ["mmd"],
        "primary_metric": "mmd",
        "description": (
            "Adaptive splitting with target-space MMD against target samples on EDM "
            "(2D GMM)"
        ),
    },
}
