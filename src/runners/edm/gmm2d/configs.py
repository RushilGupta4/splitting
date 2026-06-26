from runners.edm.configs import _DPMPP_2S_PARAMS, _EDM_PARAMS

X_GRID = [round(-1.7 + 0.1 * i, 1) for i in range(35)]

_SAMPLING_DEFAULTS = {
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "rho": 7.0,
}

_N100_SCALE = {
    100_000: 100,
    200_000: 110,
    500_000: 122,
    1_000_000: 132,
    2_000_000: 144,
    5_000_000: 155,
}
_N100_FIXED = {
    100_000: 100,
    200_000: 100,
    500_000: 100,
    1_000_000: 100,
    2_000_000: 100,
    5_000_000: 100,
}

_N40_SCALE = {
    100_000: 40,
    200_000: 44,
    500_000: 49,
    1_000_000: 53,
    2_000_000: 57,
    5_000_000: 62,
}
_N40_FIXED = {
    100_000: 40,
    200_000: 40,
    500_000: 40,
    1_000_000: 40,
    2_000_000: 40,
    5_000_000: 40,
}

_SAMPLING_BASE = {
    "sampling_defaults": _SAMPLING_DEFAULTS,
    "step_schedules": {
        "n40_scale": {
            "edm_stochastic": _N40_SCALE,
            "dpmpp_2s": _N40_SCALE,
        },
        # "n40_fixed": {
        #     "edm_stochastic": _N40_FIXED,
        #     "dpmpp_2s": _N40_FIXED,
        # },
    },
    "baseline_step_schedules": {
        # "n40_scale": {
        #     "edm_stochastic": _N40_SCALE,
        #     "dpmpp_2s": _N40_SCALE,
        # },
        "n40_fixed": {
            "edm_stochastic": _N40_FIXED,
            "dpmpp_2s": _N40_FIXED,
        },
    },
    "sampling_configs": [
        *[
            {
                "sampler": "dpmpp_2s",
                "sampler_params": {
                    **_DPMPP_2S_PARAMS,
                    "stochastic_churn_rate": i,
                },
            }
            for i in [20]
        ],
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
        # 2_000_000,
        # 5_000_000,
    ],
    "B1_list": [10_000, 25_000, 50_000],
    "free_B1_list": [500_000],
    "baselines": [
        "fixed_N",
        "edm_stochastic_churn0",
        "dpmpp_2s_churn2.5",
    ],
}

_METADATA_DEFAULTS = {
    "pilot_m": 2.0,
    "independent_n2": 5,
    "optimization_modes": ["monotone"],
}

CONFIGS = {
    "default": {
        **_SAMPLING_BASE,
        **_METADATA_DEFAULTS,
        "description": "Adaptive splitting vs deterministic EDM baselines on EDM (2D GMM)",
        "comparison_mode": "true_samples",
        "sigma_modes": ["pilot_tree"],
        "reuse_flags": [True],
        "split_percentages_list": [
            # [0.46],
            [0.7, 0.45],
        ],
        "num_base_samples": 5_000_000,
        "n_runs": 50,
        "x_grid": X_GRID,
    },
}
