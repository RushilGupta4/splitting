from runners.common_configs import (
    OPTIMIZATION_MODES_WITH_LEARNED_C,
    mmd_sibling_config,
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
    ],
    "B1_list": [
        "10,0.66",
    ],
    "baselines": [
        "fixed_N",
        # "uniform_c",
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

# The allocation is designed in pixel space (the denoiser's own); samples are scored
# in Inception and DINOv2 embeddings. Any of "mmd" (pixels), "mmd_inception" and
# "mmd_dino" can be listed.
_MMD_DEFAULTS = {
    **_MODEL_REFERENCE,
    "metrics": ["mmd_inception", "mmd_dino"],
    "primary_metric": "mmd_inception",
    "metric_params": {
        "mmd": _MMD_METRIC_PARAMS,
    },
    "split_percentages_list": _SPLITS,
    "optimization_modes": OPTIMIZATION_MODES_WITH_LEARNED_C,
    "num_base_samples": _MODEL_REFERENCE_SAMPLE_COUNT,
    "max_sampling_batch_size": 2500,
    "batching": {"phase1": 2500, "phase2": 2500, "max_paths_in_flight": None, "scoring": 500},
    "n_runs": 100,
}

_MMD = mmd_sibling_config(
    {
        **_SAMPLING_BASE,
        **_MMD_DEFAULTS,
        "description": "CIFAR-10 HF DDPM splitting with Inception and DINOv2 MMD",
    }
)

CONFIGS = {
    "default": _MMD,
    "mmd": _MMD,
    # Per-run timings at one budget (run with compare.py --timing).
    "timing": {
        **_MMD,
        "B_list": [200_000],
        "n_runs": 10,
        "description": "CIFAR-10 per-run stage timings at B=200k",
    },
}
