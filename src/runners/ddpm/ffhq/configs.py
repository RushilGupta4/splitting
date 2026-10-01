from copy import deepcopy

from runners.common_configs import (
    OPTIMIZATION_MODES_WITH_LEARNED_C,
    mmd_sibling_config,
    split_schedules,
)

_DEFAULT = {
    "sampling_configs": [{"sampler": "ddpm", "timestep_spacing": "trailing"}],
    # round(200 * (B / 50_000) ** (1 / 3)).
    "step_schedules": {
        "n200_scale": {"ddpm": {50_000: 200, 100_000: 252, 200_000: 317, 500_000: 430}},
    },
    "baseline_step_schedules": {},
    "B_list": [
        50_000,
        100_000,
        200_000,
        500_000,
    ],
    "B1_list": ["10,0.66"],
    "baselines": ["fixed_N"],
    "comparison_mode": "true_samples",
    "reference_generation_config": {
        "method": "ldm_ddpm_samples",
        "sampler": "ddpm",
        "T": 1000,
        "sampling_steps": 1000,
        "timestep_spacing": "trailing",
    },
    # The allocation is designed in the latent (the denoiser's own); samples are scored
    # in Inception and DINOv2 embeddings. Any of "mmd" (latent), "mmd_pixel",
    # "mmd_inception" and "mmd_dino" can be listed.
    "metrics": ["mmd_inception", "mmd_dino"],
    "primary_metric": "mmd_inception",
    "metric_params": {"mmd": {"batch_size": 256}},
    "batching": {"phase1": 2500, "phase2": 2500, "max_paths_in_flight": None, "scoring": 64},
    "split_percentages_list": split_schedules(),
    "optimization_modes": OPTIMIZATION_MODES_WITH_LEARNED_C,
    "num_base_samples": 20_000,
    "max_sampling_batch_size": 2500,
    "n_runs": 100,
    "description": "FFHQ LDM with Inception and DINOv2 MMD against a full DDPM reference",
}

CONFIGS = {name: mmd_sibling_config(deepcopy(_DEFAULT)) for name in ("default", "mmd")}
# Per-run timings at one budget (run with compare.py --timing).
CONFIGS["timing"] = {
    **deepcopy(CONFIGS["mmd"]),
    "B_list": [200_000],
    "n_runs": 10,
    "description": "FFHQ per-run stage timings at B=200k",
}
