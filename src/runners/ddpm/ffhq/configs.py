from copy import deepcopy

from runners.common_configs import (
    OPTIMIZATION_MODES_WITH_LEARNED_C,
    crossfit_q_config,
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
    "metrics": ["mmd"],
    "primary_metric": "mmd",
    "metric_params": {"mmd": {"batch_size": 256}},
    "split_percentages_list": split_schedules(),
    "optimization_modes": OPTIMIZATION_MODES_WITH_LEARNED_C,
    "crossfit_q_mlp_losses": ["mse"],
    "num_base_samples": 20_000,
    "max_sampling_batch_size": 2500,
    "n_runs": 100,
    **crossfit_q_config(),
    "description": "FFHQ LDM with decoded-image MMD against a full DDPM reference",
}

CONFIGS = {name: deepcopy(_DEFAULT) for name in ("default", "mmd")}
