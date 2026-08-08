from runners.sde.configs import (
    SDE_MMD_SELECTION,
    SDE_METADATA_BASE,
    SDE_SAMPLING_BASE,
)

_DEFAULT_CONFIG = {
    **SDE_SAMPLING_BASE,
    **SDE_METADATA_BASE,
    "runner_defaults": {
        **SDE_SAMPLING_BASE["runner_defaults"],
        "dimension": 2,
        "coupling_strength": 0.0,
        "terminal_time": 2.0,
    },
    "sampling_defaults": {
        "terminal_time": 2.0,
    },
    "reference_generation_config": {
        "method": "sde_terminal_samples",
        "sampler": "euler",
        "sampling_steps": 10_000,
        "terminal_time": 2.0,
    },
    "sampling_configs": [{"sampler": "euler"}],
    "phase1_query_params": {
        "subset_sizes": [1, 2],
    },
    "max_sampling_batch_size": 50_000,
    "comparison_mode": "true_samples",
    "description": "Coupled double-well overdamped Langevin adaptive splitting",
}

CONFIGS = {
    "default": _DEFAULT_CONFIG,
    "mmd": {
        **_DEFAULT_CONFIG,
        **SDE_MMD_SELECTION,
        "description": (
            "Coupled double-well overdamped Langevin adaptive splitting with "
            "target-space MMD"
        ),
    },
}


__all__ = ["CONFIGS"]
