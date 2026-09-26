"""Shared SDE sweep configuration.

Per-case modules choose exactly one thing: the terminal time (plus a human
description). Dimension, sampler, step schedules, budget grid and
reference-generation settings are standard across every SDE case and live here.
"""

from copy import deepcopy

from runners.common_configs import (
    OPTIMIZATION_MODES_WITH_LEARNED_C,
    mmd_sibling_config,
    split_schedules,
)

SDE_SPLITS = split_schedules()

SDE_DIMENSION = 2
SDE_SAMPLER = "euler"
SDE_COMPARISON_MODE = "true_samples"
SDE_REFERENCE_METHOD = "sde_terminal_samples"
SDE_REFERENCE_SAMPLER = "euler"
SDE_REFERENCE_SAMPLING_STEPS = 20_000
SDE_MAX_SAMPLING_BATCH_SIZE = 50_000

_EULER_CALIBRATED = {
    10_000: 60,
    20_000: 75,
    50_000: 100,
    100_000: 130,
    200_000: 160,
    500_000: 220,
    1_000_000: 280,
    2_000_000: 350,
    5_000_000: 475,
    10_000_000: 600,
}

# Everything here is independent of terminal_time.
_SAMPLING_BASE = {
    "sampling_configs": [
        {"sampler": SDE_SAMPLER},
    ],
    "step_schedules": {
        "calibrated": {SDE_SAMPLER: _EULER_CALIBRATED},
    },
    "B_list": [
        100_000,
        200_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
    ],
    "B1_list": [
        "5,0.66",
    ],
    "baselines": [
        "fixed_N",
        "uniform_c",
    ],
    "max_sampling_batch_size": SDE_MAX_SAMPLING_BATCH_SIZE,
}

_METADATA_BASE = {
    "split_percentages_list": [list(split) for split in SDE_SPLITS],
    "optimization_modes": OPTIMIZATION_MODES_WITH_LEARNED_C,
    "crossfit_q_mlp_losses": ["mse"],
    "num_base_samples": 2_500_000,
    "n_runs": 25,
    "n_parallel": 25,
}

_MMD_SELECTION = {
    "metrics": ["mmd"],
    "primary_metric": "mmd",
}


def sde_case_configs(*, terminal_time: float, description: str) -> dict[str, dict]:
    """Build the {"default", "mmd"} CONFIGS dict for one SDE case.

    ``terminal_time`` is written into all three slots that consume it, so they
    cannot drift apart:

    * ``runner_defaults`` -> ``SDERunner(terminal_time=...)``
    * ``reference_generation_config`` -> hard-checked against the runner in
      ``SDERunner.normalize_reference_generation_config``
    * ``sampling_defaults`` -> merged into every ``sampling_configs`` entry by
      ``BaseRunner.get_config``, and never cross-checked against the other two
    """
    terminal_time = float(terminal_time)
    if not terminal_time > 0.0:
        raise ValueError(f"terminal_time must be positive, got {terminal_time!r}")

    default_config = {
        **_SAMPLING_BASE,
        **_METADATA_BASE,
        "comparison_mode": SDE_COMPARISON_MODE,
        "runner_defaults": {
            "dimension": SDE_DIMENSION,
            "terminal_time": terminal_time,
        },
        "sampling_defaults": {"terminal_time": terminal_time},
        "reference_generation_config": {
            "method": SDE_REFERENCE_METHOD,
            "sampler": SDE_REFERENCE_SAMPLER,
            "sampling_steps": SDE_REFERENCE_SAMPLING_STEPS,
            "terminal_time": terminal_time,
        },
        "description": str(description),
    }

    return {
        "default": deepcopy(default_config),
        "mmd": deepcopy(
            mmd_sibling_config(
                {
                    **default_config,
                    **_MMD_SELECTION,
                    "description": f"{description} with target-space MMD",
                }
            )
        ),
    }


__all__ = ["sde_case_configs"]
