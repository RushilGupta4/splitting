import numpy as np

SPLIT_COUNTS = [9, 19, 39]
OPTIMIZATION_MODES_WITH_LEARNED_C = ["monotone", "learned_c"]

# N_i = c everywhere
UNIFORM_C_BY_SPLIT_COUNT = {
    9: (1.1, 1.15, 1.25, 1.5),
    19: (1.1, 1.15, 1.2),
    39: (1.05, 1.1),
}


def uniform_c_values(num_splits: int) -> tuple[float, ...]:
    return UNIFORM_C_BY_SPLIT_COUNT.get(int(num_splits), ())


def split_schedules(counts=SPLIT_COUNTS) -> list[list[float]]:
    """Evenly spaced split percentages for each requested number of split points.

    ``j`` splits give ``[j/(j+1), ..., 1/(j+1)]`` -- fractions of the remaining
    steps, strictly decreasing as ``validate_split_percentages`` requires.
    """
    return [np.round(np.arange(j, 0, -1) / (j + 1), 2).tolist() for j in counts]


KS_PHASE1_KEYS = frozenset(
    {
        "query_params",
        "crossfit_q_folds",
        "crossfit_q_mlp_params",
        "crossfit_q_mlp_losses",
        "crossfit_q_mlp_run_parallelism",
    }
)


def mmd_sibling_config(cfg: dict, **phase1) -> dict:
    """``cfg`` with the MMD Phase 1: the crossfit query keys are dropped and the sibling pilot selected."""
    out = {key: value for key, value in cfg.items() if key not in KS_PHASE1_KEYS}
    out["phase1"] = {"estimator": "mmd_sibling", **phase1}
    return out
