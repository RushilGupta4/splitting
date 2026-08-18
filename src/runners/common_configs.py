import numpy as np

_BASE_SPLIT_SCHDULES = []
SPLIT_COUNTS = [4, 9, 19]
OPTIMIZATION_MODES = ["monotone", "monotone_cvar95"]
OPTIMIZATION_MODES = ["monotone"]

# N_i = c everywhere. Cost grows like c^j, so dense schedules only admit c near 1.
UNIFORM_C_BY_SPLIT_COUNT = {4: (1.1, 1.25, 1.5), 9: (1.1, 1.25), 19: (1.1,)}


def uniform_c_values(num_splits: int) -> tuple[float, ...]:
    return UNIFORM_C_BY_SPLIT_COUNT.get(int(num_splits), ())


def split_schedules(counts=SPLIT_COUNTS) -> list[list[float]]:
    """Evenly spaced split percentages for each requested number of split points.

    ``j`` splits give ``[j/(j+1), ..., 1/(j+1)]`` -- fractions of the remaining
    steps, strictly decreasing as ``validate_split_percentages`` requires.
    """
    return _BASE_SPLIT_SCHDULES + [
        np.round(np.arange(j, 0, -1) / (j + 1), 2).tolist() for j in counts
    ]


def crossfit_q_config(
    num_queries: int | None = None,
    losses=None,
    folds: int | None = None,
) -> dict:
    """Return a config fragment for the phase-1 CrossFit-Q estimator.

    Query and MLP defaults live in adaptive.py; pass values only for
    runner-specific sweep overrides.
    """
    cfg: dict = {}
    if num_queries is not None:
        cfg["query_params"] = {"num_queries": int(num_queries)}
    if losses is not None:
        if isinstance(losses, str):
            losses = [losses]
        cfg["crossfit_q_mlp_losses"] = [str(loss) for loss in losses]
    if folds is not None:
        cfg["crossfit_q_folds"] = int(folds)
    return cfg


__all__ = [
    "OPTIMIZATION_MODES",
    "SPLIT_COUNTS",
    "UNIFORM_C_BY_SPLIT_COUNT",
    "crossfit_q_config",
    "split_schedules",
    "uniform_c_values",
]
