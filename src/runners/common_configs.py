def crossfit_q_config(num_queries: int | None = None) -> dict:
    """Return a config fragment for CrossFit-Q sweeps.

    Common MLP/query defaults live in adaptive.py. Pass num_queries only for
    runner-specific query-count overrides.
    """
    cfg = {
        "sigma_modes": ["crossfit_q"],
    }
    if num_queries is not None:
        cfg["grid_free_params"] = {
            "num_queries": int(num_queries),
        }
    return cfg


__all__ = ["crossfit_q_config"]
