def crossfit_q_config(num_queries: int | None = None, losses=None) -> dict:
    """Return a config fragment for CrossFit-Q sweeps.

    Common MLP/query defaults live in adaptive.py. Pass num_queries or losses
    only for runner-specific sweep overrides.
    """
    cfg = {
        "sigma_modes": ["crossfit_q", "independent"],
        "joint_m": 2.0,
    }
    if num_queries is not None:
        cfg["grid_free_params"] = {
            "num_queries": int(num_queries),
        }
    if losses is not None:
        if isinstance(losses, str):
            losses = [losses]
        cfg["crossfit_q_mlp_losses"] = [str(loss) for loss in losses]
    return cfg


__all__ = ["crossfit_q_config"]
