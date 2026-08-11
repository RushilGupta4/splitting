def crossfit_q_config(num_queries: int | None = None, losses=None) -> dict:
    """Return a config fragment for the phase-1 CrossFit-Q estimator.

    Query and MLP defaults live in adaptive.py; pass num_queries or losses only
    for runner-specific sweep overrides.
    """
    cfg: dict = {}
    if num_queries is not None:
        cfg["query_params"] = {"num_queries": int(num_queries)}
    if losses is not None:
        if isinstance(losses, str):
            losses = [losses]
        cfg["crossfit_q_mlp_losses"] = [str(loss) for loss in losses]
    return cfg


__all__ = ["crossfit_q_config"]
