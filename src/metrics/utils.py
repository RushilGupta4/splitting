"""Sample coercion shared by the metric implementations and by the runners."""

import numpy as np
import torch


def coerce_samples_np(samples, expected_dim: int | None = None) -> np.ndarray:
    samples_np = (
        samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    )
    samples_np = np.asarray(samples_np, dtype=float)
    if samples_np.ndim == 0:
        samples_np = samples_np.reshape(1, 1)
    elif samples_np.ndim == 1:
        samples_np = samples_np.reshape(-1, 1)
    elif samples_np.ndim > 2:
        samples_np = samples_np.reshape(-1, samples_np.shape[-1])
    if samples_np.ndim != 2:
        raise ValueError(f"Expected samples of shape [N, D], got {samples_np.shape}")
    if expected_dim is not None and samples_np.shape[1] != int(expected_dim):
        raise ValueError(
            f"Expected samples of shape [N, {int(expected_dim)}], got {samples_np.shape}"
        )
    return samples_np
