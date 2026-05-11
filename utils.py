"""Shared utilities for DDPM/DDIM splitting experiments."""

import copy

import numpy as np
import torch


TARGET_DISTRIBUTION = {
    "name": "gaussian_mixture_2d",
    "weights": [0.5, 0.5],
    "means": [[-1.0, 1.0], [1.0, -1.0]],
    "covariances": [
        [[0.20, 0.05], [0.05, 0.30]],
        [[0.30, -0.08], [-0.08, 0.15]],
    ],
}


def get_target_distribution_spec():
    """Return a copy of the hard-coded 2D Gaussian-mixture target."""
    return copy.deepcopy(TARGET_DISTRIBUTION)


def get_target_distribution_tensors(device=None, dtype=torch.float32):
    """Return the hard-coded 2D Gaussian-mixture target as tensors."""
    spec = get_target_distribution_spec()
    return {
        "name": spec["name"],
        "weights": torch.tensor(spec["weights"], device=device, dtype=dtype),
        "means": torch.tensor(spec["means"], device=device, dtype=dtype),
        "covariances": torch.tensor(spec["covariances"], device=device, dtype=dtype),
    }


def compute_target_stats(device=None, dtype=torch.float32):
    """Compute exact mixture mean, coordinatewise std, and covariance."""
    spec = get_target_distribution_tensors(device=device, dtype=dtype)
    weights = spec["weights"]
    means = spec["means"]
    covariances = spec["covariances"]

    mean = (weights[:, None] * means).sum(dim=0)
    second_moment = (
        weights[:, None, None]
        * (covariances + means[:, :, None] * means[:, None, :])
    ).sum(dim=0)
    covariance = second_moment - mean[:, None] * mean[None, :]
    std = torch.sqrt(torch.diag(covariance))
    return mean, std, covariance


def sample_target_spec(target_spec, num_samples, device, dtype=torch.float32):
    """Sample from a 2D Gaussian-mixture target specification."""
    weights = torch.as_tensor(target_spec["weights"], device=device, dtype=dtype)
    means = torch.as_tensor(target_spec["means"], device=device, dtype=dtype)
    covariances = torch.as_tensor(
        target_spec["covariances"], device=device, dtype=dtype
    )

    component_ids = torch.multinomial(weights, num_samples, replacement=True)
    samples = torch.empty(num_samples, means.shape[1], device=device, dtype=means.dtype)

    for component_idx in range(weights.numel()):
        mask = component_ids == component_idx
        count = int(mask.sum().item())
        if count == 0:
            continue
        distribution = torch.distributions.MultivariateNormal(
            loc=means[component_idx], covariance_matrix=covariances[component_idx]
        )
        samples[mask] = distribution.sample((count,))

    return samples


def sample_target_distribution(num_samples, device):
    """Sample from the hard-coded 2D Gaussian-mixture target."""
    return sample_target_spec(get_target_distribution_spec(), num_samples, device)


def _coerce_stats_tensor(values, reference):
    return torch.as_tensor(values, device=reference.device, dtype=reference.dtype)


def normalize(samples, data_mean, data_std):
    """Normalize samples to coordinatewise mean 0 and std 1."""
    mean = _coerce_stats_tensor(data_mean, samples)
    std = _coerce_stats_tensor(data_std, samples)
    return (samples - mean) / std


def denormalize(samples, data_mean, data_std):
    """Denormalize samples from coordinatewise mean 0 and std 1."""
    mean = _coerce_stats_tensor(data_mean, samples)
    std = _coerce_stats_tensor(data_std, samples)
    return samples * std + mean


def get_checkpoint_target_spec(checkpoint):
    """Get target metadata from a checkpoint, defaulting to the hard-coded target."""
    return checkpoint.get("target_spec", get_target_distribution_spec())


def get_checkpoint_normalization_stats(checkpoint, device=None, dtype=torch.float32):
    """Get normalization stats from a checkpoint or the hard-coded target."""
    data_mean = checkpoint.get("data_mean")
    data_std = checkpoint.get("data_std")
    if data_mean is None or data_std is None:
        mean, std, _ = compute_target_stats(device=device, dtype=dtype)
        return mean, std
    return (
        torch.as_tensor(data_mean, device=device, dtype=dtype),
        torch.as_tensor(data_std, device=device, dtype=dtype),
    )


def infer_input_dim_from_checkpoint(checkpoint):
    """Infer model input dimension from stored normalization stats or target spec."""
    if "data_mean" in checkpoint and checkpoint["data_mean"] is not None:
        return int(torch.as_tensor(checkpoint["data_mean"]).numel())

    target_spec = get_checkpoint_target_spec(checkpoint)
    means = target_spec.get("means")
    if means is not None:
        return len(means[0])

    return 2


def get_checkpoint_T(checkpoint):
    """Infer the diffusion horizon T from checkpoint metadata."""
    model_args = checkpoint.get("args", {})
    return int(model_args.get("T", 1000))


def mixture_lower_orthant_cdf(points, target_spec=None):
    """Evaluate the exact lower-orthant CDF of the 2D Gaussian mixture."""
    from scipy.stats import multivariate_normal

    if target_spec is None:
        target_spec = get_target_distribution_spec()

    points_np = np.asarray(points, dtype=float)
    points_2d = np.atleast_2d(points_np)
    cdf = np.zeros(points_2d.shape[0], dtype=float)

    for weight, mean, covariance in zip(
        target_spec["weights"], target_spec["means"], target_spec["covariances"]
    ):
        rv = multivariate_normal(mean=np.asarray(mean), cov=np.asarray(covariance))
        cdf += float(weight) * np.asarray(rv.cdf(points_2d), dtype=float)

    if points_np.ndim == 1:
        return float(cdf[0])
    return cdf


def mixture_pdf(points, target_spec=None):
    """Evaluate the PDF of the 2D Gaussian mixture."""
    from scipy.stats import multivariate_normal

    if target_spec is None:
        target_spec = get_target_distribution_spec()

    points_np = np.asarray(points, dtype=float)
    points_2d = np.atleast_2d(points_np)
    pdf = np.zeros(points_2d.shape[0], dtype=float)

    for weight, mean, covariance in zip(
        target_spec["weights"], target_spec["means"], target_spec["covariances"]
    ):
        rv = multivariate_normal(mean=np.asarray(mean), cov=np.asarray(covariance))
        pdf += float(weight) * np.asarray(rv.pdf(points_2d), dtype=float)

    if points_np.ndim == 1:
        return float(pdf[0])
    return pdf


def validate_split_percentages(split_percentages):
    """Validate split percentages expressed as fractions of sampling steps."""
    if len(split_percentages) == 0:
        raise ValueError("split_percentages must have at least one element")

    for i, percentage in enumerate(split_percentages):
        if percentage <= 0.0 or percentage >= 1.0:
            raise ValueError(
                f"split_percentages[{i}]={percentage} must be in the open interval (0, 1)"
            )

    for i in range(len(split_percentages) - 1):
        if split_percentages[i] <= split_percentages[i + 1]:
            raise ValueError(
                "split_percentages must be strictly decreasing, "
                f"got split_percentages[{i}]={split_percentages[i]} <= split_percentages[{i+1}]={split_percentages[i+1]}"
            )


def parse_step_eta_pairs(raw: str):
    pairs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Each step_eta pair must be 'steps:eta', got '{item}'")
        steps_str, eta_str = item.split(":", 1)
        pairs.append((int(steps_str), float(eta_str)))
    if not pairs:
        raise ValueError("step_eta_pairs must contain at least one pair")
    return pairs


def parse_split_percentages(split_percentages_str: str):
    split_percentages = [
        float(x.strip()) for x in split_percentages_str.split(",") if x.strip()
    ]
    validate_split_percentages(split_percentages)
    return split_percentages


def parse_x_grid(x_grid_str: str):
    x_grid = [float(x.strip()) for x in x_grid_str.split(",") if x.strip()]
    if not x_grid:
        raise ValueError("x_grid must have at least one threshold")
    return x_grid

