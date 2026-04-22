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


def sample_target_distribution(num_samples, device):
    """Sample from the hard-coded 2D Gaussian-mixture target."""
    spec = get_target_distribution_tensors(device=device)
    weights = spec["weights"]
    means = spec["means"]
    covariances = spec["covariances"]

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


def validate_split_params(split_points, split_sizes, T):
    """Validate split parameters.

    Args:
        split_points: List of timesteps where splitting occurs (must be decreasing)
        split_sizes: List of split factors at each split point
        T: Total diffusion timesteps

    Raises:
        ValueError: If parameters are invalid
    """
    if len(split_points) != len(split_sizes):
        raise ValueError(
            f"split_points and split_sizes must have same length, "
            f"got {len(split_points)} and {len(split_sizes)}"
        )

    if len(split_points) == 0:
        raise ValueError("split_points must have at least one element")

    for i, sp in enumerate(split_points):
        if sp <= 0 or sp >= T:
            raise ValueError(f"split_points[{i}]={sp} must be in range (0, {T})")

    for i in range(len(split_points) - 1):
        if split_points[i] <= split_points[i + 1]:
            raise ValueError(
                f"split_points must be strictly decreasing, "
                f"got split_points[{i}]={split_points[i]} <= split_points[{i+1}]={split_points[i+1]}"
            )

    for i, ss in enumerate(split_sizes):
        if ss < 1:
            raise ValueError(f"split_sizes[{i}]={ss} must be >= 1")


def compute_n1(B, T, split_points, split_sizes):
    r"""Compute N_1 (number of initial paths) given budget and parameters.

    With K split points, the total computational cost is:
        cost = N_1 * (T - split_points[0])
             + N_1 * split_sizes[0] * (split_points[0] - split_points[1])
             + N_1 * split_sizes[0] * split_sizes[1] * (split_points[1] - split_points[2])
             + ...
             + N_1 * prod(split_sizes) * split_points[-1]

    This simplifies to:
        cost = N_1 * cost_per_n1

    Args:
        B: Total computational budget (in denoising steps)
        T: Total diffusion timesteps
        split_points: List of timesteps where splitting occurs (must be decreasing)
        split_sizes: List of split factors at each split point

    Returns:
        Tuple of (n1, cost_per_n1, used_budget)
    """
    validate_split_params(split_points, split_sizes, T)

    cost_per_n1 = T - split_points[0]
    cumulative_split = 1
    for i in range(len(split_points)):
        cumulative_split *= split_sizes[i]
        if i + 1 < len(split_points):
            segment_length = split_points[i] - split_points[i + 1]
        else:
            segment_length = split_points[i]
        cost_per_n1 += cumulative_split * segment_length

    n1 = B // cost_per_n1
    used_budget = n1 * cost_per_n1
    return n1, cost_per_n1, used_budget


def compute_total_samples(n1, split_sizes):
    """Compute total number of final samples."""
    total = n1
    for ss in split_sizes:
        total *= ss
    return total


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


def add_common_args(parser):
    """Add common arguments shared between inference scripts."""
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=2_500_000,
        help="Computational budget (in denoising steps)",
    )
    parser.add_argument(
        "--n_runs", type=int, default=100, help="Number of runs for statistics"
    )
    parser.add_argument(
        "--runs_batch_size",
        type=int,
        default=100,
        help="Number of runs to process in parallel per batch",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=1000,
        help="Total diffusion steps (for noise schedule)",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="Number of DDIM sampling steps (default: T)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help=r"DDIM \eta parameter (0=deterministic, 1=DDPM)",
    )
    parser.add_argument(
        "--split_percentages",
        type=str,
        default="0.5",
        help=(
            "Comma-separated split percentages (strictly decreasing), "
            "e.g. '0.666,0.333'. Each split point is resolved as round(steps * percentage)."
        ),
    )
    parser.add_argument(
        "--split_sizes",
        type=str,
        default="1",
        help=(
            "Comma-separated list of split sizes (same length as split_percentages), "
            "e.g. '2,2,4'"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def parse_split_args(args):
    """Parse split_percentages and split_sizes from comma-separated strings."""
    split_percentages = [
        float(x.strip()) for x in args.split_percentages.split(",") if x.strip()
    ]
    validate_split_percentages(split_percentages)
    split_sizes = [int(x.strip()) for x in args.split_sizes.split(",")]
    return split_percentages, split_sizes
