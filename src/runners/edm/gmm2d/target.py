from __future__ import annotations

import copy

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
    return copy.deepcopy(TARGET_DISTRIBUTION)


def compute_target_stats(device=None, dtype=torch.float32):
    weights = torch.tensor(TARGET_DISTRIBUTION["weights"], device=device, dtype=dtype)
    means = torch.tensor(TARGET_DISTRIBUTION["means"], device=device, dtype=dtype)
    covariances = torch.tensor(
        TARGET_DISTRIBUTION["covariances"], device=device, dtype=dtype
    )

    mean = (weights[:, None] * means).sum(dim=0)
    second_moment = (
        weights[:, None, None]
        * (covariances + means[:, :, None] * means[:, None, :])
    ).sum(dim=0)
    covariance = second_moment - mean[:, None] * mean[None, :]
    std = torch.sqrt(torch.diag(covariance))
    return mean, std, covariance


def sample_target_spec(target_spec, num_samples, device, dtype=torch.float32):
    weights = torch.as_tensor(target_spec["weights"], device=device, dtype=dtype)
    means = torch.as_tensor(target_spec["means"], device=device, dtype=dtype)
    covariances = torch.as_tensor(
        target_spec["covariances"],
        device=device,
        dtype=dtype,
    )

    component_ids = torch.multinomial(weights, num_samples, replacement=True)
    samples = torch.empty(num_samples, means.shape[1], device=device, dtype=means.dtype)

    for component_idx in range(weights.numel()):
        mask = component_ids == component_idx
        count = int(mask.sum().item())
        if count == 0:
            continue
        distribution = torch.distributions.MultivariateNormal(
            loc=means[component_idx],
            covariance_matrix=covariances[component_idx],
        )
        samples[mask] = distribution.sample((count,))

    return samples


def sample_target_distribution(num_samples, device):
    return sample_target_spec(get_target_distribution_spec(), num_samples, device)


def _coerce_stats_tensor(values, reference):
    return torch.as_tensor(values, device=reference.device, dtype=reference.dtype)


def normalize(samples, data_mean, data_std):
    mean = _coerce_stats_tensor(data_mean, samples)
    std = _coerce_stats_tensor(data_std, samples)
    return (samples - mean) / std


def denormalize(samples, data_mean, data_std):
    mean = _coerce_stats_tensor(data_mean, samples)
    std = _coerce_stats_tensor(data_std, samples)
    return samples * std + mean
