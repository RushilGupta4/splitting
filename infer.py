import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from numba import njit
from tqdm import tqdm

from model import Denoiser
from utils import (
    denormalize,
    get_checkpoint_normalization_stats,
    get_checkpoint_target_spec,
    infer_input_dim_from_checkpoint,
    validate_split_percentages,
)

KS_QUADRATURE_POINTS = 96
KS_QUAD_NODES, KS_QUAD_WEIGHTS = np.polynomial.legendre.leggauss(KS_QUADRATURE_POINTS)
KS_QUAD_NODES = np.ascontiguousarray(KS_QUAD_NODES, dtype=np.float64)
KS_QUAD_WEIGHTS = np.ascontiguousarray(KS_QUAD_WEIGHTS, dtype=np.float64)


class DDIM:
    r"""DDIM noise schedule and reverse process."""

    def __init__(
        self,
        T=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
        eta=1.0,
        sampling_steps=None,
    ):
        self.T = T
        self.device = device
        self.eta = eta
        self.sampling_steps = sampling_steps if sampling_steps is not None else T

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self._create_timestep_schedule()

    def _create_timestep_schedule(self):
        if self.sampling_steps == self.T:
            self.timesteps = list(range(self.T - 1, -1, -1))
        else:
            step_size = max(1, self.T // self.sampling_steps)
            self.timesteps = list(range(self.T - 1, -1, -step_size))[
                : self.sampling_steps
            ]
            if self.timesteps[-1] != 0:
                self.timesteps[-1] = 0

    def segment_cost(self, start_t: int, end_t: int) -> int:
        return sum(1 for t in self.timesteps if end_t <= t < start_t)

    def p_sample(self, model, x_t, t, t_prev, generator: torch.Generator | None = None):
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        eps_pred = model(x_t, t_tensor)

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = (
            self.alphas_cumprod[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=self.device)
        )

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

        if t_prev >= 0 and self.eta > 0:
            sigma = self.eta * torch.sqrt(
                (1.0 - alpha_bar_t_prev)
                / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_t_prev)
            )
        else:
            sigma = torch.tensor(0.0, device=self.device)

        sqrt_one_minus_alpha_bar_t_prev_minus_sigma_sq = torch.sqrt(
            torch.clamp(1.0 - alpha_bar_t_prev - sigma**2, min=0.0)
        )

        sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
        x_prev = (
            sqrt_alpha_bar_t_prev * x0_pred
            + sqrt_one_minus_alpha_bar_t_prev_minus_sigma_sq * eps_pred
        )

        if t_prev >= 0 and self.eta > 0:
            noise = torch.randn(
                x_t.shape,
                device=x_t.device,
                dtype=x_t.dtype,
                generator=generator,
            )
            x_prev = x_prev + sigma * noise

        return x_prev

    def sample_loop(
        self, model, x_T, start_t, end_t, generator: torch.Generator | None = None
    ):
        x = x_T
        relevant_timesteps = [t for t in self.timesteps if end_t <= t < start_t]

        with torch.inference_mode():
            for i, t in enumerate(relevant_timesteps):
                if i + 1 < len(relevant_timesteps):
                    t_prev = relevant_timesteps[i + 1]
                else:
                    t_prev = end_t - 1
                x = self.p_sample(model, x, t, t_prev, generator=generator)

        return x


def _coerce_samples_np(samples) -> np.ndarray:
    samples_np = (
        samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    )
    samples_np = np.asarray(samples_np, dtype=float).reshape(-1, 2)
    if samples_np.ndim != 2 or samples_np.shape[1] != 2:
        raise ValueError(f"Expected samples of shape [N, 2], got {samples_np.shape}")
    return samples_np


def _coerce_samples_tensor(samples, device=None) -> torch.Tensor:
    if isinstance(samples, torch.Tensor):
        tensor = samples.detach()
        if device is not None:
            tensor = tensor.to(device=device)
    else:
        tensor = torch.as_tensor(samples, device=device)
    tensor = tensor.to(dtype=torch.float32).reshape(-1, 2).contiguous()
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError(f"Expected samples of shape [N, 2], got {tuple(tensor.shape)}")
    return tensor


def _empirical_lower_orthant_cdf_torch(
    samples: torch.Tensor,
    points: torch.Tensor,
    *,
    max_pairs: int = 16_000_000,
) -> torch.Tensor:
    samples = _coerce_samples_tensor(samples, device=points.device)
    points = _coerce_samples_tensor(points, device=samples.device)
    if points.shape[0] == 0:
        return torch.empty(0, device=samples.device, dtype=torch.float32)
    if samples.shape[0] == 0:
        raise ValueError("Cannot compute an empirical CDF from zero samples")

    counts = torch.zeros(points.shape[0], device=samples.device, dtype=torch.float32)
    query_chunk = max(
        1, min(points.shape[0], int(max_pairs // max(samples.shape[0], 1)))
    )
    for q_start in range(0, points.shape[0], query_chunk):
        q_end = min(q_start + query_chunk, points.shape[0])
        query = points[q_start:q_end]
        sample_chunk = max(1, int(max_pairs // max(query.shape[0], 1)))
        chunk_counts = torch.zeros(
            query.shape[0], device=samples.device, dtype=torch.float32
        )
        for s_start in range(0, samples.shape[0], sample_chunk):
            s_end = min(s_start + sample_chunk, samples.shape[0])
            sample = samples[s_start:s_end]
            in_lower_orthant = (
                sample[:, 0].unsqueeze(0) <= query[:, 0].unsqueeze(1)
            ) & (sample[:, 1].unsqueeze(0) <= query[:, 1].unsqueeze(1))
            chunk_counts += in_lower_orthant.sum(dim=1, dtype=torch.float32)
        counts[q_start:q_end] = chunk_counts
    return counts / float(samples.shape[0])


def _standard_normal_cdf_torch(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))


def _bivariate_normal_lower_orthant_cdf_torch(
    points: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    *,
    quadrature_points: int = 96,
) -> torch.Tensor:
    dtype = points.dtype
    device = points.device
    std = torch.sqrt(torch.diag(covariance).clamp_min(torch.finfo(dtype).eps))
    rho = (covariance[0, 1] / (std[0] * std[1])).clamp(-0.999999, 0.999999)
    a = (points[:, 0] - mean[0]) / std[0]
    b = (points[:, 1] - mean[1]) / std[1]

    lower = torch.full_like(a, -12.0)
    empty_mask = a <= lower
    upper = torch.maximum(a, lower)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_points)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)

    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    x = midpoint.unsqueeze(1) + half_width.unsqueeze(1) * nodes.unsqueeze(0)
    normal_density = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    conditional_arg = (b.unsqueeze(1) - rho * x) / torch.sqrt(1.0 - rho * rho)
    integrand = normal_density * _standard_normal_cdf_torch(conditional_arg)
    cdf = half_width * torch.sum(weights.unsqueeze(0) * integrand, dim=1)
    return torch.where(empty_mask, torch.zeros_like(cdf), cdf.clamp(0.0, 1.0))


def _mixture_lower_orthant_cdf_torch(
    points: torch.Tensor,
    target_spec: Dict[str, Any],
    *,
    chunk_size: int = 8192,
) -> torch.Tensor:
    points = _coerce_samples_tensor(points)
    weights = torch.as_tensor(
        target_spec["weights"], device=points.device, dtype=points.dtype
    )
    means = torch.as_tensor(
        target_spec["means"], device=points.device, dtype=points.dtype
    )
    covariances = torch.as_tensor(
        target_spec["covariances"], device=points.device, dtype=points.dtype
    )
    out = torch.empty(points.shape[0], device=points.device, dtype=points.dtype)
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        chunk = points[start:end]
        cdf = torch.zeros(chunk.shape[0], device=points.device, dtype=points.dtype)
        for weight, mean, covariance in zip(weights, means, covariances):
            cdf = cdf + weight * _bivariate_normal_lower_orthant_cdf_torch(
                chunk, mean, covariance
            )
        out[start:end] = cdf
    return out


def _prepare_empirical_cdf_state(samples) -> Dict[str, Any]:
    samples_np = _coerce_samples_np(samples)
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot prepare an empirical CDF state from zero samples")

    y_values, y_ranks = np.unique(samples_np[:, 1], return_inverse=True)
    x_order = np.argsort(samples_np[:, 0], kind="mergesort")
    return {
        "count": int(samples_np.shape[0]),
        "samples": _coerce_samples_tensor(samples).detach().cpu(),
        "x_sorted": np.ascontiguousarray(samples_np[x_order, 0], dtype=np.float64),
        "y_ranks_sorted": np.ascontiguousarray(
            y_ranks[x_order].astype(np.int64) + 1, dtype=np.int64
        ),
        "y_values": np.ascontiguousarray(y_values, dtype=np.float64),
    }


def prepare_reference_cdf_state(samples) -> Dict[str, Any]:
    return _prepare_empirical_cdf_state(samples)


@njit(cache=True)
def _fenwick_add(tree: np.ndarray, index: int):
    while index < tree.shape[0]:
        tree[index] += 1
        index += index & -index


@njit(cache=True)
def _fenwick_prefix_sum(tree: np.ndarray, index: int) -> int:
    total = 0
    while index > 0:
        total += int(tree[index])
        index -= index & -index
    return total


@njit(cache=True)
def _upper_bound(sorted_values: np.ndarray, target: float) -> int:
    left = 0
    right = sorted_values.shape[0]
    while left < right:
        mid = (left + right) // 2
        if sorted_values[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left


@njit(cache=True)
def _empirical_lower_orthant_cdf_numba(
    x_sorted: np.ndarray,
    y_ranks_sorted: np.ndarray,
    y_values: np.ndarray,
    points_np: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    query_order = np.argsort(points_np[:, 0])
    counts = np.empty(points_np.shape[0], dtype=np.float64)
    tree = np.zeros(y_values.shape[0] + 1, dtype=np.int64)

    sample_idx = 0
    for ordered_idx in range(query_order.shape[0]):
        query_idx = query_order[ordered_idx]
        query_x = points_np[query_idx, 0]
        while sample_idx < sample_count and x_sorted[sample_idx] <= query_x:
            _fenwick_add(tree, int(y_ranks_sorted[sample_idx]))
            sample_idx += 1
        y_limit = _upper_bound(y_values, points_np[query_idx, 1])
        counts[query_idx] = _fenwick_prefix_sum(tree, y_limit) / float(sample_count)

    return counts


def _empirical_lower_orthant_cdf_from_state(
    state: Dict[str, Any], points: np.ndarray
) -> np.ndarray:
    points_np = np.ascontiguousarray(
        np.asarray(points, dtype=np.float64).reshape(-1, 2)
    )
    if points_np.shape[0] == 0:
        return np.empty(0, dtype=float)
    return _empirical_lower_orthant_cdf_numba(
        state["x_sorted"],
        state["y_ranks_sorted"],
        state["y_values"],
        points_np,
        int(state["count"]),
    )


@njit(cache=True, nogil=True)
def _segment_tree_update(
    sums: np.ndarray,
    max_prefix: np.ndarray,
    min_prefix: np.ndarray,
    size: int,
    index: int,
    delta: float,
):
    pos = size + index
    sums[pos] += delta
    max_prefix[pos] = max(0.0, sums[pos])
    min_prefix[pos] = min(0.0, sums[pos])
    pos //= 2
    while pos >= 1:
        left = pos * 2
        right = left + 1
        sums[pos] = sums[left] + sums[right]
        max_prefix[pos] = max(max_prefix[left], sums[left] + max_prefix[right])
        min_prefix[pos] = min(min_prefix[left], sums[left] + min_prefix[right])
        pos //= 2


@njit(cache=True, nogil=True)
def _exact_two_sample_lower_orthant_ks_numba(
    x_a: np.ndarray,
    y_rank_a: np.ndarray,
    x_b: np.ndarray,
    y_rank_b: np.ndarray,
    n_y: int,
) -> float:
    size = 1
    while size < n_y:
        size *= 2

    tree_len = 2 * size
    sums = np.zeros(tree_len, dtype=np.float64)
    max_prefix = np.zeros(tree_len, dtype=np.float64)
    min_prefix = np.zeros(tree_len, dtype=np.float64)

    n_a = x_a.shape[0]
    n_b = x_b.shape[0]
    weight_a = 1.0 / float(n_a)
    weight_b = -1.0 / float(n_b)
    i = 0
    j = 0
    best = 0.0

    while i < n_a or j < n_b:
        if j >= n_b or (i < n_a and x_a[i] <= x_b[j]):
            current_x = x_a[i]
        else:
            current_x = x_b[j]

        while i < n_a and x_a[i] == current_x:
            _segment_tree_update(
                sums, max_prefix, min_prefix, size, int(y_rank_a[i]), weight_a
            )
            i += 1

        while j < n_b and x_b[j] == current_x:
            _segment_tree_update(
                sums, max_prefix, min_prefix, size, int(y_rank_b[j]), weight_b
            )
            j += 1

        if max_prefix[1] > best:
            best = max_prefix[1]
        if -min_prefix[1] > best:
            best = -min_prefix[1]

    return best


def _sorted_empirical_ks_inputs(samples: np.ndarray, union_y: np.ndarray):
    samples_np = np.ascontiguousarray(
        np.asarray(samples, dtype=np.float64).reshape(-1, 2)
    )
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot compute KS distance from zero samples")
    order = np.argsort(samples_np[:, 0], kind="mergesort")
    x_sorted = np.ascontiguousarray(samples_np[order, 0], dtype=np.float64)
    y_ranks = np.searchsorted(union_y, samples_np[order, 1]).astype(np.int64)
    return x_sorted, np.ascontiguousarray(y_ranks, dtype=np.int64)


def _exact_two_sample_lower_orthant_ks_from_state(
    samples, reference_cdf_state: Dict[str, Any]
) -> float:
    samples_np = _coerce_samples_np(samples)
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot compute KS distance from zero samples")

    ref_count = int(reference_cdf_state["count"])
    if ref_count == 0:
        raise ValueError("Cannot compute KS distance against zero reference samples")

    ref_y_values = np.asarray(reference_cdf_state["y_values"], dtype=np.float64)
    union_y = np.unique(np.concatenate([samples_np[:, 1], ref_y_values]))
    x_a, y_rank_a = _sorted_empirical_ks_inputs(samples_np, union_y)

    ref_rank_map = np.searchsorted(union_y, ref_y_values).astype(np.int64)
    y_rank_b = ref_rank_map[
        np.asarray(reference_cdf_state["y_ranks_sorted"], dtype=np.int64) - 1
    ]
    x_b = np.ascontiguousarray(reference_cdf_state["x_sorted"], dtype=np.float64)
    y_rank_b = np.ascontiguousarray(y_rank_b, dtype=np.int64)

    return float(
        _exact_two_sample_lower_orthant_ks_numba(
            x_a, y_rank_a, x_b, y_rank_b, int(union_y.shape[0])
        )
    )


def _mixture_marginal_cdf_torch(
    values: torch.Tensor, target_spec: Dict[str, Any], dim: int
) -> torch.Tensor:
    values = values.reshape(-1)
    weights = torch.as_tensor(
        target_spec["weights"], device=values.device, dtype=values.dtype
    )
    means = torch.as_tensor(
        target_spec["means"], device=values.device, dtype=values.dtype
    )[:, dim]
    covariances = torch.as_tensor(
        target_spec["covariances"], device=values.device, dtype=values.dtype
    )
    std = torch.sqrt(covariances[:, dim, dim].clamp_min(torch.finfo(values.dtype).eps))
    z = (values.unsqueeze(1) - means.unsqueeze(0)) / std.unsqueeze(0)
    return torch.sum(weights.unsqueeze(0) * _standard_normal_cdf_torch(z), dim=1)


def _target_spec_arrays(target_spec: Dict[str, Any]):
    weights = np.ascontiguousarray(target_spec["weights"], dtype=np.float64)
    means = np.ascontiguousarray(target_spec["means"], dtype=np.float64)
    covariances = np.ascontiguousarray(target_spec["covariances"], dtype=np.float64)
    return weights, means, covariances


@njit(cache=True, nogil=True)
def _standard_normal_cdf_numba(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


@njit(cache=True, nogil=True)
def _bivariate_normal_lower_orthant_cdf_numba(
    x_value: float,
    y_value: float,
    mean0: float,
    mean1: float,
    cov00: float,
    cov01: float,
    cov11: float,
    quad_nodes: np.ndarray,
    quad_weights: np.ndarray,
) -> float:
    eps = 1e-15
    std0 = math.sqrt(max(cov00, eps))
    std1 = math.sqrt(max(cov11, eps))
    rho = cov01 / (std0 * std1)
    if rho < -0.999999:
        rho = -0.999999
    elif rho > 0.999999:
        rho = 0.999999

    a = (x_value - mean0) / std0
    b = (y_value - mean1) / std1
    lower = -12.0
    if a <= lower:
        return 0.0

    half_width = 0.5 * (a - lower)
    midpoint = 0.5 * (a + lower)
    denom = math.sqrt(1.0 - rho * rho)
    total = 0.0
    for idx in range(quad_nodes.shape[0]):
        z = midpoint + half_width * quad_nodes[idx]
        normal_density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        conditional_arg = (b - rho * z) / denom
        total += (
            quad_weights[idx]
            * normal_density
            * _standard_normal_cdf_numba(conditional_arg)
        )
    cdf = half_width * total
    if cdf < 0.0:
        return 0.0
    if cdf > 1.0:
        return 1.0
    return cdf


@njit(cache=True, nogil=True)
def _mixture_lower_orthant_cdf_numba(
    x_value: float,
    y_value: float,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    quad_nodes: np.ndarray,
    quad_weights: np.ndarray,
) -> float:
    total = 0.0
    for idx in range(weights.shape[0]):
        total += weights[idx] * _bivariate_normal_lower_orthant_cdf_numba(
            x_value,
            y_value,
            means[idx, 0],
            means[idx, 1],
            covariances[idx, 0, 0],
            covariances[idx, 0, 1],
            covariances[idx, 1, 1],
            quad_nodes,
            quad_weights,
        )
    if total < 0.0:
        return 0.0
    if total > 1.0:
        return 1.0
    return total


@njit(cache=True, nogil=True)
def _mixture_marginal_cdf_numba(
    value: float,
    dim: int,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> float:
    total = 0.0
    for idx in range(weights.shape[0]):
        variance = covariances[idx, dim, dim]
        std = math.sqrt(max(variance, 1e-15))
        total += weights[idx] * _standard_normal_cdf_numba(
            (value - means[idx, dim]) / std
        )
    if total < 0.0:
        return 0.0
    if total > 1.0:
        return 1.0
    return total


@njit(cache=True, nogil=True)
def _exact_empirical_target_lower_orthant_ks_numba(
    x_values: np.ndarray,
    x_counts: np.ndarray,
    y_values: np.ndarray,
    y_ranks_by_x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    quad_nodes: np.ndarray,
    quad_weights: np.ndarray,
    n_samples: int,
) -> float:
    counts_by_y = np.zeros(y_values.shape[0], dtype=np.int64)
    group_counts_by_y = np.zeros(y_values.shape[0], dtype=np.int64)
    inv_n = 1.0 / float(n_samples)
    best = 0.0
    cursor = 0

    for x_idx in range(x_values.shape[0]):
        x_value = x_values[x_idx]
        x_count = int(x_counts[x_idx])
        before_total = cursor

        for offset in range(x_count):
            y_rank = int(y_ranks_by_x[cursor + offset])
            counts_by_y[y_rank] += 1
            group_counts_by_y[y_rank] += 1

        cursor += x_count
        marginal_x = _mixture_marginal_cdf_numba(
            x_value, 0, weights, means, covariances
        )
        diff = abs(before_total * inv_n - marginal_x)
        if diff > best:
            best = diff
        diff = abs(cursor * inv_n - marginal_x)
        if diff > best:
            best = diff

        prefix_after = 0
        group_prefix = 0
        for y_idx in range(y_values.shape[0]):
            current_y_count = counts_by_y[y_idx]
            current_group_count = group_counts_by_y[y_idx]
            prefix_after += current_y_count
            group_prefix += current_group_count

            prefix_before = prefix_after - group_prefix
            prefix_after_y_before = prefix_after - current_y_count
            prefix_before_y_before = prefix_before - (
                current_y_count - current_group_count
            )
            target_cdf = _mixture_lower_orthant_cdf_numba(
                x_value,
                y_values[y_idx],
                weights,
                means,
                covariances,
                quad_nodes,
                quad_weights,
            )

            diff = abs(prefix_after * inv_n - target_cdf)
            if diff > best:
                best = diff
            diff = abs(prefix_before * inv_n - target_cdf)
            if diff > best:
                best = diff
            diff = abs(prefix_after_y_before * inv_n - target_cdf)
            if diff > best:
                best = diff
            diff = abs(prefix_before_y_before * inv_n - target_cdf)
            if diff > best:
                best = diff

        for offset in range(x_count):
            group_counts_by_y[int(y_ranks_by_x[cursor - x_count + offset])] = 0

    prefix_y = 0
    for y_idx in range(y_values.shape[0]):
        prefix_y += counts_by_y[y_idx]
        target_marginal = _mixture_marginal_cdf_numba(
            y_values[y_idx], 1, weights, means, covariances
        )
        diff = abs(prefix_y * inv_n - target_marginal)
        if diff > best:
            best = diff
        diff = abs((prefix_y - counts_by_y[y_idx]) * inv_n - target_marginal)
        if diff > best:
            best = diff

    return best


def _exact_empirical_target_lower_orthant_ks(
    samples,
    target_spec: Dict[str, Any],
    device: torch.device | None = None,
) -> float:
    samples_np = _coerce_samples_np(samples)
    n_samples = int(samples_np.shape[0])
    if n_samples == 0:
        raise ValueError("Cannot compute KS distance from zero samples")

    x_values, x_counts = np.unique(samples_np[:, 0], return_counts=True)
    y_values, y_inverse = np.unique(samples_np[:, 1], return_inverse=True)
    x_order = np.argsort(samples_np[:, 0], kind="mergesort")
    y_ranks_by_x = np.ascontiguousarray(y_inverse[x_order], dtype=np.int64)
    weights, means, covariances = _target_spec_arrays(target_spec)
    return float(
        _exact_empirical_target_lower_orthant_ks_numba(
            np.ascontiguousarray(x_values, dtype=np.float64),
            np.ascontiguousarray(x_counts, dtype=np.int64),
            np.ascontiguousarray(y_values, dtype=np.float64),
            y_ranks_by_x,
            weights,
            means,
            covariances,
            KS_QUAD_NODES,
            KS_QUAD_WEIGHTS,
            n_samples,
        )
    )


def _warm_target_ks_kernel(target_spec: Dict[str, Any]):
    weights, means, covariances = _target_spec_arrays(target_spec)
    _exact_empirical_target_lower_orthant_ks_numba(
        np.ascontiguousarray([0.0], dtype=np.float64),
        np.ascontiguousarray([1], dtype=np.int64),
        np.ascontiguousarray([0.0], dtype=np.float64),
        np.ascontiguousarray([0], dtype=np.int64),
        weights,
        means,
        covariances,
        KS_QUAD_NODES,
        KS_QUAD_WEIGHTS,
        1,
    )


def _warm_ks_kernel_for_mode(
    reference_mode: str,
    target_spec: Dict[str, Any],
):
    if reference_mode == "true_dist":
        _warm_target_ks_kernel(target_spec)
    elif reference_mode in {"true_samples", "ddpm_samples"}:
        _exact_two_sample_lower_orthant_ks_numba(
            np.ascontiguousarray([0.0], dtype=np.float64),
            np.ascontiguousarray([0], dtype=np.int64),
            np.ascontiguousarray([0.0], dtype=np.float64),
            np.ascontiguousarray([0], dtype=np.int64),
            1,
        )


def warm_reference_ks_kernel(reference_cdf_state: Dict[str, Any]):
    sample_x = reference_cdf_state["x_sorted"][0]
    sample_y = reference_cdf_state["y_values"][0]
    warm_points = np.ascontiguousarray([[sample_x, sample_y]], dtype=np.float64)
    _empirical_lower_orthant_cdf_numba(
        reference_cdf_state["x_sorted"],
        reference_cdf_state["y_ranks_sorted"],
        reference_cdf_state["y_values"],
        warm_points,
        int(reference_cdf_state["count"]),
    )
    _exact_two_sample_lower_orthant_ks_numba(
        np.ascontiguousarray([sample_x], dtype=np.float64),
        np.ascontiguousarray([0], dtype=np.int64),
        np.ascontiguousarray([sample_x], dtype=np.float64),
        np.ascontiguousarray([0], dtype=np.int64),
        1,
    )


def compute_reference_ks_distance(samples, reference_cdf_state: Dict[str, Any]):
    """Compute exact 2D lower-orthant KS distance against empirical samples."""
    ks_distance = _exact_two_sample_lower_orthant_ks_from_state(
        samples, reference_cdf_state
    )
    empty = torch.empty(0)
    return ks_distance, empty, empty


def compute_target_ks_distance(samples, target_spec: Dict[str, Any]):
    """Compute exact 2D lower-orthant KS distance against the target CDF."""
    samples_np = _coerce_samples_np(samples)
    ks_distance = _exact_empirical_target_lower_orthant_ks(samples_np, target_spec)
    empty = torch.empty(0)
    return ks_distance, empty, empty


def _compute_ks_distance(
    samples,
    target_spec: Dict[str, Any],
    reference_mode: str,
    reference_cdf_state: Dict[str, Any] | None,
):
    if reference_mode in {"true_samples", "ddpm_samples"}:
        if reference_cdf_state is None:
            raise ValueError(
                f"reference_cdf_state is required when reference_mode='{reference_mode}'"
            )
        return compute_reference_ks_distance(samples, reference_cdf_state)
    if reference_mode == "true_dist":
        return compute_target_ks_distance(samples, target_spec)
    raise ValueError(f"Unknown reference_mode '{reference_mode}'")


def _sample_loop(
    ddim: DDIM,
    model,
    x: torch.Tensor,
    start_t: int,
    end_t: int,
    generator: torch.Generator | None = None,
):
    if x.shape[0] == 0:
        return x
    return ddim.sample_loop(model, x, start_t, end_t, generator=generator)


def _parse_split_percentages(split_percentages_str: str) -> List[float]:
    split_percentages = [
        float(x.strip()) for x in split_percentages_str.split(",") if x.strip()
    ]
    validate_split_percentages(split_percentages)
    return split_percentages


def _parse_x_grid(x_grid_str: str) -> List[float]:
    x_grid = [float(x.strip()) for x in x_grid_str.split(",") if x.strip()]
    if not x_grid:
        raise ValueError("x_grid must have at least one threshold")
    return x_grid


def _debug_log(enabled: bool, message: str):
    if enabled:
        print(f"[debug] {message}")


def _make_torch_generator(seed: int | None, device: str):
    if seed is None:
        return None
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


PHASE2_SEED_OFFSET = 1_000_000
SUPPORTED_BIAS_TYPES = {"biased", "unbiased"}


def _validate_bias_type(bias_type: str):
    if bias_type not in SUPPORTED_BIAS_TYPES:
        supported = ", ".join(sorted(SUPPORTED_BIAS_TYPES))
        raise ValueError(f"bias_type must be one of: {supported}; got '{bias_type}'")


def _iter_run_chunks(n_runs: int, n_parallel: int):
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if n_parallel < 1:
        raise ValueError("n_parallel must be at least 1")
    for start in range(0, n_runs, n_parallel):
        yield start, min(start + n_parallel, n_runs)


def _resolve_split_percentages(
    ddim: DDIM, split_percentages: Sequence[float]
) -> Tuple[List[int], List[int]]:
    validate_split_percentages(split_percentages)
    remaining_steps = [
        int(round(ddim.sampling_steps * pct)) for pct in split_percentages
    ]

    for i, steps_left in enumerate(remaining_steps):
        if steps_left <= 0 or steps_left >= ddim.sampling_steps:
            raise ValueError(
                "Each split percentage must map to an interior split. "
                f"Got round({ddim.sampling_steps} * {split_percentages[i]}) = {steps_left}."
            )

    for i in range(len(remaining_steps) - 1):
        if remaining_steps[i] <= remaining_steps[i + 1]:
            raise ValueError(
                "Rounded split points must be strictly decreasing. "
                f"Got {remaining_steps[i]} <= {remaining_steps[i + 1]} from split_percentages "
                f"{split_percentages[i]} and {split_percentages[i + 1]}."
            )

    split_points = [
        int(ddim.timesteps[ddim.sampling_steps - steps_left] + 1)
        for steps_left in remaining_steps
    ]
    return remaining_steps, split_points


def _load_model_and_stats(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_args = checkpoint["args"]
    input_dim = infer_input_dim_from_checkpoint(checkpoint)

    model = Denoiser(
        input_dim=input_dim,
        hidden_dim=model_args.get("hidden_dim", 128),
        num_blocks=model_args.get("num_blocks", 4),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if hasattr(torch, "compile") and not getattr(args, "no_compile", False):
        model = torch.compile(model, dynamic=True)
        _debug_log(
            getattr(args, "debug", False), "Compiled denoiser with torch.compile"
        )
    elif getattr(args, "no_compile", False):
        _debug_log(getattr(args, "debug", False), "Skipped torch.compile")

    target_spec = get_checkpoint_target_spec(checkpoint)
    data_mean, data_std = get_checkpoint_normalization_stats(
        checkpoint, device=args.device
    )
    return model, target_spec, data_mean, data_std


def _model_input_dim(model) -> int:
    input_dim = getattr(model, "input_dim", None)
    if input_dim is not None:
        return int(input_dim)
    orig_model = getattr(model, "_orig_mod", None)
    if orig_model is not None and getattr(orig_model, "input_dim", None) is not None:
        return int(orig_model.input_dim)
    raise AttributeError("Could not determine model input dimension")


def _pilot_cost_per_root(ddim: DDIM, split_points: Sequence[int], m_pilot: float):
    start_points = [ddim.T] + list(split_points)
    end_points = list(split_points) + [0]
    cost = 0.0
    for idx, (start_t, end_t) in enumerate(zip(start_points, end_points)):
        cost += (m_pilot ** (idx + 1)) * ddim.segment_cost(start_t, end_t)
    return cost


def _pilot_tree_shape_from_scale(pilot_scale: float):
    if pilot_scale <= 1.0:
        raise ValueError("pilot_scale must be > 1")
    pilot_roots = max(1, int(pilot_scale * pilot_scale))
    return pilot_roots, float(pilot_scale)


def _pilot_tree_expected_cost(
    ddim: DDIM, split_points: Sequence[int], pilot_scale: float
):
    pilot_roots, m_pilot = _pilot_tree_shape_from_scale(pilot_scale)
    return pilot_roots * _pilot_cost_per_root(ddim, split_points, m_pilot)


def _derive_pilot_tree_shape(ddim: DDIM, split_points: Sequence[int], B1: int):
    # Use pilot_scale = y so pilot_roots ~ y^2 and per-level branching ~ y.
    min_pilot_scale = 1.0 + 1e-6
    min_cost = _pilot_tree_expected_cost(ddim, split_points, min_pilot_scale)
    if B1 < min_cost:
        raise ValueError(
            f"B1={B1} is too small for pilot_tree; need at least {min_cost:.6f}"
        )

    lower = min_pilot_scale
    upper = 2.0
    while _pilot_tree_expected_cost(ddim, split_points, upper) <= B1:
        upper *= 2.0

    for _ in range(80):
        mid = 0.5 * (lower + upper)
        if _pilot_tree_expected_cost(ddim, split_points, mid) <= B1:
            lower = mid
        else:
            upper = mid

    pilot_roots, m_pilot = _pilot_tree_shape_from_scale(lower)
    return float(lower), int(pilot_roots), float(m_pilot)


def _independent_sigma_cost(
    ddim: DDIM,
    t_curr: int,
    t_next: int,
    outer_count: int,
    middle_count: int,
    inner_count: int,
):
    return (
        outer_count * ddim.segment_cost(ddim.T, t_curr)
        + outer_count * middle_count * ddim.segment_cost(t_curr, t_next)
        + outer_count * middle_count * inner_count * ddim.segment_cost(t_next, 0)
    )


def _derive_independent_counts(
    ddim: DDIM,
    sigma_times: Sequence[int],
    B1: int,
    independent_n2: int,
    bias_type: str = "unbiased",
) -> Tuple[float, List[Dict[str, int]], int]:
    _validate_bias_type(bias_type)
    if not sigma_times:
        raise ValueError("sigma_times must be non-empty")
    if independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")

    budget_per_sigma = B1 / len(sigma_times)
    if budget_per_sigma <= 0:
        raise ValueError("B1 must be positive for independent sigma estimation")

    counts_by_sigma = []
    total_used_B1 = 0
    next_times = list(sigma_times[1:]) + [0]

    for t_curr, t_next in zip(sigma_times, next_times):
        segment1 = ddim.segment_cost(ddim.T, t_curr)
        segment2 = ddim.segment_cost(t_curr, t_next)
        segment3 = ddim.segment_cost(t_next, 0)
        min_inner_count = 2 if bias_type == "unbiased" and segment3 > 0 else 1
        min_cost = (
            segment1
            + independent_n2 * segment2
            + independent_n2 * min_inner_count * segment3
        )
        if budget_per_sigma < min_cost:
            raise ValueError(
                "B1 is too small for equal per-sigma independent estimation with "
                f"independent_n2={independent_n2}; budget_per_sigma={budget_per_sigma:.6f}, "
                f"minimum required={min_cost:.6f} at sigma time {t_curr}"
            )

        def cost_for_y(y: float):
            return (y**2) * (segment1 + independent_n2 * segment2) + (y**3) * (
                independent_n2 * segment3
            )

        lower = 1.0
        upper = 2.0
        while cost_for_y(upper) <= budget_per_sigma:
            upper *= 2.0

        for _ in range(80):
            mid = 0.5 * (lower + upper)
            if cost_for_y(mid) <= budget_per_sigma:
                lower = mid
            else:
                upper = mid

        outer_count = max(1, int(lower * lower))
        # For the final sigma block, t_next == 0, so inner repeats are exact
        # zero-step duplicates. They do not change the sigma estimate but would
        # overweight those x0 samples when reuse_phase1_samples=True.
        if segment3 == 0:
            inner_count = 1
        elif bias_type == "unbiased":
            inner_count = max(2, int(lower))
        else:
            inner_count = max(1, int(lower))
        used_budget = _independent_sigma_cost(
            ddim, t_curr, t_next, outer_count, independent_n2, inner_count
        )

        counts_by_sigma.append(
            {
                "time": int(t_curr),
                "next_time": int(t_next),
                "outer_count": int(outer_count),
                "middle_count": int(independent_n2),
                "inner_count": int(inner_count),
                "used_budget": int(used_budget),
            }
        )
        total_used_B1 += used_budget

    return budget_per_sigma, counts_by_sigma, int(total_used_B1)


def _sample_branch_counts(
    num_parents: int,
    branching_factor: float,
    device: str,
    generator: torch.Generator | None = None,
):
    if branching_factor <= 1.0:
        raise ValueError("Branching factor must be > 1 so parents can have children")

    base_copies = math.floor(branching_factor)
    extra_prob = branching_factor - base_copies
    counts = torch.full((num_parents,), base_copies, device=device, dtype=torch.long)
    if extra_prob > 0.0:
        counts = counts + (
            torch.rand(num_parents, device=device, generator=generator) < extra_prob
        ).to(torch.long)
    return counts


def _repeat_by_counts(x: torch.Tensor, counts: torch.Tensor):
    return x.repeat_interleave(counts, dim=0)


def _actual_pilot_cost(
    ddim: DDIM,
    split_points: Sequence[int],
    level_counts: Sequence[int],
    leaf_count: int,
):
    cost = level_counts[0] * ddim.segment_cost(ddim.T, split_points[0])
    for idx in range(1, len(split_points)):
        cost += level_counts[idx] * ddim.segment_cost(
            split_points[idx - 1], split_points[idx]
        )
    cost += leaf_count * ddim.segment_cost(split_points[-1], 0)
    return int(cost)


def _build_pilot_tree_batch(
    model,
    ddim: DDIM,
    chunk_size: int,
    pilot_roots: int,
    split_points: Sequence[int],
    m_pilot: float,
    generator: torch.Generator | None = None,
):
    input_dim = _model_input_dim(model)
    x = torch.randn(
        chunk_size * pilot_roots,
        input_dim,
        device=ddim.device,
        generator=generator,
    )
    run_ids = torch.repeat_interleave(
        torch.arange(chunk_size, device=ddim.device, dtype=torch.long),
        pilot_roots,
    )
    child_counts_by_level = []
    parent_run_ids_by_level = []
    used_B1_by_run = torch.zeros(chunk_size, device=ddim.device, dtype=torch.long)

    for idx, split_point in enumerate(split_points):
        parent_run_ids = run_ids
        child_counts = _sample_branch_counts(
            x.shape[0], m_pilot, ddim.device, generator=generator
        )
        child_counts_by_level.append(child_counts)
        parent_run_ids_by_level.append(parent_run_ids)
        x = _repeat_by_counts(x, child_counts)
        run_ids = run_ids.repeat_interleave(child_counts, dim=0)
        start_t = ddim.T if idx == 0 else split_points[idx - 1]
        x = _sample_loop(ddim, model, x, start_t, split_point, generator=generator)
        used_B1_by_run = used_B1_by_run + torch.bincount(
            run_ids, minlength=chunk_size
        ) * int(ddim.segment_cost(start_t, split_point))

    parent_run_ids = run_ids
    child_counts = _sample_branch_counts(
        x.shape[0], m_pilot, ddim.device, generator=generator
    )
    child_counts_by_level.append(child_counts)
    parent_run_ids_by_level.append(parent_run_ids)
    x = _repeat_by_counts(x, child_counts)
    run_ids = run_ids.repeat_interleave(child_counts, dim=0)
    x = _sample_loop(ddim, model, x, split_points[-1], 0, generator=generator)
    used_B1_by_run = used_B1_by_run + torch.bincount(
        run_ids, minlength=chunk_size
    ) * int(ddim.segment_cost(split_points[-1], 0))

    return (
        x,
        run_ids,
        child_counts_by_level,
        parent_run_ids_by_level,
        used_B1_by_run.detach().cpu().tolist(),
    )


def _rectangle_indicator_grid(values: torch.Tensor, x_grid: Sequence[float]):
    thresholds = torch.as_tensor(x_grid, device=values.device, dtype=values.dtype)
    x1_below = values[:, 0:1] <= thresholds.unsqueeze(0)
    x2_below = values[:, 1:2] <= thresholds.unsqueeze(0)
    return (
        (x1_below.unsqueeze(2) & x2_below.unsqueeze(1))
        .to(values.dtype)
        .reshape(values.shape[0], -1)
    )


def _segment_mean_and_unbiased_var(values: torch.Tensor, counts: torch.Tensor):
    counts = counts.to(device=values.device, dtype=torch.long)
    means = torch.segment_reduce(values, reduce="mean", lengths=counts)
    valid_mask = counts >= 2
    if not valid_mask.any():
        return means, None

    sums = torch.segment_reduce(values, reduce="sum", lengths=counts)
    sums_sq = torch.segment_reduce(values * values, reduce="sum", lengths=counts)
    count_values = counts.to(dtype=values.dtype).unsqueeze(1)
    denom = torch.clamp(count_values - 1.0, min=1.0)
    variances = (sums_sq - (sums * sums) / count_values) / denom
    variances = torch.clamp(variances, min=0.0)
    return means, variances[valid_mask]


def _segment_mean_and_unbiased_var_all(values: torch.Tensor, counts: torch.Tensor):
    counts = counts.to(device=values.device, dtype=torch.long)
    means = torch.segment_reduce(values, reduce="mean", lengths=counts)
    sums = torch.segment_reduce(values, reduce="sum", lengths=counts)
    sums_sq = torch.segment_reduce(values * values, reduce="sum", lengths=counts)
    count_values = counts.to(dtype=values.dtype).unsqueeze(1)
    denom = torch.clamp(count_values - 1.0, min=1.0)
    variances = (sums_sq - (sums * sums) / count_values) / denom
    variances = torch.clamp(variances, min=0.0)
    valid_mask = counts >= 2
    return means, variances, valid_mask


def _grouped_mean(values: torch.Tensor, group_ids: torch.Tensor, num_groups: int):
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape {tuple(values.shape)}")
    group_ids = group_ids.to(device=values.device, dtype=torch.long)
    sums = torch.zeros(
        num_groups, values.shape[1], device=values.device, dtype=values.dtype
    )
    sums.index_add_(0, group_ids, values)
    counts = torch.bincount(group_ids, minlength=num_groups).to(
        device=values.device, dtype=values.dtype
    )
    safe_counts = counts.clamp_min(1.0).unsqueeze(1)
    return sums / safe_counts, counts


def _grouped_unbiased_var(
    values: torch.Tensor, group_ids: torch.Tensor, num_groups: int
):
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape {tuple(values.shape)}")
    group_ids = group_ids.to(device=values.device, dtype=torch.long)
    sums = torch.zeros(
        num_groups, values.shape[1], device=values.device, dtype=values.dtype
    )
    sums_sq = torch.zeros_like(sums)
    sums.index_add_(0, group_ids, values)
    sums_sq.index_add_(0, group_ids, values * values)
    counts = torch.bincount(group_ids, minlength=num_groups).to(
        device=values.device, dtype=values.dtype
    )
    safe_counts = counts.clamp_min(1.0).unsqueeze(1)
    denom = torch.clamp(safe_counts - 1.0, min=1.0)
    variances = (sums_sq - (sums * sums) / safe_counts) / denom
    variances = torch.where(
        (counts >= 2).unsqueeze(1),
        torch.clamp(variances, min=0.0),
        torch.zeros_like(variances),
    )
    means = sums / safe_counts
    return variances, means, counts


def _grid_tensor_to_nested_list(values: torch.Tensor, x_grid: Sequence[float]):
    grid_size = len(x_grid)
    return values.reshape(grid_size, grid_size).detach().cpu().tolist()


def _sigma_estimate_from_flat(
    sigma_time: int,
    sigma2_flat: torch.Tensor,
    sigma_flat: torch.Tensor,
    p_hat_mean_flat: torch.Tensor,
    x_grid: Sequence[float],
):
    sigma2_grid = _grid_tensor_to_nested_list(sigma2_flat, x_grid)
    sigma_grid = _grid_tensor_to_nested_list(sigma_flat, x_grid)
    p_hat_mean_grid = _grid_tensor_to_nested_list(p_hat_mean_flat, x_grid)
    sigma2_array = np.array(sigma2_grid, dtype=float)
    max_index = np.unravel_index(np.argmax(sigma2_array), sigma2_array.shape)
    return {
        "time": int(sigma_time),
        "x_grid": list(x_grid),
        "sigma2_grid": sigma2_grid,
        "sigma_grid": sigma_grid,
        "p_hat_mean_grid": p_hat_mean_grid,
        "sigma2_max": float(sigma2_array[max_index]),
        "sigma": float(math.sqrt(max(float(sigma2_array[max_index]), 0.0))),
        "max_location": [
            float(x_grid[max_index[0]]),
            float(x_grid[max_index[1]]),
        ],
    }


def _tau_estimate_from_flat(
    tau2_flat: torch.Tensor,
    tau_flat: torch.Tensor,
    tau_mean_flat: torch.Tensor,
    x_grid: Sequence[float],
):
    tau2_grid = _grid_tensor_to_nested_list(tau2_flat, x_grid)
    tau_grid = _grid_tensor_to_nested_list(tau_flat, x_grid)
    tau_mean_grid = _grid_tensor_to_nested_list(tau_mean_flat, x_grid)
    tau2_array = np.array(tau2_grid, dtype=float)
    tau_max_index = np.unravel_index(np.argmax(tau2_array), tau2_array.shape)
    return {
        "x_grid": list(x_grid),
        "tau2_grid": tau2_grid,
        "tau_grid": tau_grid,
        "p_hat_mean_grid": tau_mean_grid,
        "tau2_max": float(tau2_array[tau_max_index]),
        "tau": float(math.sqrt(max(float(tau2_array[tau_max_index]), 0.0))),
        "max_location": [
            float(x_grid[tau_max_index[0]]),
            float(x_grid[tau_max_index[1]]),
        ],
    }


def _estimate_sigmas_from_tree(
    leaf_x0: torch.Tensor,
    sigma_times: Sequence[int],
    child_counts_by_level: Sequence[torch.Tensor],
    x_grid: Sequence[float],
    bias_type: str = "unbiased",
):
    _validate_bias_type(bias_type)
    num_levels = len(sigma_times)
    current = _rectangle_indicator_grid(leaf_x0, x_grid)
    grid_size = len(x_grid)
    raw_var_by_level = [None] * num_levels
    valid_mask_by_level = [None] * num_levels
    mean_by_level = [None] * num_levels

    for rev_idx in range(num_levels - 1, -1, -1):
        counts = child_counts_by_level[rev_idx].to(
            device=current.device, dtype=torch.long
        )
        if int(counts.sum().item()) != current.shape[0]:
            raise ValueError("Pilot tree child counts do not match leaf layout")

        child_means, child_vars, valid_mask = _segment_mean_and_unbiased_var_all(
            current, counts
        )
        if not valid_mask.any():
            raise ValueError(
                "Pilot tree produced no parents with at least two children; increase B1"
            )

        mean_flat = child_means.mean(dim=0)
        raw_var_by_level[rev_idx] = child_vars
        valid_mask_by_level[rev_idx] = valid_mask
        mean_by_level[rev_idx] = mean_flat
        current = child_means

    sigma2_by_level = [None] * num_levels
    sigma_by_level = [None] * num_levels
    for level_idx in range(num_levels - 1, -1, -1):
        corrected_vars = raw_var_by_level[level_idx]
        if bias_type == "unbiased" and level_idx + 1 < num_levels:
            lower_counts = child_counts_by_level[level_idx + 1].to(
                device=corrected_vars.device, dtype=corrected_vars.dtype
            )
            lower_noise_by_child = raw_var_by_level[
                level_idx + 1
            ] / lower_counts.unsqueeze(1)
            parent_counts = child_counts_by_level[level_idx].to(
                device=corrected_vars.device, dtype=torch.long
            )
            lower_noise_by_parent = torch.segment_reduce(
                lower_noise_by_child, reduce="mean", lengths=parent_counts
            )
            corrected_vars = corrected_vars - lower_noise_by_parent

        valid_mask = valid_mask_by_level[level_idx]
        sigma2_flat = torch.clamp(corrected_vars[valid_mask].mean(dim=0), min=0.0)
        sigma2_by_level[level_idx] = sigma2_flat
        sigma_by_level[level_idx] = torch.sqrt(sigma2_flat)

    if current.shape[0] >= 2:
        tau2_flat = torch.clamp(current.var(dim=0, unbiased=True), min=0.0)
    else:
        tau2_flat = torch.zeros(
            grid_size * grid_size, device=current.device, dtype=current.dtype
        )
    tau_flat = torch.sqrt(torch.clamp(tau2_flat, min=0.0))
    tau_mean_flat = current.mean(dim=0)

    sigma2_grids = [
        _grid_tensor_to_nested_list(values, x_grid) for values in sigma2_by_level
    ]
    sigma_grids = [
        _grid_tensor_to_nested_list(values, x_grid) for values in sigma_by_level
    ]
    p_hat_mean_grids = [
        _grid_tensor_to_nested_list(values, x_grid) for values in mean_by_level
    ]
    tau2_grid = _grid_tensor_to_nested_list(tau2_flat, x_grid)
    tau_grid = _grid_tensor_to_nested_list(tau_flat, x_grid)
    tau_mean_grid = _grid_tensor_to_nested_list(tau_mean_flat, x_grid)

    estimates = []
    for level_idx, sigma_time in enumerate(sigma_times):
        sigma2_grid = sigma2_grids[level_idx]
        sigma_grid = sigma_grids[level_idx]
        p_hat_mean_grid = p_hat_mean_grids[level_idx]
        sigma2_array = np.array(sigma2_grid, dtype=float)
        max_index = np.unravel_index(np.argmax(sigma2_array), sigma2_array.shape)
        estimates.append(
            {
                "time": sigma_time,
                "x_grid": list(x_grid),
                "sigma2_grid": sigma2_grid,
                "sigma_grid": sigma_grid,
                "p_hat_mean_grid": p_hat_mean_grid,
                "sigma2_max": float(sigma2_array[max_index]),
                "sigma": float(math.sqrt(max(float(sigma2_array[max_index]), 0.0))),
                "max_location": [
                    float(x_grid[max_index[0]]),
                    float(x_grid[max_index[1]]),
                ],
            }
        )

    tau2_array = np.array(tau2_grid, dtype=float)
    tau_max_index = np.unravel_index(np.argmax(tau2_array), tau2_array.shape)
    tau_estimate = {
        "x_grid": list(x_grid),
        "tau2_grid": tau2_grid,
        "tau_grid": tau_grid,
        "p_hat_mean_grid": tau_mean_grid,
        "tau2_max": float(tau2_array[tau_max_index]),
        "tau": float(math.sqrt(max(float(tau2_array[tau_max_index]), 0.0))),
        "max_location": [
            float(x_grid[tau_max_index[0]]),
            float(x_grid[tau_max_index[1]]),
        ],
    }

    return estimates, tau_estimate


def _estimate_sigmas_from_tree_batch(
    leaf_x0: torch.Tensor,
    leaf_run_ids: torch.Tensor,
    sigma_times: Sequence[int],
    child_counts_by_level: Sequence[torch.Tensor],
    parent_run_ids_by_level: Sequence[torch.Tensor],
    x_grid: Sequence[float],
    chunk_size: int,
    bias_type: str = "unbiased",
):
    _validate_bias_type(bias_type)
    num_levels = len(sigma_times)
    if len(child_counts_by_level) != num_levels:
        raise ValueError("child_counts_by_level must match sigma_times length")
    if len(parent_run_ids_by_level) != num_levels:
        raise ValueError("parent_run_ids_by_level must match sigma_times length")

    current = _rectangle_indicator_grid(leaf_x0, x_grid)
    current_run_ids = leaf_run_ids.to(device=current.device, dtype=torch.long)
    grid_size = len(x_grid)
    raw_var_by_level = [None] * num_levels
    valid_mask_by_level = [None] * num_levels
    mean_by_level = [None] * num_levels

    for rev_idx in range(num_levels - 1, -1, -1):
        counts = child_counts_by_level[rev_idx].to(
            device=current.device, dtype=torch.long
        )
        parent_run_ids = parent_run_ids_by_level[rev_idx].to(
            device=current.device, dtype=torch.long
        )
        if counts.shape[0] != parent_run_ids.shape[0]:
            raise ValueError("Pilot tree parent run ids do not match child counts")

        child_means, child_vars, valid_mask = _segment_mean_and_unbiased_var_all(
            current, counts
        )

        mean_flat, _ = _grouped_mean(child_means, parent_run_ids, chunk_size)
        raw_var_by_level[rev_idx] = child_vars
        valid_mask_by_level[rev_idx] = valid_mask
        mean_by_level[rev_idx] = mean_flat
        current = child_means
        current_run_ids = parent_run_ids

    sigma2_by_level = [None] * num_levels
    sigma_by_level = [None] * num_levels
    for level_idx in range(num_levels - 1, -1, -1):
        corrected_vars = raw_var_by_level[level_idx]
        if bias_type == "unbiased" and level_idx + 1 < num_levels:
            lower_counts = child_counts_by_level[level_idx + 1].to(
                device=corrected_vars.device, dtype=corrected_vars.dtype
            )
            lower_noise_by_child = raw_var_by_level[
                level_idx + 1
            ] / lower_counts.unsqueeze(1)
            parent_counts = child_counts_by_level[level_idx].to(
                device=corrected_vars.device, dtype=torch.long
            )
            lower_noise_by_parent = torch.segment_reduce(
                lower_noise_by_child, reduce="mean", lengths=parent_counts
            )
            corrected_vars = corrected_vars - lower_noise_by_parent

        valid_mask = valid_mask_by_level[level_idx]
        parent_run_ids = parent_run_ids_by_level[level_idx].to(
            device=corrected_vars.device, dtype=torch.long
        )
        sigma2_flat, _ = _grouped_mean(
            corrected_vars[valid_mask], parent_run_ids[valid_mask], chunk_size
        )
        sigma2_flat = torch.clamp(sigma2_flat, min=0.0)
        sigma2_by_level[level_idx] = sigma2_flat
        sigma_by_level[level_idx] = torch.sqrt(sigma2_flat)

    tau2_by_run, tau_mean_by_run, _ = _grouped_unbiased_var(
        current, current_run_ids, chunk_size
    )
    tau_by_run = torch.sqrt(torch.clamp(tau2_by_run, min=0.0))

    sigma2_np = torch.stack(sigma2_by_level, dim=1).detach().cpu().numpy()
    sigma_np = torch.stack(sigma_by_level, dim=1).detach().cpu().numpy()
    mean_np = torch.stack(mean_by_level, dim=1).detach().cpu().numpy()
    tau2_np = tau2_by_run.detach().cpu().numpy()
    tau_np = tau_by_run.detach().cpu().numpy()
    tau_mean_np = tau_mean_by_run.detach().cpu().numpy()

    sigma_estimates_by_run = []
    tau_estimates = []
    for run_idx in range(chunk_size):
        run_estimates = []
        for level_idx, sigma_time in enumerate(sigma_times):
            sigma2_grid = (
                sigma2_np[run_idx, level_idx].reshape(grid_size, grid_size).tolist()
            )
            sigma_grid = (
                sigma_np[run_idx, level_idx].reshape(grid_size, grid_size).tolist()
            )
            p_hat_mean_grid = (
                mean_np[run_idx, level_idx].reshape(grid_size, grid_size).tolist()
            )
            sigma2_array = np.asarray(sigma2_grid, dtype=float)
            max_index = np.unravel_index(np.argmax(sigma2_array), sigma2_array.shape)
            run_estimates.append(
                {
                    "time": int(sigma_time),
                    "x_grid": list(x_grid),
                    "sigma2_grid": sigma2_grid,
                    "sigma_grid": sigma_grid,
                    "p_hat_mean_grid": p_hat_mean_grid,
                    "sigma2_max": float(sigma2_array[max_index]),
                    "sigma": float(math.sqrt(max(float(sigma2_array[max_index]), 0.0))),
                    "max_location": [
                        float(x_grid[max_index[0]]),
                        float(x_grid[max_index[1]]),
                    ],
                }
            )

        tau2_grid = tau2_np[run_idx].reshape(grid_size, grid_size).tolist()
        tau_grid = tau_np[run_idx].reshape(grid_size, grid_size).tolist()
        tau_mean_grid = tau_mean_np[run_idx].reshape(grid_size, grid_size).tolist()
        tau2_array = np.asarray(tau2_grid, dtype=float)
        tau_max_index = np.unravel_index(np.argmax(tau2_array), tau2_array.shape)
        tau_estimates.append(
            {
                "x_grid": list(x_grid),
                "tau2_grid": tau2_grid,
                "tau_grid": tau_grid,
                "p_hat_mean_grid": tau_mean_grid,
                "tau2_max": float(tau2_array[tau_max_index]),
                "tau": float(math.sqrt(max(float(tau2_array[tau_max_index]), 0.0))),
                "max_location": [
                    float(x_grid[tau_max_index[0]]),
                    float(x_grid[tau_max_index[1]]),
                ],
            }
        )
        sigma_estimates_by_run.append(run_estimates)

    return sigma_estimates_by_run, tau_estimates


def _build_sigma2_matrix(
    sigma_estimates: Sequence[Dict[str, Any]], tau_estimate: Dict[str, Any]
):
    if not sigma_estimates:
        raise ValueError("sigma_estimates must be non-empty")

    tau2_array = np.asarray(tau_estimate["tau2_grid"], dtype=float)
    if tau2_array.ndim != 2:
        raise ValueError(
            f"tau_estimate tau2_grid must be 2D, got shape {tau2_array.shape}"
        )
    if not np.isfinite(tau2_array).all():
        raise ValueError("tau_estimate tau2_grid contains non-finite values")

    sigma2_columns = []
    grid_shape = None
    for level_idx, estimate in enumerate(sigma_estimates):
        sigma2_array = np.asarray(estimate["sigma2_grid"], dtype=float)
        if sigma2_array.ndim != 2:
            raise ValueError(
                f"sigma_estimates[{level_idx}] sigma2_grid must be 2D, got shape {sigma2_array.shape}"
            )
        if not np.isfinite(sigma2_array).all():
            raise ValueError(
                f"sigma_estimates[{level_idx}] sigma2_grid contains non-finite values"
            )

        if grid_shape is None:
            grid_shape = sigma2_array.shape
        elif sigma2_array.shape != grid_shape:
            raise ValueError(
                "All sigma2 grids must share the same shape. "
                f"Expected {grid_shape}, got {sigma2_array.shape} at level {level_idx}."
            )

        sigma2_columns.append(np.clip(sigma2_array.reshape(-1), a_min=0.0, a_max=None))

    if tau2_array.shape != grid_shape:
        raise ValueError(
            "tau_estimate tau2_grid must match sigma grid shape. "
            f"Expected {grid_shape}, got {tau2_array.shape}."
        )

    sigma2_columns[0] = sigma2_columns[0] + np.clip(
        tau2_array.reshape(-1), a_min=0.0, a_max=None
    )

    return np.stack(sigma2_columns, axis=1), grid_shape


def _segment_costs(ddim: DDIM, split_points: Sequence[int]) -> List[float]:
    start_points = [ddim.T] + list(split_points)
    end_points = list(split_points) + [0]
    return [
        float(ddim.segment_cost(start_t, end_t))
        for start_t, end_t in zip(start_points, end_points)
    ]


def _grid_index_to_point(
    flat_index: int, x_grid: Sequence[float], grid_shape: Sequence[int]
):
    _, n_cols = grid_shape
    x1_idx, x2_idx = divmod(flat_index, n_cols)
    return [float(x_grid[x1_idx]), float(x_grid[x2_idx])]


def _normalize_simplex_with_floor(weights: np.ndarray, floor: float):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("simplex weights must be one-dimensional")
    num_weights = weights.shape[0]
    if num_weights == 0:
        raise ValueError("simplex weights must be non-empty")

    floor = min(max(float(floor), 0.0), 0.5 / float(num_weights))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.maximum(weights, 0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return np.full(num_weights, 1.0 / num_weights, dtype=float)

    weights = weights / weight_sum
    if floor <= 0.0 or np.all(weights >= floor):
        return weights

    above_floor = np.maximum(weights - floor, 0.0)
    above_sum = float(above_floor.sum())
    if above_sum <= 0.0:
        return np.full(num_weights, 1.0 / num_weights, dtype=float)

    free_mass = max(1.0 - floor * num_weights, 0.0)
    return floor + free_mass * above_floor / above_sum


def _allocation_objective_values(
    weighted_sigma2_matrix: np.ndarray, simplex_weights: np.ndarray
):
    return weighted_sigma2_matrix @ (1.0 / simplex_weights)


def _allocation_dual_objective(
    weighted_sigma2_matrix: np.ndarray, dual_weights: np.ndarray
):
    c_values = weighted_sigma2_matrix.T @ dual_weights
    return float(np.square(np.sqrt(np.maximum(c_values, 0.0)).sum()))


def _top_indices(values: np.ndarray, count: int):
    count = min(max(int(count), 0), int(values.shape[0]))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == values.shape[0]:
        return np.arange(values.shape[0], dtype=np.int64)
    return np.argpartition(values, -count)[-count:].astype(np.int64, copy=False)


def _softmax_from_values(values: np.ndarray, temperature: float):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    temperature = max(float(temperature), 1e-300)
    shifted = (values - float(np.max(values))) / temperature
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return np.full(values.shape[0], 1.0 / values.shape[0], dtype=float)
    return weights / weight_sum


def _active_primal_initial_points(
    active_matrix: np.ndarray,
    y_initial: np.ndarray,
    y_floor: float,
):
    num_active, num_levels = active_matrix.shape
    starts = [
        _normalize_simplex_with_floor(y_initial, y_floor),
        np.full(num_levels, 1.0 / num_levels, dtype=float),
    ]

    column_means = np.mean(active_matrix, axis=0) if num_active else np.ones(num_levels)
    starts.append(_normalize_simplex_with_floor(np.sqrt(column_means), y_floor))

    if num_active:
        row_scores = np.sqrt(np.maximum(active_matrix, 0.0)).sum(axis=1)
        for row_idx in _top_indices(row_scores, min(num_levels + 1, num_active)):
            starts.append(
                _normalize_simplex_with_floor(
                    np.sqrt(np.maximum(active_matrix[int(row_idx)], 0.0)), y_floor
                )
            )

    return starts


def _maximize_active_dual_weights(
    active_matrix: np.ndarray,
    initial_dual_weights: np.ndarray,
    *,
    max_iters: int = 200,
):
    num_active = active_matrix.shape[0]
    dual_weights = _normalize_simplex_with_floor(initial_dual_weights, 0.0)
    best_weights = dual_weights.copy()
    best_objective = _allocation_dual_objective(active_matrix, best_weights)
    c_floor = 1e-15

    for _ in range(max_iters):
        c_values = active_matrix.T @ dual_weights
        sqrt_c = np.sqrt(np.maximum(c_values, c_floor))
        sum_sqrt_c = float(sqrt_c.sum())
        gradient = sum_sqrt_c * np.sum(active_matrix / sqrt_c[None, :], axis=1)
        gradient_scale = float(np.max(np.abs(gradient)))
        if gradient_scale <= 0.0 or not np.isfinite(gradient_scale):
            break

        step = 0.5 / gradient_scale
        accepted = False
        for _ in range(16):
            exponent = step * gradient
            exponent = np.clip(exponent - float(np.max(exponent)), -50.0, 50.0)
            candidate_weights = _normalize_simplex_with_floor(
                dual_weights * np.exp(exponent), 0.0
            )
            candidate_objective = _allocation_dual_objective(
                active_matrix, candidate_weights
            )
            if candidate_objective >= best_objective * (1.0 - 1e-12):
                dual_weights = candidate_weights
                accepted = True
                if candidate_objective > best_objective:
                    best_weights = candidate_weights.copy()
                    best_objective = candidate_objective
                break
            step *= 0.5

        if not accepted:
            break

    if best_weights.shape[0] != num_active:
        raise RuntimeError("active dual optimizer returned invalid shape")
    return best_weights


def _solve_active_primal_inner(
    active_matrix: np.ndarray,
    y_initial: np.ndarray,
    *,
    y_floor: float,
    max_iters: int,
):
    best_y = _normalize_simplex_with_floor(y_initial, y_floor)
    best_values = _allocation_objective_values(active_matrix, best_y)
    best_objective = float(np.max(best_values))

    for start in _active_primal_initial_points(active_matrix, y_initial, y_floor):
        y = _normalize_simplex_with_floor(start, y_floor)
        values = _allocation_objective_values(active_matrix, y)
        objective = float(np.max(values))
        no_improvement = 0

        for iter_idx in range(max_iters):
            temperature_factor = max(
                1e-4, 5e-2 * (0.2 ** (iter_idx / max(max_iters - 1, 1)))
            )
            temperature = max(objective * temperature_factor, 1e-300)
            active_weights = _softmax_from_values(values, temperature)
            gradient = -(active_matrix.T @ active_weights) / np.square(y)
            gradient_scale = float(np.max(np.abs(gradient)))
            if gradient_scale <= 0.0 or not np.isfinite(gradient_scale):
                break

            step = 0.5 / gradient_scale
            accepted = False
            for _ in range(16):
                exponent = -step * gradient
                exponent = np.clip(exponent - float(np.max(exponent)), -50.0, 50.0)
                y_candidate = _normalize_simplex_with_floor(
                    y * np.exp(exponent), y_floor
                )
                candidate_values = _allocation_objective_values(
                    active_matrix, y_candidate
                )
                candidate_objective = float(np.max(candidate_values))
                if candidate_objective <= objective * (1.0 + 1e-12):
                    improvement = objective - candidate_objective
                    y = y_candidate
                    values = candidate_values
                    objective = candidate_objective
                    accepted = True
                    if improvement <= max(1e-12, 1e-10 * max(objective, 1.0)):
                        no_improvement += 1
                    else:
                        no_improvement = 0
                    break
                step *= 0.5

            if not accepted:
                no_improvement += 1
            if no_improvement >= 25:
                break

        if objective < best_objective:
            best_y = y
            best_values = values
            best_objective = objective

    final_temperature = max(best_objective * 1e-4, 1e-300)
    active_dual_weights = _softmax_from_values(best_values, final_temperature)
    active_dual_weights = _maximize_active_dual_weights(
        active_matrix, active_dual_weights
    )
    return best_y, best_objective, active_dual_weights


def _solve_active_set_primal_allocation(
    weighted_sigma2_matrix: np.ndarray,
    y_initial: np.ndarray,
    *,
    y_floor: float,
    relative_tol: float = 2e-3,
    max_outer_iters: int = 30,
    max_inner_iters: int = 300,
):
    num_points, num_levels = weighted_sigma2_matrix.shape
    if num_points == 0 or num_levels == 0:
        raise ValueError("weighted_sigma2_matrix must be non-empty")

    initial_active_count = min(num_points, max(32, 4 * num_levels))
    add_count = min(num_points, max(8, num_levels))
    max_active = min(num_points, max(256, 16 * num_levels))

    row_scores = np.sqrt(np.maximum(weighted_sigma2_matrix, 0.0)).sum(axis=1)
    active_indices = set(
        int(idx) for idx in _top_indices(row_scores, initial_active_count)
    )

    uniform_y = np.full(num_levels, 1.0 / num_levels, dtype=float)
    for probe_y in (y_initial, uniform_y):
        probe_y = _normalize_simplex_with_floor(probe_y, y_floor)
        probe_values = _allocation_objective_values(weighted_sigma2_matrix, probe_y)
        active_indices.update(
            int(idx) for idx in _top_indices(probe_values, add_count)
        )

    y_best = _normalize_simplex_with_floor(y_initial, y_floor)
    best_result = None

    for outer_iter in range(max_outer_iters):
        active_array = np.array(sorted(active_indices), dtype=np.int64)
        active_matrix = weighted_sigma2_matrix[active_array]
        y_candidate, active_upper, active_dual = _solve_active_primal_inner(
            active_matrix,
            y_best,
            y_floor=y_floor,
            max_iters=max_inner_iters,
        )

        full_values = _allocation_objective_values(weighted_sigma2_matrix, y_candidate)
        full_upper = float(np.max(full_values))

        full_dual_weights = np.zeros(num_points, dtype=float)
        full_dual_weights[active_array] = active_dual
        dual_lower = _allocation_dual_objective(
            weighted_sigma2_matrix, full_dual_weights
        )
        relative_gap = max(full_upper - dual_lower, 0.0) / max(
            abs(full_upper), 1.0
        )

        best_result = {
            "simplex_weights": y_candidate,
            "dual_weights": full_dual_weights,
            "dual_objective": dual_lower,
            "worst_case_objective": full_upper,
            "relative_gap": relative_gap,
            "outer_iterations": outer_iter + 1,
            "active_size": int(active_array.shape[0]),
            "active_objective": float(active_upper),
        }

        y_best = y_candidate
        if relative_gap <= relative_tol:
            break

        worst_indices = _top_indices(full_values, add_count)
        previous_size = len(active_indices)
        active_indices.update(int(idx) for idx in worst_indices)
        if len(active_indices) > max_active:
            keep_indices = _top_indices(full_values, max_active)
            active_indices = set(int(idx) for idx in keep_indices)
            active_indices.update(int(idx) for idx in active_array)
            if len(active_indices) > max_active:
                ranked = sorted(
                    active_indices, key=lambda idx: full_values[idx], reverse=True
                )
                active_indices = set(ranked[:max_active])

        if len(active_indices) == previous_size:
            break

    if best_result is None:
        raise RuntimeError("active-set primal allocation did not run")

    return best_result


def _solve_optimal_split_factors(
    sigma_estimates: Sequence[Dict[str, Any]],
    tau_estimate: Dict[str, Any],
    x_grid: Sequence[float],
    cost_weights: Sequence[float],
):
    t_matrix_start = time.perf_counter()
    sigma2_matrix, grid_shape = _build_sigma2_matrix(sigma_estimates, tau_estimate)
    matrix_build_time = time.perf_counter() - t_matrix_start
    t_opt_start = time.perf_counter()
    num_points, num_levels = sigma2_matrix.shape
    weight_floor = 1e-12

    cost_weights_array = np.asarray(cost_weights, dtype=float)
    if cost_weights_array.shape != (num_levels,):
        raise ValueError(
            "cost_weights must have one entry per allocation level. "
            f"Expected {(num_levels,)}, got {cost_weights_array.shape}."
        )
    if not np.isfinite(cost_weights_array).all() or np.any(cost_weights_array <= 0.0):
        raise ValueError("cost_weights must contain finite positive values")
    cost_weights_array = cost_weights_array / float(cost_weights_array.sum())

    weighted_sigma2_matrix = sigma2_matrix * cost_weights_array[None, :]

    def allocation_from_simplex_weights(simplex_weights: np.ndarray):
        simplex_weights = np.asarray(simplex_weights, dtype=float)
        simplex_weights = np.maximum(simplex_weights, weight_floor)
        simplex_weights /= float(simplex_weights.sum())
        allocation_weights = simplex_weights / cost_weights_array
        allocation_weights /= float(cost_weights_array @ allocation_weights)
        return allocation_weights

    def split_factors_from_allocation(allocation_weights: np.ndarray):
        return allocation_weights[1:] / allocation_weights[:-1]

    def worst_case_objective(allocation_weights: np.ndarray):
        return float(
            np.max(np.sum(sigma2_matrix / allocation_weights[None, :], axis=1))
        )

    if num_levels < 2:
        sigma2_max = float(np.max(sigma2_matrix))
        j_star = int(np.argmax(sigma2_matrix.sum(axis=1)))
        dual_weights = np.zeros(num_points, dtype=float)
        dual_weights[j_star] = 1.0
        return {
            "split_factors": [],
            "allocation_weights": [1.0],
            "dual_weights": dual_weights.tolist(),
            "dual_status": 0,
            "dual_success": True,
            "dual_message": "Trivial single-level allocation",
            "dual_objective": sigma2_max,
            "worst_case_simplex_objective": sigma2_max,
            "active_grid_points": [
                {
                    "point": _grid_index_to_point(j_star, x_grid, grid_shape),
                    "weight": 1.0,
                }
            ],
            "matrix_build_time": matrix_build_time,
            "optimize_time": time.perf_counter() - t_opt_start,
        }

    # Fast path: single worst-case grid point t* = argmax_j sum_k sigma_k(j)
    # gives y_k ∝ sqrt(cost_k * sigma2_k(t*)) in closed form, where
    # y_k = cost_k * M_k and sum_k y_k = 1. KKT check confirms whether this is
    # globally optimal; if not, fall back to SLSQP on the dual simplex.
    sigma_matrix = np.sqrt(np.maximum(weighted_sigma2_matrix, 0.0))
    row_sums = sigma_matrix.sum(axis=1)
    j_star = int(np.argmax(row_sums))
    sigmas_at_j = sigma_matrix[j_star]
    total = float(sigmas_at_j.sum())

    if total <= 0.0:
        allocation_weights = np.ones(num_levels, dtype=float)
        dual_weights = np.full(num_points, 1.0 / num_points, dtype=float)
        return {
            "split_factors": split_factors_from_allocation(allocation_weights).tolist(),
            "allocation_weights": allocation_weights.tolist(),
            "dual_weights": dual_weights.tolist(),
            "dual_status": 0,
            "dual_success": True,
            "dual_message": "All sigma estimates are zero; using uniform allocation",
            "dual_objective": 0.0,
            "worst_case_simplex_objective": 0.0,
            "active_grid_points": [
                {
                    "point": _grid_index_to_point(idx, x_grid, grid_shape),
                    "weight": float(weight),
                }
                for idx, weight in enumerate(dual_weights)
                if weight > 1e-9
            ],
            "matrix_build_time": matrix_build_time,
            "optimize_time": time.perf_counter() - t_opt_start,
        }

    y_candidate = sigmas_at_j / total
    kkt_tol = 1e-7
    zero_levels = y_candidate <= 0.0
    kkt_feasible = not (
        np.any(zero_levels) and np.any(weighted_sigma2_matrix[:, zero_levels] > 0.0)
    )
    if kkt_feasible:
        safe_y = np.where(y_candidate > 0.0, y_candidate, 1.0)
        per_level = np.where(
            y_candidate[None, :] > 0.0,
            weighted_sigma2_matrix / safe_y[None, :],
            0.0,
        )
        worst_case_values = per_level.sum(axis=1)
        best = float(worst_case_values[j_star])
        kkt_passes = best > 0.0 and float(worst_case_values.max()) <= best * (
            1.0 + kkt_tol
        )
    else:
        kkt_passes = False

    if kkt_passes:
        y_out = np.maximum(y_candidate, weight_floor)
        y_out /= y_out.sum()
        M_out = allocation_from_simplex_weights(y_out)
        split_factors = split_factors_from_allocation(M_out)
        dual_weights = np.zeros(num_points, dtype=float)
        dual_weights[j_star] = 1.0
        worst_case = worst_case_objective(M_out)
        return {
            "split_factors": split_factors.tolist(),
            "allocation_weights": M_out.tolist(),
            "dual_weights": dual_weights.tolist(),
            "dual_status": 0,
            "dual_success": True,
            "dual_message": "Single-atom closed form (KKT verified)",
            "dual_objective": worst_case,
            "worst_case_simplex_objective": worst_case,
            "active_grid_points": [
                {
                    "point": _grid_index_to_point(j_star, x_grid, grid_shape),
                    "weight": 1.0,
                }
            ],
            "matrix_build_time": matrix_build_time,
            "optimize_time": time.perf_counter() - t_opt_start,
        }

    if num_levels == 2:
        # One split point leaves only one scalar allocation variable. Avoid the
        # high-dimensional SLSQP dual over all grid points; directly minimize the
        # convex primal max_j a_j / y_0 + b_j / (1 - y_0), where y_k = c_k M_k.
        a = weighted_sigma2_matrix[:, 0]
        b = weighted_sigma2_matrix[:, 1]
        lo = weight_floor
        hi = 1.0 - weight_floor

        def objective(y0: float):
            return float(np.max(a / y0 + b / (1.0 - y0)))

        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        c = hi - inv_phi * (hi - lo)
        d = lo + inv_phi * (hi - lo)
        f_c = objective(c)
        f_d = objective(d)
        for _ in range(100):
            if f_c <= f_d:
                hi = d
                d = c
                f_d = f_c
                c = hi - inv_phi * (hi - lo)
                f_c = objective(c)
            else:
                lo = c
                c = d
                f_c = f_d
                d = lo + inv_phi * (hi - lo)
                f_d = objective(d)

        y0 = 0.5 * (lo + hi)
        y_out = np.array([y0, 1.0 - y0], dtype=float)
        M_out = allocation_from_simplex_weights(y_out)
        worst_case_values = np.sum(sigma2_matrix / M_out[None, :], axis=1)
        worst_case = float(np.max(worst_case_values))
        active_index = int(np.argmax(worst_case_values))
        dual_weights = np.zeros(num_points, dtype=float)
        dual_weights[active_index] = 1.0
        split_factors = split_factors_from_allocation(M_out)
        return {
            "split_factors": split_factors.tolist(),
            "allocation_weights": M_out.tolist(),
            "dual_weights": dual_weights.tolist(),
            "dual_status": 0,
            "dual_success": True,
            "dual_message": "Fast two-level primal search",
            "dual_objective": worst_case,
            "worst_case_simplex_objective": worst_case,
            "active_grid_points": [
                {
                    "point": _grid_index_to_point(active_index, x_grid, grid_shape),
                    "weight": 1.0,
                }
            ],
            "matrix_build_time": matrix_build_time,
            "optimize_time": time.perf_counter() - t_opt_start,
        }

    # KKT failed and there are at least three allocation levels. Optimize the
    # low-dimensional primal simplex with an active set of grid points, then
    # verify each candidate against the full grid. This keeps optimization cost
    # tied mostly to the number of split levels instead of the number of grid
    # points.
    active_solver_error = None
    try:
        active_relative_tol = 2e-3
        active_result = _solve_active_set_primal_allocation(
            weighted_sigma2_matrix,
            y_candidate,
            y_floor=weight_floor,
            relative_tol=active_relative_tol,
        )
        y_out = active_result["simplex_weights"]
        M_out = allocation_from_simplex_weights(y_out)
        split_factors = split_factors_from_allocation(M_out)
        dual_weights = active_result["dual_weights"]
        simplex_tol = 1e-8
        active_grid_points = [
            {
                "point": _grid_index_to_point(idx, x_grid, grid_shape),
                "weight": float(weight),
            }
            for idx, weight in enumerate(dual_weights)
            if weight > simplex_tol
        ]
        relative_gap = float(active_result["relative_gap"])
        return {
            "split_factors": split_factors.tolist(),
            "allocation_weights": M_out.tolist(),
            "dual_weights": dual_weights.tolist(),
            "dual_status": 0 if relative_gap <= active_relative_tol else 1,
            "dual_success": bool(np.isfinite(active_result["worst_case_objective"])),
            "dual_message": (
                "Active-set primal solver "
                f"(gap={relative_gap:.3e}, active={active_result['active_size']}, "
                f"outer={active_result['outer_iterations']})"
            ),
            "dual_objective": float(active_result["dual_objective"]),
            "worst_case_simplex_objective": float(
                active_result["worst_case_objective"]
            ),
            "active_grid_points": active_grid_points,
            "matrix_build_time": matrix_build_time,
            "optimize_time": time.perf_counter() - t_opt_start,
        }
    except Exception as exc:
        active_solver_error = exc

    # Last-resort fallback: solve the high-dimensional dual problem
    # max_mu (sum_k sqrt(mu^T (cost_k * sigma2[:, k])))^2 over the simplex,
    # warm-started from a single-atom point at j_star.
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "scipy is required to solve the simplex allocation problem after "
            f"active-set primal allocation failed: {active_solver_error}"
        ) from exc

    c_floor = 1e-15
    simplex_tol = 1e-8

    def _phi_and_grad(mu: np.ndarray):
        c = weighted_sigma2_matrix.T @ mu
        c_safe = np.maximum(c, c_floor)
        sqrt_c = np.sqrt(c_safe)
        sum_sqrt_c = float(np.sum(sqrt_c))
        phi = sum_sqrt_c**2
        grad = sum_sqrt_c * np.sum(weighted_sigma2_matrix / sqrt_c[None, :], axis=1)
        return phi, grad, c

    def objective(mu: np.ndarray):
        phi, _, _ = _phi_and_grad(mu)
        return -phi

    def objective_grad(mu: np.ndarray):
        _, grad, _ = _phi_and_grad(mu)
        return -grad

    initial_mu = np.full(num_points, 0.1 / num_points, dtype=float)
    initial_mu[j_star] += 0.9
    constraints = [
        {
            "type": "eq",
            "fun": lambda mu: float(np.sum(mu) - 1.0),
            "jac": lambda mu: np.ones_like(mu),
        }
    ]
    bounds = [(0.0, 1.0) for _ in range(num_points)]

    solver_result = minimize(
        objective,
        initial_mu,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-7, "maxiter": 1000},
    )

    dual_weights = np.clip(
        np.asarray(solver_result.x, dtype=float), a_min=0.0, a_max=None
    )
    dual_weight_sum = float(dual_weights.sum())
    if dual_weight_sum <= 0.0:
        raise ValueError("Simplex allocation solver returned zero dual weights")
    dual_weights /= dual_weight_sum

    _, _, c_opt = _phi_and_grad(dual_weights)
    simplex_weights = np.sqrt(np.maximum(c_opt, 0.0))
    if simplex_weights.sum() <= 0.0:
        simplex_weights = np.full(num_levels, 1.0 / num_levels, dtype=float)
    else:
        simplex_weights /= simplex_weights.sum()
    primal_weights = allocation_from_simplex_weights(simplex_weights)

    split_factors = split_factors_from_allocation(primal_weights)
    worst_case_simplex_objective = float(
        np.max(np.sum(sigma2_matrix / primal_weights[None, :], axis=1))
    )
    if not np.isfinite(worst_case_simplex_objective):
        raise ValueError(
            "Simplex allocation solver produced a non-finite objective value"
        )

    active_grid_points = [
        {
            "point": _grid_index_to_point(idx, x_grid, grid_shape),
            "weight": float(weight),
        }
        for idx, weight in enumerate(dual_weights)
        if weight > simplex_tol
    ]

    return {
        "split_factors": split_factors.tolist(),
        "allocation_weights": primal_weights.tolist(),
        "dual_weights": dual_weights.tolist(),
        "dual_status": int(getattr(solver_result, "status", 0)),
        "dual_success": bool(getattr(solver_result, "success", False)),
        "dual_message": str(getattr(solver_result, "message", "")),
        "dual_objective": float(-solver_result.fun),
        "worst_case_simplex_objective": worst_case_simplex_objective,
        "active_grid_points": active_grid_points,
        "matrix_build_time": matrix_build_time,
        "optimize_time": time.perf_counter() - t_opt_start,
    }


def _expected_cost_per_root(
    ddim: DDIM, split_points: Sequence[int], split_factors: Sequence[float]
):
    cost = float(ddim.segment_cost(ddim.T, split_points[0]))
    cumulative_split = 1.0

    for idx, split_factor in enumerate(split_factors):
        cumulative_split *= split_factor
        if idx + 1 < len(split_points):
            segment_cost = ddim.segment_cost(split_points[idx], split_points[idx + 1])
        else:
            segment_cost = ddim.segment_cost(split_points[idx], 0)
        cost += cumulative_split * segment_cost

    return cost


def _probabilistic_split_with_run_ids(
    x: torch.Tensor,
    run_ids: torch.Tensor,
    split_factors_by_run: torch.Tensor,
    generator: torch.Generator | None = None,
):
    if x.shape[0] == 0:
        return x, run_ids

    particle_factors = split_factors_by_run[run_ids]
    base_copies = torch.floor(particle_factors).to(dtype=torch.long)
    extra_probs = particle_factors - base_copies.to(dtype=particle_factors.dtype)
    has_extra = (
        torch.rand(x.shape[0], device=x.device, generator=generator) < extra_probs
    )
    counts = base_copies + has_extra.to(dtype=torch.long)

    return x.repeat_interleave(counts, dim=0), run_ids.repeat_interleave(counts, dim=0)


def _run_probabilistic_inference_batch(
    model,
    ddim: DDIM,
    n0_by_run: Sequence[int],
    split_points: Sequence[int],
    split_factors_by_run: Sequence[Sequence[float]],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    generator: torch.Generator | None = None,
):
    if len(n0_by_run) == 0:
        return [], [], 0.0
    if len(n0_by_run) != len(split_factors_by_run):
        raise ValueError("n0_by_run and split_factors_by_run must have the same length")
    if any(int(n0) < 1 for n0 in n0_by_run):
        raise ValueError("all n0 values must be at least 1")

    input_dim = int(torch.as_tensor(data_mean).numel())
    n0_tensor = torch.as_tensor(n0_by_run, device=ddim.device, dtype=torch.long)

    num_runs = int(n0_tensor.numel())
    run_ids = torch.repeat_interleave(
        torch.arange(num_runs, device=ddim.device, dtype=torch.long), n0_tensor
    )
    x = torch.randn(
        int(n0_tensor.sum().item()), input_dim, device=ddim.device, generator=generator
    )

    realized_costs = n0_tensor * int(ddim.segment_cost(ddim.T, split_points[0]))
    sampling_start = time.perf_counter()
    x = _sample_loop(ddim, model, x, ddim.T, split_points[0], generator=generator)

    split_factors_array = np.asarray(split_factors_by_run, dtype=float)
    if not np.isfinite(split_factors_array).all() or np.any(split_factors_array < 0.0):
        raise ValueError("split_factors_by_run must contain finite nonnegative values")
    split_factors_tensor = torch.as_tensor(
        split_factors_array, device=ddim.device, dtype=x.dtype
    )
    if split_factors_tensor.shape != (num_runs, len(split_points)):
        raise ValueError(
            "split_factors_by_run must have shape "
            f"({num_runs}, {len(split_points)}), got {tuple(split_factors_tensor.shape)}"
        )

    for idx, split_point in enumerate(split_points):
        x, run_ids = _probabilistic_split_with_run_ids(
            x,
            run_ids,
            split_factors_tensor[:, idx],
            generator=generator,
        )

        if idx + 1 < len(split_points):
            end_t = split_points[idx + 1]
        else:
            end_t = 0

        current_counts = torch.bincount(run_ids, minlength=num_runs)
        realized_costs = realized_costs + current_counts * int(
            ddim.segment_cost(split_point, end_t)
        )
        x = _sample_loop(ddim, model, x, split_point, end_t, generator=generator)

    x = denormalize(x, data_mean, data_std)
    sampling_time = time.perf_counter() - sampling_start

    samples_by_run = []
    for run_idx in range(num_runs):
        samples_by_run.append(x[run_ids == run_idx])

    return samples_by_run, realized_costs.detach().cpu().tolist(), sampling_time


def _summarize_results(ks_distances: np.ndarray):
    valid_mask = ~np.isnan(ks_distances)
    valid_ks = ks_distances[valid_mask]
    if valid_ks.size == 0:
        return {
            "mean_ks": float("nan"),
            "std_ks": float("nan"),
            "var_ks": float("nan"),
            "n_valid_runs": 0,
        }

    return {
        "mean_ks": float(np.mean(valid_ks)),
        "std_ks": float(np.std(valid_ks)),
        "var_ks": float(np.var(valid_ks)),
        "n_valid_runs": int(valid_ks.size),
    }


def _build_sampling_trial_result(
    samples: torch.Tensor,
    realized_cost: int,
    phase2_sampling_time: float,
    target_spec: Dict[str, Any],
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    phase1_x0_samples: torch.Tensor | None = None,
):
    return _build_sampling_trial_result_from_tensor(
        samples=samples,
        realized_cost=realized_cost,
        phase2_sampling_time=phase2_sampling_time,
        target_spec=target_spec,
        reference_cdf_state=reference_cdf_state,
        reference_mode=reference_mode,
        phase1_x0_samples=phase1_x0_samples,
    )


def _build_sampling_trial_result_from_array(
    samples_np: np.ndarray,
    realized_cost: int,
    phase2_sampling_time: float,
    target_spec: Dict[str, Any],
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    phase1_x0_samples: torch.Tensor | np.ndarray | None = None,
):
    samples_np = _coerce_samples_np(samples_np)
    leaf_count = int(samples_np.shape[0])
    ks_samples = samples_np
    if phase1_x0_samples is not None and int(phase1_x0_samples.shape[0]) > 0:
        phase1_np = _coerce_samples_np(phase1_x0_samples)
        ks_samples = np.concatenate([phase1_np, samples_np], axis=0)

    ks_start = time.perf_counter()
    ks_distance = float("nan")
    if ks_samples.shape[0] > 0:
        ks_distance, _, _ = _compute_ks_distance(
            ks_samples, target_spec, reference_mode, reference_cdf_state
        )
    phase2_ks_time = time.perf_counter() - ks_start

    return {
        "ks_distance": float(ks_distance),
        "realized_cost": int(realized_cost),
        "leaf_count": leaf_count,
        "phase2_sampling_time": float(phase2_sampling_time),
        "phase2_ks_time": float(phase2_ks_time),
        "samples": samples_np,
    }


def _build_sampling_trial_result_from_tensor(
    samples: torch.Tensor,
    realized_cost: int,
    phase2_sampling_time: float,
    target_spec: Dict[str, Any],
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    phase1_x0_samples: torch.Tensor | np.ndarray | None = None,
):
    return _build_sampling_trial_result_from_array(
        samples_np=_coerce_samples_np(samples),
        realized_cost=realized_cost,
        phase2_sampling_time=phase2_sampling_time,
        target_spec=target_spec,
        reference_cdf_state=reference_cdf_state,
        reference_mode=reference_mode,
        phase1_x0_samples=phase1_x0_samples,
    )


def _build_sampling_trial_results_parallel(
    samples_by_run: Sequence[torch.Tensor | np.ndarray],
    realized_costs: Sequence[int | float],
    phase2_sampling_time: float,
    target_spec: Dict[str, Any],
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    *,
    phase1_x0_samples_by_run: Sequence[torch.Tensor | np.ndarray | None] | None = None,
    max_workers: int = 1,
):
    if phase1_x0_samples_by_run is None:
        phase1_x0_samples_by_run = [None] * len(samples_by_run)
    if len(samples_by_run) != len(realized_costs):
        raise ValueError("samples_by_run and realized_costs must have the same length")
    if len(samples_by_run) != len(phase1_x0_samples_by_run):
        raise ValueError("phase1_x0_samples_by_run must match samples_by_run length")

    work_items = list(zip(samples_by_run, realized_costs, phase1_x0_samples_by_run))
    worker_count = max(1, min(int(max_workers), len(work_items)))
    if worker_count == 1:
        return [
            _build_sampling_trial_result_from_array(
                samples_np=_coerce_samples_np(samples),
                realized_cost=int(realized_cost),
                phase2_sampling_time=phase2_sampling_time,
                target_spec=target_spec,
                reference_cdf_state=reference_cdf_state,
                reference_mode=reference_mode,
                phase1_x0_samples=phase1_x0_samples,
            )
            for samples, realized_cost, phase1_x0_samples in work_items
        ]
    trial_results = [None] * len(work_items)
    _warm_ks_kernel_for_mode(reference_mode, target_spec)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        _submit_sampling_trial_result_futures(
            executor=executor,
            futures=futures,
            run_indices=range(len(work_items)),
            samples_by_run=samples_by_run,
            realized_costs=realized_costs,
            phase2_sampling_time=phase2_sampling_time,
            target_spec=target_spec,
            reference_cdf_state=reference_cdf_state,
            reference_mode=reference_mode,
            phase1_x0_samples_by_run=phase1_x0_samples_by_run,
        )
        _collect_sampling_trial_result_futures(trial_results, futures)
    return trial_results


def _submit_sampling_trial_result_futures(
    executor: ThreadPoolExecutor,
    futures: List[Tuple[int, Any]],
    run_indices: Sequence[int],
    samples_by_run: Sequence[torch.Tensor | np.ndarray],
    realized_costs: Sequence[int | float],
    phase2_sampling_time: float,
    target_spec: Dict[str, Any],
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    *,
    phase1_x0_samples_by_run: Sequence[torch.Tensor | np.ndarray | None] | None = None,
):
    run_indices = list(run_indices)
    if phase1_x0_samples_by_run is None:
        phase1_x0_samples_by_run = [None] * len(samples_by_run)
    if len(samples_by_run) != len(realized_costs):
        raise ValueError("samples_by_run and realized_costs must have the same length")
    if len(samples_by_run) != len(phase1_x0_samples_by_run):
        raise ValueError("phase1_x0_samples_by_run must match samples_by_run length")
    if len(samples_by_run) != len(run_indices):
        raise ValueError("run_indices must match samples_by_run length")

    samples_cpu = [_coerce_samples_np(samples) for samples in samples_by_run]
    phase1_cpu = [
        None if samples is None else _coerce_samples_np(samples)
        for samples in phase1_x0_samples_by_run
    ]
    for local_idx, run_idx in enumerate(run_indices):
        future = executor.submit(
            _build_sampling_trial_result_from_array,
            samples_cpu[local_idx],
            int(realized_costs[local_idx]),
            phase2_sampling_time,
            target_spec,
            reference_cdf_state,
            reference_mode,
            phase1_cpu[local_idx],
        )
        futures.append((int(run_idx), future))


def _collect_sampling_trial_result_futures(
    trial_results: List[Dict[str, Any] | None],
    futures: Sequence[Tuple[int, Any]],
    *,
    desc: str = "KS",
):
    for run_idx, future in tqdm(futures, desc=desc, leave=False):
        trial_results[run_idx] = future.result()


def _summarize_sampling_trials(trial_results: Sequence[Dict[str, Any]]):
    ks_distances = np.array(
        [trial["ks_distance"] for trial in trial_results], dtype=float
    )
    realized_costs = np.array(
        [trial["realized_cost"] for trial in trial_results], dtype=float
    )
    leaf_counts = np.array(
        [trial["leaf_count"] for trial in trial_results], dtype=float
    )
    phase2_sampling_times = np.array(
        [trial["phase2_sampling_time"] for trial in trial_results], dtype=float
    )
    phase2_ks_times = np.array(
        [trial["phase2_ks_time"] for trial in trial_results], dtype=float
    )
    stats = _summarize_results(ks_distances)
    extinction_rate = float(np.mean(leaf_counts == 0.0)) if leaf_counts.size else 0.0

    return {
        "mean_ks": stats["mean_ks"],
        "std_ks": stats["std_ks"],
        "var_ks": stats["var_ks"],
        "n_valid_runs": stats["n_valid_runs"],
        "ks_distances": ks_distances.tolist(),
        "realized_costs": realized_costs.tolist(),
        "realized_cost_mean": (
            float(realized_costs.mean()) if realized_costs.size else 0.0
        ),
        "realized_cost_std": (
            float(realized_costs.std()) if realized_costs.size else 0.0
        ),
        "leaf_counts": leaf_counts.astype(int).tolist(),
        "leaf_count_mean": float(leaf_counts.mean()) if leaf_counts.size else 0.0,
        "leaf_count_std": float(leaf_counts.std()) if leaf_counts.size else 0.0,
        "extinction_rate": float(extinction_rate),
        "phase2_sampling_times": phase2_sampling_times.tolist(),
        "phase2_sampling_time_mean": (
            float(phase2_sampling_times.mean()) if phase2_sampling_times.size else 0.0
        ),
        "phase2_sampling_time_std": (
            float(phase2_sampling_times.std()) if phase2_sampling_times.size else 0.0
        ),
        "phase2_ks_times": phase2_ks_times.tolist(),
        "phase2_ks_time_mean": (
            float(phase2_ks_times.mean()) if phase2_ks_times.size else 0.0
        ),
        "phase2_ks_time_std": (
            float(phase2_ks_times.std()) if phase2_ks_times.size else 0.0
        ),
        "samples": [trial["samples"] for trial in trial_results],
    }


def _sample_dpmpp_2m(model, x: torch.Tensor, T: int, sampling_steps: int):
    from diffusers import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=T,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    scheduler.set_timesteps(sampling_steps, device=x.device)
    # Avoid scheduler step-index initialization from a CUDA scalar timestep; that
    # path can synchronize every step. The scheduler increments this internally.
    scheduler._step_index = 0

    with torch.inference_mode():
        for t in scheduler.timesteps:
            if isinstance(t, torch.Tensor):
                t_tensor = t.to(device=x.device, dtype=torch.long).expand(x.shape[0])
            else:
                t_tensor = torch.full(
                    (x.shape[0],), int(t), device=x.device, dtype=torch.long
                )
            eps_pred = model(x, t_tensor)
            x = scheduler.step(eps_pred, t, x).prev_sample
    return x


def _sample_full_solver(
    model,
    solver: str,
    x: torch.Tensor,
    *,
    T: int,
    sampling_steps: int,
    eta: float,
    device: str,
    generator: torch.Generator | None = None,
):
    if solver == "ddim":
        ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
        return ddim.sample_loop(model, x, ddim.T, 0, generator=generator)
    if solver == "dpmpp_2m":
        return _sample_dpmpp_2m(model, x, T=T, sampling_steps=sampling_steps)
    raise ValueError(f"Unknown solver baseline '{solver}'")


def _run_solver_baseline_batch(
    model,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    *,
    solver: str,
    chunk_size: int,
    n0: int,
    T: int,
    sampling_steps: int,
    eta: float,
    device: str,
    generator: torch.Generator | None = None,
):
    input_dim = _model_input_dim(model)
    sampling_start = time.perf_counter()
    x = torch.randn(chunk_size * n0, input_dim, device=device, generator=generator)
    samples = _sample_full_solver(
        model,
        solver,
        x,
        T=T,
        sampling_steps=sampling_steps,
        eta=eta,
        device=device,
        generator=generator,
    )
    samples = denormalize(samples, data_mean, data_std)
    sampling_time = time.perf_counter() - sampling_start
    samples = _coerce_samples_tensor(samples, device=device).reshape(chunk_size, n0, -1)
    return [samples[i] for i in range(chunk_size)], sampling_time


def run_solver_baseline_sampling(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    solver: str,
    B: int,
    T: int,
    sampling_steps: int,
    eta: float = 0.0,
    n_runs: int,
    seed: int | None,
    device: str,
    reference_mode: str = "ddpm_samples",
    debug: bool = False,
    n_parallel: int = 1,
    run_start_index: int = 0,
    return_trial_results: bool = False,
):
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be at least 1")

    n0 = int(B // sampling_steps)
    if n0 < 1:
        raise ValueError(
            f"Budget B={B} is too small for solver baseline with {sampling_steps} steps"
        )

    _debug_log(
        debug,
        "Starting solver baseline "
        f"with solver={solver}, B={B}, sampling_steps={sampling_steps}, eta={eta}",
    )
    _debug_log(
        debug,
        "Solver baseline sampling setup: "
        f"n0={n0}, expected_cost_per_root={sampling_steps:.6f}, "
        f"expected_total_samples={float(n0):.6f}",
    )

    trial_results = [None] * n_runs
    chunks = list(_iter_run_chunks(n_runs, n_parallel))
    ks_futures = []
    _warm_ks_kernel_for_mode(reference_mode, target_spec)
    ks_workers = max(1, min(int(n_parallel), n_runs))
    with ThreadPoolExecutor(max_workers=ks_workers) as ks_executor:
        for start, end in tqdm(chunks, desc="Runs", leave=False):
            chunk_size = end - start
            run_seed = (
                None
                if seed is None
                else seed + PHASE2_SEED_OFFSET + run_start_index + start
            )
            samples_by_run, sampling_time = _run_solver_baseline_batch(
                model=model,
                data_mean=data_mean,
                data_std=data_std,
                solver=solver,
                chunk_size=chunk_size,
                n0=n0,
                T=T,
                sampling_steps=sampling_steps,
                eta=eta,
                device=device,
                generator=_make_torch_generator(run_seed, device),
            )
            phase2_sampling_time = sampling_time / chunk_size
            _submit_sampling_trial_result_futures(
                executor=ks_executor,
                futures=ks_futures,
                run_indices=range(start, end),
                samples_by_run=samples_by_run,
                realized_costs=[int(n0 * sampling_steps)] * chunk_size,
                phase2_sampling_time=phase2_sampling_time,
                target_spec=target_spec,
                reference_cdf_state=reference_cdf_state,
                reference_mode=reference_mode,
            )
        _collect_sampling_trial_result_futures(trial_results, ks_futures)

    stats = _summarize_sampling_trials(trial_results)
    _debug_log(
        debug,
        f"Solver baseline sampling complete: mean_ks={stats['mean_ks']:.6f}, "
        f"extinction_rate={stats['extinction_rate']:.6f}",
    )
    expected_total_samples = float(n0)
    result = {
        "mode": "solver_baseline",
        "solver": solver,
        "B": int(B),
        "B1": 0,
        "used_B1": 0,
        "B2": int(B),
        "sampling_steps": int(sampling_steps),
        "eta": float(eta),
        "reference_mode": reference_mode,
        "solver_nfe": int(sampling_steps),
        "N_i": [],
        "n0": int(n0),
        "final_root_count": int(n0),
        "expected_cost_per_root": float(sampling_steps),
        "expected_total_samples": expected_total_samples,
        "target_spec": target_spec,
        **stats,
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result


def _mean_scalar(values: Sequence[float | int]):
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _mean_vector(values: Sequence[Sequence[float | int]]):
    if not values:
        return []
    return np.mean(np.asarray(values, dtype=float), axis=0).tolist()


def _std_vector(values: Sequence[Sequence[float | int]]):
    if not values:
        return []
    return np.std(np.asarray(values, dtype=float), axis=0).tolist()


def _mean_grid(values: Sequence[Sequence[Sequence[float]]]):
    if not values:
        return []
    return np.mean(np.asarray(values, dtype=float), axis=0).tolist()


def _average_sigma_estimates(
    sigma_estimates_by_run: Sequence[Sequence[Dict[str, Any]]], x_grid: Sequence[float]
):
    if not sigma_estimates_by_run:
        return []

    averaged_estimates = []
    for level_idx in range(len(sigma_estimates_by_run[0])):
        sigma2_grid = _mean_grid(
            [run[level_idx]["sigma2_grid"] for run in sigma_estimates_by_run]
        )
        sigma_grid = _mean_grid(
            [run[level_idx]["sigma_grid"] for run in sigma_estimates_by_run]
        )
        p_hat_mean_grid = _mean_grid(
            [run[level_idx]["p_hat_mean_grid"] for run in sigma_estimates_by_run]
        )
        sigma2_array = np.asarray(sigma2_grid, dtype=float)
        max_index = np.unravel_index(np.argmax(sigma2_array), sigma2_array.shape)
        averaged_estimates.append(
            {
                "time": int(sigma_estimates_by_run[0][level_idx]["time"]),
                "x_grid": list(x_grid),
                "sigma2_grid": sigma2_grid,
                "sigma_grid": sigma_grid,
                "p_hat_mean_grid": p_hat_mean_grid,
                "sigma2_max": float(sigma2_array[max_index]),
                "sigma": float(math.sqrt(max(float(sigma2_array[max_index]), 0.0))),
                "max_location": [
                    float(x_grid[max_index[0]]),
                    float(x_grid[max_index[1]]),
                ],
            }
        )

    return averaged_estimates


def _average_tau_estimate(
    tau_estimates: Sequence[Dict[str, Any]], x_grid: Sequence[float]
):
    if not tau_estimates:
        return {}

    tau2_grid = _mean_grid(
        [tau_estimate["tau2_grid"] for tau_estimate in tau_estimates]
    )
    tau_grid = _mean_grid([tau_estimate["tau_grid"] for tau_estimate in tau_estimates])
    p_hat_mean_grid = _mean_grid(
        [tau_estimate["p_hat_mean_grid"] for tau_estimate in tau_estimates]
    )
    tau2_array = np.asarray(tau2_grid, dtype=float)
    max_index = np.unravel_index(np.argmax(tau2_array), tau2_array.shape)
    return {
        "x_grid": list(x_grid),
        "tau2_grid": tau2_grid,
        "tau_grid": tau_grid,
        "p_hat_mean_grid": p_hat_mean_grid,
        "tau2_max": float(tau2_array[max_index]),
        "tau": float(math.sqrt(max(float(tau2_array[max_index]), 0.0))),
        "max_location": [
            float(x_grid[max_index[0]]),
            float(x_grid[max_index[1]]),
        ],
    }


def _summarize_allocation_messages(messages: Sequence[str]):
    unique_messages = list(dict.fromkeys(messages))
    if not unique_messages:
        return ""
    if len(unique_messages) == 1:
        return unique_messages[0]
    return "Multiple allocation solver messages across runs"


def _active_grid_points_from_dual_weights(
    dual_weights: Sequence[float], x_grid: Sequence[float], grid_shape: Sequence[int]
):
    simplex_tol = 1e-8
    return [
        {
            "point": _grid_index_to_point(point_idx, x_grid, grid_shape),
            "weight": float(weight),
        }
        for point_idx, weight in enumerate(dual_weights)
        if weight > simplex_tol
    ]


def _run_sampling_trials(
    model,
    target_spec: Dict[str, Any],
    ddim: DDIM,
    n_runs: int,
    n0: int,
    split_points: Sequence[int],
    split_factors: Sequence[float],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    seed: int | None = None,
    debug: bool = False,
    n_parallel: int = 1,
    run_start_index: int = 0,
    return_trial_results: bool = False,
):
    _debug_log(
        debug,
        f"Sampling phase: running {n_runs} trials with n0={n0} and split factors {list(split_factors)}",
    )

    trial_results = [None] * n_runs
    chunks = list(_iter_run_chunks(n_runs, n_parallel))
    ks_futures = []
    _warm_ks_kernel_for_mode(reference_mode, target_spec)
    ks_workers = max(1, min(int(n_parallel), n_runs))
    with ThreadPoolExecutor(max_workers=ks_workers) as ks_executor:
        for start, end in tqdm(chunks, desc="Runs", leave=False):
            chunk_size = end - start
            run_seed = (
                None
                if seed is None
                else seed + PHASE2_SEED_OFFSET + run_start_index + start
            )
            samples_by_run, realized_costs, sampling_time = (
                _run_probabilistic_inference_batch(
                    model=model,
                    ddim=ddim,
                    n0_by_run=[n0] * chunk_size,
                    split_points=split_points,
                    split_factors_by_run=[split_factors] * chunk_size,
                    data_mean=data_mean,
                    data_std=data_std,
                    generator=_make_torch_generator(run_seed, ddim.device),
                )
            )
            phase2_sampling_time = sampling_time / chunk_size
            _submit_sampling_trial_result_futures(
                executor=ks_executor,
                futures=ks_futures,
                run_indices=range(start, end),
                samples_by_run=samples_by_run,
                realized_costs=realized_costs,
                phase2_sampling_time=phase2_sampling_time,
                target_spec=target_spec,
                reference_cdf_state=reference_cdf_state,
                reference_mode=reference_mode,
            )
        _collect_sampling_trial_result_futures(trial_results, ks_futures)

    ks_distances = []
    realized_costs = []
    leaf_counts = []
    extinction_count = 0
    samples_by_run = []
    for trial_result in trial_results:
        realized_costs.append(trial_result["realized_cost"])
        leaf_counts.append(trial_result["leaf_count"])
        samples_by_run.append(trial_result["samples"])

        if trial_result["leaf_count"] == 0:
            extinction_count += 1
            ks_distances.append(float("nan"))
            continue

        ks_distances.append(float(trial_result["ks_distance"]))

    ks_distances = np.array(ks_distances, dtype=float)
    realized_costs = np.array(realized_costs, dtype=float)
    leaf_counts = np.array(leaf_counts, dtype=float)
    stats = _summarize_results(ks_distances)
    extinction_rate = extinction_count / max(n_runs, 1)

    result = {
        "mean_ks": stats["mean_ks"],
        "std_ks": stats["std_ks"],
        "var_ks": stats["var_ks"],
        "n_valid_runs": stats["n_valid_runs"],
        "ks_distances": ks_distances.tolist(),
        "realized_costs": realized_costs.tolist(),
        "realized_cost_mean": (
            float(realized_costs.mean()) if realized_costs.size else 0.0
        ),
        "realized_cost_std": (
            float(realized_costs.std()) if realized_costs.size else 0.0
        ),
        "leaf_counts": leaf_counts.astype(int).tolist(),
        "leaf_count_mean": float(leaf_counts.mean()) if leaf_counts.size else 0.0,
        "leaf_count_std": float(leaf_counts.std()) if leaf_counts.size else 0.0,
        "extinction_rate": float(extinction_rate),
        "samples": samples_by_run,
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result


def _run_pilot_tree_phase1_sampling_batch(
    model,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    *,
    B1: int,
    T: int,
    sampling_steps: int,
    eta: float,
    split_percentages: Sequence[float],
    x_grid: Sequence[float],
    bias_type: str,
    reuse_phase1_samples: bool,
    device: str,
    chunk_size: int,
    debug: bool = False,
    generator: torch.Generator | None = None,
):
    _validate_bias_type(bias_type)
    ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
    resolved_step_points, split_points = _resolve_split_percentages(
        ddim, split_percentages
    )
    sigma_times = [T] + list(split_points)
    pilot_scale, pilot_roots, derived_m_pilot = _derive_pilot_tree_shape(
        ddim=ddim,
        split_points=split_points,
        B1=B1,
    )

    _debug_log(
        debug,
        "Phase 1 batch (pilot tree): "
        f"chunk_size={chunk_size}, pilot_scale={pilot_scale:.6f}, "
        f"pilot_roots={pilot_roots}, derived_m_pilot={derived_m_pilot:.6f}",
    )

    diffusion_start = time.perf_counter()
    (
        leaf_x0,
        leaf_run_ids,
        child_counts_by_level,
        parent_run_ids_by_level,
        used_B1_by_run,
    ) = _build_pilot_tree_batch(
        model=model,
        ddim=ddim,
        chunk_size=chunk_size,
        pilot_roots=pilot_roots,
        split_points=split_points,
        m_pilot=derived_m_pilot,
        generator=generator,
    )
    leaf_x0 = denormalize(leaf_x0, data_mean, data_std)
    diffusion_time = (time.perf_counter() - diffusion_start) / chunk_size

    estimation_start = time.perf_counter()
    sigma_estimates_by_run, tau_estimates = _estimate_sigmas_from_tree_batch(
        leaf_x0=leaf_x0,
        leaf_run_ids=leaf_run_ids,
        sigma_times=sigma_times,
        child_counts_by_level=child_counts_by_level,
        parent_run_ids_by_level=parent_run_ids_by_level,
        x_grid=x_grid,
        chunk_size=chunk_size,
        bias_type=bias_type,
    )
    estimation_time = (time.perf_counter() - estimation_start) / chunk_size

    phase1_x0_by_run = [None] * chunk_size
    if reuse_phase1_samples:
        leaf_counts_by_run = (
            torch.bincount(leaf_run_ids, minlength=chunk_size).detach().cpu().tolist()
        )
        phase1_x0_by_run = [
            _coerce_samples_np(run_samples)
            for run_samples in torch.split(leaf_x0, leaf_counts_by_run)
        ]

    phase1_payloads = []
    for run_idx in range(chunk_size):
        phase1_payloads.append(
            {
                "resolved_step_points": [int(x) for x in resolved_step_points],
                "split_points": [int(x) for x in split_points],
                "sigma_times": [int(x) for x in sigma_times],
                "bias_type": bias_type,
                "sigma_estimates": sigma_estimates_by_run[run_idx],
                "tau_estimate": tau_estimates[run_idx],
                "used_B1": int(used_B1_by_run[run_idx]),
                "phase1_x0_samples": phase1_x0_by_run[run_idx],
                "phase1_diffusion_time": float(diffusion_time),
                "phase1_estimation_time": float(estimation_time),
                "pilot_roots": int(pilot_roots),
                "m_pilot": float(derived_m_pilot),
            }
        )
    return phase1_payloads


def _run_independent_phase1_sampling_batch(
    model,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    *,
    B1: int,
    T: int,
    sampling_steps: int,
    eta: float,
    split_percentages: Sequence[float],
    x_grid: Sequence[float],
    independent_n2: int,
    bias_type: str,
    reuse_phase1_samples: bool,
    device: str,
    chunk_size: int,
    debug: bool = False,
    generator: torch.Generator | None = None,
):
    _validate_bias_type(bias_type)
    ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
    resolved_step_points, split_points = _resolve_split_percentages(
        ddim, split_percentages
    )
    sigma_times = [T] + list(split_points)
    _, counts_by_sigma, _ = _derive_independent_counts(
        ddim=ddim,
        sigma_times=sigma_times,
        B1=B1,
        independent_n2=independent_n2,
        bias_type=bias_type,
    )
    _debug_log(
        debug,
        "Phase 1 batch (independent): "
        f"chunk_size={chunk_size}, counts_by_sigma={counts_by_sigma}",
    )

    input_dim = _model_input_dim(model)
    next_times = list(sigma_times[1:]) + [0]
    sigma_estimates_by_run = [[] for _ in range(chunk_size)]
    tau_estimates = [None] * chunk_size
    phase1_samples_by_run = [[] for _ in range(chunk_size)]
    used_B1 = 0
    diffusion_time = 0.0
    estimation_time_by_run = [0.0] * chunk_size

    for idx, ((t_curr, t_next), count_spec) in enumerate(
        zip(zip(sigma_times, next_times), counts_by_sigma)
    ):
        outer_count = int(count_spec["outer_count"])
        middle_count = int(count_spec["middle_count"])
        inner_count = int(count_spec["inner_count"])
        leaf_count_per_run = outer_count * middle_count * inner_count

        sampling_start = time.perf_counter()
        root_x_T = torch.randn(
            chunk_size * outer_count,
            input_dim,
            device=device,
            generator=generator,
        )
        x_curr = _sample_loop(
            ddim, model, root_x_T, ddim.T, t_curr, generator=generator
        )
        middle_counts = torch.full(
            (x_curr.shape[0],), middle_count, device=device, dtype=torch.long
        )
        x_curr_rep = _repeat_by_counts(x_curr, middle_counts)
        x_next = _sample_loop(
            ddim, model, x_curr_rep, t_curr, t_next, generator=generator
        )
        inner_counts = torch.full(
            (x_next.shape[0],), inner_count, device=device, dtype=torch.long
        )
        x_next_rep = _repeat_by_counts(x_next, inner_counts)
        x_0 = _sample_loop(ddim, model, x_next_rep, t_next, 0, generator=generator)
        x_0 = denormalize(x_0, data_mean, data_std)
        diffusion_time += (time.perf_counter() - sampling_start) / chunk_size

        if reuse_phase1_samples:
            x0_by_run = x_0.reshape(chunk_size, leaf_count_per_run, input_dim)
            for run_idx in range(chunk_size):
                phase1_samples_by_run[run_idx].append(x0_by_run[run_idx])

        used_B1 += _independent_sigma_cost(
            ddim, t_curr, t_next, outer_count, middle_count, inner_count
        )

        estimation_start = time.perf_counter()
        indicators = _rectangle_indicator_grid(x_0, x_grid)
        num_grid_points = indicators.shape[1]
        indicators = indicators.reshape(
            chunk_size, outer_count, middle_count, inner_count, num_grid_points
        )
        middle_means = indicators.mean(dim=3)
        if middle_count < 2:
            raise ValueError("Independent sigma estimation needs middle_count >= 2")

        p_hat_mean = middle_means.mean(dim=(1, 2))
        outer_vars = torch.clamp(middle_means.var(dim=2, unbiased=True), min=0.0)
        raw_sigma2 = outer_vars.mean(dim=1)
        if (
            bias_type == "unbiased"
            and inner_count >= 2
            and ddim.segment_cost(t_next, 0) > 0
        ):
            inner_vars = torch.clamp(indicators.var(dim=3, unbiased=True), min=0.0)
            sigma2 = raw_sigma2 - inner_vars.mean(dim=(1, 2)) / float(inner_count)
        else:
            sigma2 = raw_sigma2
        sigma2 = torch.clamp(sigma2, min=0.0)
        sigma = torch.sqrt(torch.clamp(sigma2, min=0.0))
        outer_means = middle_means.mean(dim=2)
        if outer_count >= 2:
            tau2 = torch.clamp(outer_means.var(dim=1, unbiased=True), min=0.0)
        else:
            tau2 = torch.zeros(
                chunk_size, num_grid_points, device=device, dtype=x_0.dtype
            )
        tau = torch.sqrt(torch.clamp(tau2, min=0.0))
        tau_mean = outer_means.mean(dim=1)
        estimation_elapsed = (time.perf_counter() - estimation_start) / chunk_size

        for run_idx in range(chunk_size):
            sigma_estimates_by_run[run_idx].append(
                _sigma_estimate_from_flat(
                    int(t_curr),
                    sigma2[run_idx],
                    sigma[run_idx],
                    p_hat_mean[run_idx],
                    x_grid,
                )
            )
            estimation_time_by_run[run_idx] += estimation_elapsed
            if idx == 0:
                tau_estimates[run_idx] = _tau_estimate_from_flat(
                    tau2[run_idx], tau[run_idx], tau_mean[run_idx], x_grid
                )

    phase1_payloads = []
    for run_idx in range(chunk_size):
        if reuse_phase1_samples and phase1_samples_by_run[run_idx]:
            phase1_x0_samples = _coerce_samples_np(
                torch.cat(phase1_samples_by_run[run_idx], dim=0)
            )
        else:
            phase1_x0_samples = None
        phase1_payloads.append(
            {
                "resolved_step_points": [int(x) for x in resolved_step_points],
                "split_points": [int(x) for x in split_points],
                "sigma_times": [int(x) for x in sigma_times],
                "bias_type": bias_type,
                "sigma_estimates": sigma_estimates_by_run[run_idx],
                "tau_estimate": tau_estimates[run_idx],
                "used_B1": int(used_B1),
                "phase1_x0_samples": phase1_x0_samples,
                "phase1_diffusion_time": float(diffusion_time),
                "phase1_estimation_time": float(estimation_time_by_run[run_idx]),
                "pilot_roots": int(counts_by_sigma[0]["outer_count"]),
                "m_pilot": None,
            }
        )
    return phase1_payloads


def _solve_phase1_allocation(
    payload: Dict[str, Any],
    *,
    B: int,
    T: int,
    sampling_steps: int,
    eta: float,
    x_grid: Sequence[float],
):
    ddim = DDIM(T=T, device="cpu", eta=eta, sampling_steps=sampling_steps)
    segment_costs = _segment_costs(ddim, payload["split_points"])
    segment_cost_sum = float(np.sum(segment_costs))
    if segment_cost_sum <= 0.0:
        raise ValueError("Segment costs must have positive total cost")
    cost_weights = [float(cost / segment_cost_sum) for cost in segment_costs]

    allocation_result = _solve_optimal_split_factors(
        sigma_estimates=payload["sigma_estimates"],
        tau_estimate=payload["tau_estimate"],
        x_grid=x_grid,
        cost_weights=cost_weights,
    )
    split_factors = allocation_result["split_factors"]
    expected_cost_per_root = _expected_cost_per_root(
        ddim, payload["split_points"], split_factors
    )
    B2 = B - int(payload["used_B1"])
    n0 = int(B2 // expected_cost_per_root)
    if n0 < 1:
        raise ValueError(
            f"Remaining budget B2={B2} is too small; expected cost per root is {expected_cost_per_root:.6f}"
        )

    expected_total_samples = float(n0)
    for split_factor in split_factors:
        expected_total_samples *= split_factor

    return {
        **payload,
        "optimal_M_k": [float(x) for x in allocation_result["allocation_weights"]],
        "allocation_dual_weights": [
            float(x) for x in allocation_result["dual_weights"]
        ],
        "allocation_solver_success": bool(allocation_result["dual_success"]),
        "allocation_solver_status": int(allocation_result["dual_status"]),
        "allocation_solver_message": allocation_result["dual_message"],
        "allocation_dual_objective": float(allocation_result["dual_objective"]),
        "allocation_worst_case_simplex_objective": float(
            allocation_result["worst_case_simplex_objective"]
        ),
        "allocation_segment_costs": [float(x) for x in segment_costs],
        "allocation_cost_weights": [float(x) for x in cost_weights],
        "N_i": [float(x) for x in split_factors],
        "n0": int(n0),
        "final_root_count": int(n0),
        "expected_cost_per_root": float(expected_cost_per_root),
        "expected_total_samples": float(expected_total_samples),
        "B2": int(B2),
        "phase1_sigma_matrix_time": float(allocation_result["matrix_build_time"]),
        "phase1_optimize_time": float(allocation_result["optimize_time"]),
    }


def _solve_phase1_allocations_threaded(
    payloads: Sequence[Dict[str, Any]],
    *,
    B: int,
    T: int,
    sampling_steps: int,
    eta: float,
    x_grid: Sequence[float],
):
    if not payloads:
        return []

    def solve_one(payload):
        return _solve_phase1_allocation(
            payload,
            B=B,
            T=T,
            sampling_steps=sampling_steps,
            eta=eta,
            x_grid=x_grid,
        )

    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        return list(executor.map(solve_one, payloads))


def run_estimate_and_sample(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    B: int,
    B1: int,
    T: int,
    sampling_steps: int,
    eta: float,
    split_percentages: Sequence[float],
    x_grid: Sequence[float],
    independent_n2: int,
    sigma_estimation_mode: str,
    bias_type: str,
    reuse_phase1_samples: bool,
    n_runs: int,
    seed: int | None,
    device: str,
    reference_mode: str = "ddpm_samples",
    debug: bool = False,
    n_parallel: int = 1,
    run_start_index: int = 0,
    return_trial_results: bool = False,
):
    if B1 < 0:
        raise ValueError("B1 must be nonnegative")
    if B1 >= B:
        raise ValueError("B1 must be strictly smaller than B")
    _validate_bias_type(bias_type)
    if sigma_estimation_mode == "independent" and independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")

    _debug_log(
        debug,
        "Starting estimate_and_sample run "
        f"with B={B}, B1={B1}, sampling_steps={sampling_steps}, eta={eta}, "
        f"bias_type={bias_type}",
    )

    chunks = list(_iter_run_chunks(n_runs, n_parallel))
    trial_results = [None] * n_runs
    allocation_futures = []
    allocation_workers = max(1, min(n_runs, n_parallel))

    with ThreadPoolExecutor(max_workers=allocation_workers) as allocation_executor:
        if sigma_estimation_mode == "pilot_tree":
            for start, end in tqdm(chunks, desc="Phase 1", leave=False):
                chunk_size = end - start
                run_seed = None if seed is None else seed + run_start_index + start
                phase1_payloads = _run_pilot_tree_phase1_sampling_batch(
                    model=model,
                    data_mean=data_mean,
                    data_std=data_std,
                    B1=B1,
                    T=T,
                    sampling_steps=sampling_steps,
                    eta=eta,
                    split_percentages=split_percentages,
                    x_grid=x_grid,
                    bias_type=bias_type,
                    reuse_phase1_samples=reuse_phase1_samples,
                    device=device,
                    chunk_size=chunk_size,
                    debug=debug,
                    generator=_make_torch_generator(run_seed, device),
                )
                # Overlap CPU allocation solves with subsequent GPU Phase 1 chunks.
                for local_idx, payload in enumerate(phase1_payloads):
                    run_idx = start + local_idx
                    future = allocation_executor.submit(
                        _solve_phase1_allocation,
                        payload,
                        B=B,
                        T=T,
                        sampling_steps=sampling_steps,
                        eta=eta,
                        x_grid=x_grid,
                    )
                    allocation_futures.append((run_idx, future))
        else:
            for start, end in tqdm(chunks, desc="Phase 1", leave=False):
                chunk_size = end - start
                run_seed = None if seed is None else seed + run_start_index + start
                phase1_payloads = _run_independent_phase1_sampling_batch(
                    model=model,
                    data_mean=data_mean,
                    data_std=data_std,
                    B1=B1,
                    T=T,
                    sampling_steps=sampling_steps,
                    eta=eta,
                    split_percentages=split_percentages,
                    x_grid=x_grid,
                    independent_n2=independent_n2,
                    bias_type=bias_type,
                    reuse_phase1_samples=reuse_phase1_samples,
                    device=device,
                    chunk_size=chunk_size,
                    debug=debug,
                    generator=_make_torch_generator(run_seed, device),
                )
                # Overlap CPU allocation solves with subsequent GPU Phase 1 chunks.
                for local_idx, payload in enumerate(phase1_payloads):
                    run_idx = start + local_idx
                    future = allocation_executor.submit(
                        _solve_phase1_allocation,
                        payload,
                        B=B,
                        T=T,
                        sampling_steps=sampling_steps,
                        eta=eta,
                        x_grid=x_grid,
                    )
                    allocation_futures.append((run_idx, future))

        for run_idx, future in tqdm(
            allocation_futures, desc="Optimize N_i", leave=False
        ):
            trial_results[run_idx] = future.result()

    if any(trial is None for trial in trial_results):
        raise RuntimeError("Phase 1 allocation did not produce all trial results")

    ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
    ks_futures = []
    _warm_ks_kernel_for_mode(reference_mode, target_spec)
    ks_workers = max(1, min(int(n_parallel), n_runs))
    with ThreadPoolExecutor(max_workers=ks_workers) as ks_executor:
        for start, end in tqdm(chunks, desc="Phase 2", leave=False):
            chunk = trial_results[start:end]
            chunk_size = end - start
            run_seed = (
                None
                if seed is None
                else seed + PHASE2_SEED_OFFSET + run_start_index + start
            )
            samples_by_run, realized_costs, sampling_time = (
                _run_probabilistic_inference_batch(
                    model=model,
                    ddim=ddim,
                    n0_by_run=[int(trial["n0"]) for trial in chunk],
                    split_points=chunk[0]["split_points"],
                    split_factors_by_run=[trial["N_i"] for trial in chunk],
                    data_mean=data_mean,
                    data_std=data_std,
                    generator=_make_torch_generator(run_seed, device),
                )
            )
            phase2_sampling_time = sampling_time / chunk_size
            _submit_sampling_trial_result_futures(
                executor=ks_executor,
                futures=ks_futures,
                run_indices=range(start, end),
                samples_by_run=samples_by_run,
                realized_costs=realized_costs,
                phase2_sampling_time=phase2_sampling_time,
                target_spec=target_spec,
                reference_cdf_state=reference_cdf_state,
                reference_mode=reference_mode,
                phase1_x0_samples_by_run=[
                    trial.get("phase1_x0_samples") for trial in chunk
                ],
            )

        for run_idx, future in tqdm(ks_futures, desc="KS", leave=False):
            sampling_result = future.result()
            trial = trial_results[run_idx]
            trial.update(sampling_result)
            print(
                f"[timings] diffusion={trial['phase1_diffusion_time']:.3f}s "
                f"estimate_sigmas={trial['phase1_estimation_time']:.3f}s "
                f"sigma_matrix={trial['phase1_sigma_matrix_time']:.4f}s "
                f"optimize_N_i={trial['phase1_optimize_time']:.4f}s "
                f"phase2_sampling={sampling_result['phase2_sampling_time']:.3f}s "
                f"phase2_ks={sampling_result['phase2_ks_time']:.3f}s"
            )

    stats = _summarize_sampling_trials(trial_results)
    sigma_estimates = _average_sigma_estimates(
        [trial["sigma_estimates"] for trial in trial_results], x_grid
    )
    tau_estimate = _average_tau_estimate(
        [trial["tau_estimate"] for trial in trial_results], x_grid
    )
    allocation_dual_weights = _mean_vector(
        [trial["allocation_dual_weights"] for trial in trial_results]
    )
    grid_shape = np.asarray(
        trial_results[0]["tau_estimate"]["tau2_grid"], dtype=float
    ).shape
    _debug_log(
        debug,
        f"Sampling complete: mean_ks={stats['mean_ks']:.6f}, extinction_rate={stats['extinction_rate']:.6f}",
    )

    result = {
        "mode": "estimate_and_sample",
        "B": int(B),
        "B1": int(B1),
        "sampling_steps": int(sampling_steps),
        "eta": float(eta),
        "reference_mode": reference_mode,
        "split_percentages": [float(x) for x in split_percentages],
        "split_step_points": trial_results[0]["resolved_step_points"],
        "split_points": trial_results[0]["split_points"],
        "x_grid": [float(x) for x in x_grid],
        "sigma_estimation_mode": sigma_estimation_mode,
        "bias_type": bias_type,
        "independent_n2": int(independent_n2),
        "reuse_phase1_samples": bool(reuse_phase1_samples),
        "sigma_times": trial_results[0]["sigma_times"],
        "sigma_estimates": sigma_estimates,
        "tau_estimate": tau_estimate,
        "optimal_M_k": _mean_vector([trial["optimal_M_k"] for trial in trial_results]),
        "allocation_dual_weights": allocation_dual_weights,
        "allocation_active_grid_points": _active_grid_points_from_dual_weights(
            allocation_dual_weights,
            x_grid,
            grid_shape,
        ),
        "allocation_solver_success": all(
            trial["allocation_solver_success"] for trial in trial_results
        ),
        "allocation_solver_success_rate": _mean_scalar(
            [trial["allocation_solver_success"] for trial in trial_results]
        ),
        "allocation_solver_status": int(
            round(
                _mean_scalar(
                    [trial["allocation_solver_status"] for trial in trial_results]
                )
            )
        ),
        "allocation_solver_message": _summarize_allocation_messages(
            [trial["allocation_solver_message"] for trial in trial_results]
        ),
        "allocation_dual_objective": _mean_scalar(
            [trial["allocation_dual_objective"] for trial in trial_results]
        ),
        "allocation_worst_case_simplex_objective": _mean_scalar(
            [
                trial["allocation_worst_case_simplex_objective"]
                for trial in trial_results
            ]
        ),
        "allocation_segment_costs": trial_results[0]["allocation_segment_costs"],
        "allocation_cost_weights": trial_results[0]["allocation_cost_weights"],
        "N_i": _mean_vector([trial["N_i"] for trial in trial_results]),
        "N_i_std": _std_vector([trial["N_i"] for trial in trial_results]),
        "N_i_count": int(len(trial_results)),
        "n0": _mean_scalar([trial["n0"] for trial in trial_results]),
        "final_root_count": _mean_scalar(
            [trial["final_root_count"] for trial in trial_results]
        ),
        "expected_cost_per_root": _mean_scalar(
            [trial["expected_cost_per_root"] for trial in trial_results]
        ),
        "expected_total_samples": _mean_scalar(
            [trial["expected_total_samples"] for trial in trial_results]
        ),
        "target_spec": target_spec,
        "pilot_roots": _mean_scalar([trial["pilot_roots"] for trial in trial_results]),
        "m_pilot": (
            _mean_scalar(
                [
                    trial["m_pilot"]
                    for trial in trial_results
                    if trial["m_pilot"] is not None
                ]
            )
            if any(trial["m_pilot"] is not None for trial in trial_results)
            else None
        ),
        "used_B1": _mean_scalar([trial["used_B1"] for trial in trial_results]),
        "B2": _mean_scalar([trial["B2"] for trial in trial_results]),
        **stats,
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result


def run_fixed_N_sampling(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    B: int,
    T: int,
    sampling_steps: int,
    eta: float,
    split_percentages: Sequence[float],
    N_i_list: Sequence[float],
    n_runs: int,
    seed: int | None,
    device: str,
    reference_mode: str = "ddpm_samples",
    debug: bool = False,
    n_parallel: int = 1,
    run_start_index: int = 0,
    return_trial_results: bool = False,
):
    _debug_log(
        debug,
        "Starting fixed_N run "
        f"with B={B}, sampling_steps={sampling_steps}, eta={eta}, N_i={list(N_i_list)}",
    )
    ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
    resolved_step_points, split_points = _resolve_split_percentages(
        ddim, split_percentages
    )

    split_factors = [float(x) for x in N_i_list]
    expected_cost_per_root = _expected_cost_per_root(ddim, split_points, split_factors)
    n0 = int(B // expected_cost_per_root)
    if n0 < 1:
        raise ValueError(
            f"Budget B={B} is too small; expected cost per root is {expected_cost_per_root:.6f}"
        )

    expected_total_samples = float(n0)
    for split_factor in split_factors:
        expected_total_samples *= split_factor

    _debug_log(
        debug,
        "Fixed_N sampling setup: "
        f"split_points={split_points}, expected_cost_per_root={expected_cost_per_root:.6f}, n0={n0}",
    )
    stats = _run_sampling_trials(
        model=model,
        target_spec=target_spec,
        ddim=ddim,
        n_runs=n_runs,
        n0=n0,
        split_points=split_points,
        split_factors=split_factors,
        data_mean=data_mean,
        data_std=data_std,
        reference_cdf_state=reference_cdf_state,
        reference_mode=reference_mode,
        seed=seed,
        debug=debug,
        n_parallel=n_parallel,
        run_start_index=run_start_index,
        return_trial_results=return_trial_results,
    )
    _debug_log(
        debug,
        f"Fixed_N sampling complete: mean_ks={stats['mean_ks']:.6f}, extinction_rate={stats['extinction_rate']:.6f}",
    )

    return {
        "mode": "fixed_N",
        "B": int(B),
        "B1": 0,
        "used_B1": 0,
        "B2": int(B),
        "sampling_steps": int(sampling_steps),
        "eta": float(eta),
        "reference_mode": reference_mode,
        "split_percentages": [float(x) for x in split_percentages],
        "split_step_points": [int(x) for x in resolved_step_points],
        "split_points": [int(x) for x in split_points],
        "input_N_i": [float(x) for x in split_factors],
        "N_i": [float(x) for x in split_factors],
        "n0": int(n0),
        "final_root_count": int(n0),
        "expected_cost_per_root": float(expected_cost_per_root),
        "expected_total_samples": float(expected_total_samples),
        "target_spec": target_spec,
        **stats,
    }
