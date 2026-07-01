import math
from typing import Any, Dict

import numpy as np
import torch
from numba import njit

KS_QUADRATURE_POINTS = 96
KS_QUAD_NODES, KS_QUAD_WEIGHTS = np.polynomial.legendre.leggauss(KS_QUADRATURE_POINTS)
KS_QUAD_NODES = np.ascontiguousarray(KS_QUAD_NODES, dtype=np.float32)
KS_QUAD_WEIGHTS = np.ascontiguousarray(KS_QUAD_WEIGHTS, dtype=np.float32)

APPROX_LOWER_ORTHANT_KS_VERSION = 1
APPROX_LOWER_ORTHANT_KS_DEFAULTS: Dict[str, Any] = {
    "num_queries": 512,
    "tail_eps": 1e-3,
    "paired_fraction": 0.25,
    "reference_eval_samples": 50_000,
    "generated_eval_samples": 50_000,
    "query_chunk_size": 32,
    "dedup_decimals": 12,
}


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


def coerce_samples_tensor(
    samples, device=None, expected_dim: int | None = None
) -> torch.Tensor:
    if isinstance(samples, torch.Tensor):
        tensor = samples.detach()
        if device is not None:
            tensor = tensor.to(device=device)
    else:
        tensor = torch.as_tensor(samples, device=device)
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    tensor = tensor.contiguous()
    if tensor.ndim != 2:
        raise ValueError(f"Expected samples of shape [N, D], got {tuple(tensor.shape)}")
    if expected_dim is not None and tensor.shape[1] != int(expected_dim):
        raise ValueError(
            f"Expected samples of shape [N, {int(expected_dim)}], got {tuple(tensor.shape)}"
        )
    return tensor


def _require_sample_dim(samples_np: np.ndarray, dim: int) -> np.ndarray:
    samples_np = np.asarray(samples_np, dtype=float)
    if samples_np.ndim != 2 or samples_np.shape[1] != int(dim):
        raise ValueError(
            f"Expected samples of shape [N, {int(dim)}], got {samples_np.shape}"
        )
    return samples_np


def _normalize_approx_lower_orthant_ks_params(
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = dict(APPROX_LOWER_ORTHANT_KS_DEFAULTS)
    if params:
        merged.update(dict(params))

    for key in (
        "num_queries",
        "reference_eval_samples",
        "generated_eval_samples",
        "query_chunk_size",
    ):
        merged[key] = int(merged[key])
        if merged[key] < 1:
            raise ValueError(f"approximate KS parameter {key} must be >= 1")

    merged["tail_eps"] = float(merged["tail_eps"])
    if not math.isfinite(merged["tail_eps"]) or not (0.0 <= merged["tail_eps"] < 0.5):
        raise ValueError("approximate KS parameter tail_eps must be in [0, 0.5)")

    merged["paired_fraction"] = float(merged["paired_fraction"])
    if not math.isfinite(merged["paired_fraction"]):
        raise ValueError("approximate KS parameter paired_fraction must be finite")
    merged["paired_fraction"] = min(max(merged["paired_fraction"], 0.0), 1.0)

    merged["dedup_decimals"] = int(merged.get("dedup_decimals", 12))
    if merged["dedup_decimals"] < 0:
        raise ValueError("approximate KS parameter dedup_decimals must be >= 0")
    return merged


def reference_ks_metric_cache_key(dimension: int) -> Dict[str, Any] | None:
    if int(dimension) <= 2:
        return None
    return {
        "metric": "approx_lower_orthant_two_sample_ks",
        "version": int(APPROX_LOWER_ORTHANT_KS_VERSION),
        "parameters": _normalize_approx_lower_orthant_ks_params(),
    }


def _deterministic_sample_subset(samples_np: np.ndarray, max_count: int) -> np.ndarray:
    samples_np = np.asarray(samples_np, dtype=np.float32)
    count = int(samples_np.shape[0])
    if count == 0:
        raise ValueError("Cannot build an evaluation subset from zero samples")
    limit = min(int(max_count), count)
    if limit == count:
        return np.ascontiguousarray(samples_np, dtype=np.float32)
    indices = np.floor((np.arange(limit, dtype=np.float32) + 0.5) * count / limit)
    indices = np.clip(indices.astype(np.int64), 0, count - 1)
    return np.ascontiguousarray(samples_np[indices], dtype=np.float32)


def prepare_reference_cdf_state(samples) -> Dict[str, Any]:
    samples_np = coerce_samples_np(samples)
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot prepare an empirical CDF state from zero samples")

    if samples_np.shape[1] == 1:
        return {
            "dimension": 1,
            "count": int(samples_np.shape[0]),
            "x_sorted": np.ascontiguousarray(
                np.sort(samples_np[:, 0]), dtype=np.float32
            ),
        }
    if samples_np.shape[1] != 2:
        params = _normalize_approx_lower_orthant_ks_params()
        reference_eval = _deterministic_sample_subset(
            samples_np, int(params["reference_eval_samples"])
        )
        return {
            "dimension": int(samples_np.shape[1]),
            "count": int(samples_np.shape[0]),
            "metric": "approx_lower_orthant_two_sample_ks",
            "metric_version": int(APPROX_LOWER_ORTHANT_KS_VERSION),
            "metric_params": params,
            "reference_eval_samples": reference_eval,
        }

    y_values, y_ranks = np.unique(samples_np[:, 1], return_inverse=True)
    x_order = np.argsort(samples_np[:, 0], kind="mergesort")
    return {
        "dimension": 2,
        "count": int(samples_np.shape[0]),
        "x_sorted": np.ascontiguousarray(samples_np[x_order, 0], dtype=np.float32),
        "y_ranks_sorted": np.ascontiguousarray(
            y_ranks[x_order].astype(np.int64) + 1, dtype=np.int64
        ),
        "y_values": np.ascontiguousarray(y_values, dtype=np.float32),
    }


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
    counts = np.empty(points_np.shape[0], dtype=np.float32)
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
    sums = np.zeros(tree_len, dtype=np.float32)
    max_prefix = np.zeros(tree_len, dtype=np.float32)
    min_prefix = np.zeros(tree_len, dtype=np.float32)

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
        np.asarray(samples, dtype=np.float32).reshape(-1, 2)
    )
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot compute KS distance from zero samples")
    order = np.argsort(samples_np[:, 0], kind="mergesort")
    x_sorted = np.ascontiguousarray(samples_np[order, 0], dtype=np.float32)
    y_ranks = np.searchsorted(union_y, samples_np[order, 1]).astype(np.int64)
    return x_sorted, np.ascontiguousarray(y_ranks, dtype=np.int64)


def _exact_two_sample_ks_1d_from_state(
    samples,
    reference_cdf_state: Dict[str, Any],
) -> float:
    samples_np = _require_sample_dim(coerce_samples_np(samples), 1)
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot compute KS distance from zero samples")

    x_a = np.ascontiguousarray(np.sort(samples_np[:, 0]), dtype=np.float32)
    x_b = np.ascontiguousarray(reference_cdf_state["x_sorted"], dtype=np.float32)
    if x_b.shape[0] == 0:
        raise ValueError("Cannot compute KS distance against zero reference samples")

    n_a = int(x_a.shape[0])
    n_b = int(x_b.shape[0])
    i = 0
    j = 0
    best = 0.0
    while i < n_a or j < n_b:
        if j >= n_b or (i < n_a and x_a[i] <= x_b[j]):
            value = x_a[i]
        else:
            value = x_b[j]
        while i < n_a and x_a[i] == value:
            i += 1
        while j < n_b and x_b[j] == value:
            j += 1
        diff = abs(i / float(n_a) - j / float(n_b))
        if diff > best:
            best = diff
    return float(best)


def _exact_two_sample_lower_orthant_ks_from_state(
    samples, reference_cdf_state: Dict[str, Any]
) -> float:
    samples_np = _require_sample_dim(coerce_samples_np(samples), 2)
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot compute KS distance from zero samples")

    ref_count = int(reference_cdf_state["count"])
    if ref_count == 0:
        raise ValueError("Cannot compute KS distance against zero reference samples")

    ref_y_values = np.asarray(reference_cdf_state["y_values"], dtype=np.float32)
    union_y = np.unique(np.concatenate([samples_np[:, 1], ref_y_values]))
    x_a, y_rank_a = _sorted_empirical_ks_inputs(samples_np, union_y)

    ref_rank_map = np.searchsorted(union_y, ref_y_values).astype(np.int64)
    y_rank_b = ref_rank_map[
        np.asarray(reference_cdf_state["y_ranks_sorted"], dtype=np.int64) - 1
    ]
    x_b = np.ascontiguousarray(reference_cdf_state["x_sorted"], dtype=np.float32)
    y_rank_b = np.ascontiguousarray(y_rank_b, dtype=np.int64)

    return float(
        _exact_two_sample_lower_orthant_ks_numba(
            x_a, y_rank_a, x_b, y_rank_b, int(union_y.shape[0])
        )
    )


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    limit = int(math.sqrt(value)) + 1
    for factor in range(3, limit, 2):
        if value % factor == 0:
            return False
    return True


def _first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < int(count):
        if _is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _van_der_corput(count: int, base: int) -> np.ndarray:
    values = np.empty(int(count), dtype=np.float32)
    for idx in range(int(count)):
        n = idx + 1
        denom = 1.0
        value = 0.0
        while n > 0:
            n, remainder = divmod(n, int(base))
            denom *= float(base)
            value += float(remainder) / denom
        values[idx] = value
    return values


def _deduplicate_query_points(query_points: np.ndarray, decimals: int) -> np.ndarray:
    query_points = np.asarray(query_points, dtype=np.float32)
    if query_points.ndim == 1:
        query_points = query_points.reshape(-1, 1)
    finite = np.isfinite(query_points).all(axis=1)
    query_points = query_points[finite]
    if query_points.size == 0:
        return np.empty((0, query_points.shape[1]), dtype=np.float32)
    rounded = np.round(query_points, int(decimals))
    _, keep = np.unique(rounded, axis=0, return_index=True)
    return np.ascontiguousarray(query_points[np.sort(keep)], dtype=np.float32)


def _deterministic_paired_indices(count: int, paired_count: int) -> np.ndarray:
    if int(paired_count) <= 0:
        return np.empty(0, dtype=np.int64)
    if int(paired_count) == 1:
        return np.asarray([int(count) // 2], dtype=np.int64)
    positions = np.floor(
        (np.arange(int(paired_count), dtype=np.float32) + 0.5)
        * int(count)
        / int(paired_count)
    )
    return np.clip(positions.astype(np.int64), 0, int(count) - 1)


def _approx_lower_orthant_query_points(
    generated_eval: np.ndarray,
    reference_eval: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    generated_eval = np.asarray(generated_eval, dtype=np.float32)
    reference_eval = np.asarray(reference_eval, dtype=np.float32)
    dim = int(reference_eval.shape[1])
    if generated_eval.ndim != 2 or int(generated_eval.shape[1]) != dim:
        raise ValueError(
            f"Expected generated samples of shape [N, {dim}], got {generated_eval.shape}"
        )

    pooled = np.concatenate([reference_eval, generated_eval], axis=0)
    pooled = pooled[np.isfinite(pooled).all(axis=1)]
    if pooled.shape[0] == 0:
        raise ValueError(
            "Cannot generate approximate KS queries from non-finite samples"
        )

    total = int(params["num_queries"])
    paired_target = int(round(total * float(params["paired_fraction"])))
    paired_count = min(paired_target, max(total - 1, 0), int(pooled.shape[0]))
    core_count = max(total - paired_count, 1)

    ranks = np.empty((core_count, dim), dtype=np.float32)
    bases = _first_primes(dim)
    for col in range(dim):
        ranks[:, col] = _van_der_corput(core_count, bases[col])
    eps = float(params["tail_eps"])
    ranks = eps + ranks * max(1.0 - 2.0 * eps, 1e-12)

    core = np.empty((core_count, dim), dtype=np.float32)
    for col in range(dim):
        core[:, col] = np.quantile(pooled[:, col], ranks[:, col])

    pieces = [core]
    if paired_count > 0:
        order = np.lexsort(tuple(pooled[:, col] for col in range(dim - 1, -1, -1)))
        paired_positions = _deterministic_paired_indices(len(order), paired_count)
        pieces.append(pooled[order[paired_positions]])

    queries = _deduplicate_query_points(
        np.concatenate(pieces, axis=0), int(params["dedup_decimals"])
    )
    if queries.shape[0] == 0:
        queries = np.mean(pooled, axis=0, keepdims=True)
    return np.ascontiguousarray(queries, dtype=np.float32)


def _lower_orthant_cdf_at_queries(
    samples_np: np.ndarray,
    queries: np.ndarray,
    *,
    query_chunk_size: int,
) -> np.ndarray:
    samples_np = np.asarray(samples_np, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot compute empirical CDF from zero samples")
    if queries.ndim == 1:
        queries = queries.reshape(-1, 1)
    if samples_np.ndim != 2 or queries.ndim != 2:
        raise ValueError("samples and queries must both be 2D")
    if samples_np.shape[1] != queries.shape[1]:
        raise ValueError(
            f"Sample dimension {samples_np.shape[1]} does not match query "
            f"dimension {queries.shape[1]}"
        )

    cdf = np.empty(int(queries.shape[0]), dtype=np.float32)
    chunk_size = max(int(query_chunk_size), 1)
    for start in range(0, int(queries.shape[0]), chunk_size):
        end = min(start + chunk_size, int(queries.shape[0]))
        chunk = queries[start:end]
        below = samples_np[:, None, :] <= chunk[None, :, :]
        cdf[start:end] = np.count_nonzero(np.all(below, axis=2), axis=0) / float(
            samples_np.shape[0]
        )
    return cdf


def _approx_two_sample_lower_orthant_ks_from_state(
    samples,
    reference_cdf_state: Dict[str, Any],
) -> float:
    dim = int(reference_cdf_state["dimension"])
    samples_np = _require_sample_dim(coerce_samples_np(samples), dim)
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot compute KS distance from zero samples")

    reference_eval = _require_sample_dim(
        np.asarray(reference_cdf_state["reference_eval_samples"], dtype=np.float32),
        dim,
    )
    if reference_eval.shape[0] == 0:
        raise ValueError("Cannot compute KS distance against zero reference samples")

    params = _normalize_approx_lower_orthant_ks_params(
        reference_cdf_state.get("metric_params")
    )
    generated_eval = _deterministic_sample_subset(
        samples_np, int(params["generated_eval_samples"])
    )
    queries = _approx_lower_orthant_query_points(generated_eval, reference_eval, params)
    reference_cdf = _lower_orthant_cdf_at_queries(
        reference_eval,
        queries,
        query_chunk_size=int(params["query_chunk_size"]),
    )
    generated_cdf = _lower_orthant_cdf_at_queries(
        generated_eval,
        queries,
        query_chunk_size=int(params["query_chunk_size"]),
    )
    if queries.shape[0] == 0:
        return 0.0
    distance = float(np.max(np.abs(generated_cdf - reference_cdf)))
    return float(min(max(distance, 0.0), 1.0))


def _normal_cdf_np(values: np.ndarray, *, mean: float, std: float) -> np.ndarray:
    if float(std) <= 0.0:
        raise ValueError("normal CDF std must be positive")
    z = (np.asarray(values, dtype=np.float32) - float(mean)) / (
        float(std) * math.sqrt(2.0)
    )
    erf = np.vectorize(math.erf, otypes=[np.float32])
    return 0.5 * (1.0 + erf(z))


def _target_cdf_1d(values: np.ndarray, target_spec: Dict[str, Any]) -> np.ndarray:
    cdf_spec = target_spec.get("cdf")
    if not isinstance(cdf_spec, dict):
        raise ValueError("1D true_dist target_spec must include a CDF spec")
    kind = cdf_spec.get("kind")
    values_np = np.asarray(values, dtype=np.float32)
    if kind == "normal":
        return _normal_cdf_np(
            values_np,
            mean=float(cdf_spec["mean"]),
            std=float(cdf_spec["std"]),
        )
    raise ValueError(f"Unknown 1D CDF kind {kind!r}")


def _exact_empirical_target_ks_1d(samples, target_spec: Dict[str, Any]) -> float:
    samples_np = _require_sample_dim(coerce_samples_np(samples), 1)
    n_samples = int(samples_np.shape[0])
    if n_samples == 0:
        raise ValueError("Cannot compute KS distance from zero samples")
    y = np.sort(np.ascontiguousarray(samples_np[:, 0], dtype=np.float32))
    cdf = np.clip(_target_cdf_1d(y, target_spec), 0.0, 1.0)
    j = np.arange(1, n_samples + 1, dtype=np.float32)
    d_plus = np.max(j / float(n_samples) - cdf)
    d_minus = np.max(cdf - (j - 1.0) / float(n_samples))
    return float(max(d_plus, d_minus))


def _target_spec_arrays(target_spec: Dict[str, Any]):
    weights = np.ascontiguousarray(target_spec["weights"], dtype=np.float32)
    means = np.ascontiguousarray(target_spec["means"], dtype=np.float32)
    covariances = np.ascontiguousarray(target_spec["covariances"], dtype=np.float32)
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
    samples, target_spec: Dict[str, Any]
) -> float:
    samples_np = _require_sample_dim(coerce_samples_np(samples), 2)
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
            np.ascontiguousarray(x_values, dtype=np.float32),
            np.ascontiguousarray(x_counts, dtype=np.int64),
            np.ascontiguousarray(y_values, dtype=np.float32),
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
        np.ascontiguousarray([0.0], dtype=np.float32),
        np.ascontiguousarray([1], dtype=np.int64),
        np.ascontiguousarray([0.0], dtype=np.float32),
        np.ascontiguousarray([0], dtype=np.int64),
        weights,
        means,
        covariances,
        KS_QUAD_NODES,
        KS_QUAD_WEIGHTS,
        1,
    )


def warm_ks_kernel_for_mode(reference_mode: str, target_spec: Dict[str, Any]):
    if reference_mode == "true_dist":
        if int(target_spec.get("dimension", 2)) == 1:
            _target_cdf_1d(np.asarray([0.0], dtype=np.float32), target_spec)
            return
        _warm_target_ks_kernel(target_spec)
    elif reference_mode in {"true_samples", "ddpm_samples", "edm_samples"}:
        _exact_two_sample_lower_orthant_ks_numba(
            np.ascontiguousarray([0.0], dtype=np.float32),
            np.ascontiguousarray([0], dtype=np.int64),
            np.ascontiguousarray([0.0], dtype=np.float32),
            np.ascontiguousarray([0], dtype=np.int64),
            1,
        )


def warm_reference_ks_kernel(reference_cdf_state: Dict[str, Any]):
    dimension = int(reference_cdf_state.get("dimension", 2))
    if dimension == 1 or dimension > 2:
        return
    sample_x = reference_cdf_state["x_sorted"][0]
    sample_y = reference_cdf_state["y_values"][0]
    warm_points = np.ascontiguousarray([[sample_x, sample_y]], dtype=np.float32)
    _empirical_lower_orthant_cdf_numba(
        reference_cdf_state["x_sorted"],
        reference_cdf_state["y_ranks_sorted"],
        reference_cdf_state["y_values"],
        warm_points,
        int(reference_cdf_state["count"]),
    )
    _exact_two_sample_lower_orthant_ks_numba(
        np.ascontiguousarray([sample_x], dtype=np.float32),
        np.ascontiguousarray([0], dtype=np.int64),
        np.ascontiguousarray([sample_x], dtype=np.float32),
        np.ascontiguousarray([0], dtype=np.int64),
        1,
    )


def compute_reference_ks_distance(samples, reference_cdf_state: Dict[str, Any]):
    """Compute KS distance against empirical reference samples."""
    dimension = int(reference_cdf_state.get("dimension", 2))
    if dimension == 1:
        ks_distance = _exact_two_sample_ks_1d_from_state(samples, reference_cdf_state)
    elif dimension == 2:
        ks_distance = _exact_two_sample_lower_orthant_ks_from_state(
            samples, reference_cdf_state
        )
    else:
        ks_distance = _approx_two_sample_lower_orthant_ks_from_state(
            samples, reference_cdf_state
        )
    empty = torch.empty(0)
    return ks_distance, empty, empty


def compute_target_ks_distance(samples, target_spec: Dict[str, Any]):
    """Compute exact KS distance against the target CDF."""
    samples_np = coerce_samples_np(samples)
    target_dim = int(target_spec.get("dimension", samples_np.shape[1]))
    if target_dim == 1:
        ks_distance = _exact_empirical_target_ks_1d(samples_np, target_spec)
    elif target_dim == 2:
        ks_distance = _exact_empirical_target_lower_orthant_ks(samples_np, target_spec)
    else:
        raise ValueError(f"Unsupported target dimension {target_dim}")
    empty = torch.empty(0)
    return ks_distance, empty, empty


def compute_ks_distance(
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
