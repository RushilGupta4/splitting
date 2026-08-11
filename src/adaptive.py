import logging
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from numba import njit

from runners.splitting import (
    apply_to_run_batches,
    append_by_counts as _append_by_counts,
    floor_split_total_count,
    normalize_max_sampling_batch_size,
    split_counts_by_run_batches,
)
from metrics.utils import coerce_samples_np
from trials import (
    PHASE2_SEED_OFFSET,
    iter_run_chunks,
    make_torch_generator,
    submit_sampling_trial_result_futures,
    summarize_sampling_trials,
)

log = logging.getLogger(__name__)

CROSSFIT_Q_DEFAULT_MLP_HIDDEN_DIMS = [128, 64]
CROSSFIT_Q_DEFAULT_MLP_ACTIVATION = "silu"
CROSSFIT_Q_DEFAULT_MLP_EPOCHS = 5
CROSSFIT_Q_DEFAULT_MLP_BATCH_SIZE = 24000
CROSSFIT_Q_DEFAULT_MLP_LR = 5e-3
CROSSFIT_Q_DEFAULT_MLP_WEIGHT_DECAY = 3e-4
CROSSFIT_Q_DEFAULT_MLP_LOSS = "mse"
CROSSFIT_Q_DEFAULT_MLP_DEVICE = "runner"
CROSSFIT_Q_DEFAULT_MLP_NUM_THREADS = 2
CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM = 10
CROSSFIT_Q_DEFAULT_NUM_QUERIES = 1024
CROSSFIT_Q_DEFAULT_K_MAX = 64
CROSSFIT_Q_DEFAULT_MASS_MIN = 0.05
CROSSFIT_Q_DEFAULT_MASS_MAX = 0.95
CROSSFIT_Q_DEFAULT_MASS_BINS = 16

_DEFAULT_CROSSFIT_Q_MLP_PARAMS: Dict[str, Any] = {
    "hidden_dims": list(CROSSFIT_Q_DEFAULT_MLP_HIDDEN_DIMS),
    "activation": CROSSFIT_Q_DEFAULT_MLP_ACTIVATION,
    "epochs": CROSSFIT_Q_DEFAULT_MLP_EPOCHS,
    "batch_size": CROSSFIT_Q_DEFAULT_MLP_BATCH_SIZE,
    "lr": CROSSFIT_Q_DEFAULT_MLP_LR,
    "weight_decay": CROSSFIT_Q_DEFAULT_MLP_WEIGHT_DECAY,
    "loss": CROSSFIT_Q_DEFAULT_MLP_LOSS,
    "device": CROSSFIT_Q_DEFAULT_MLP_DEVICE,
    "num_threads": CROSSFIT_Q_DEFAULT_MLP_NUM_THREADS,
}

_DEFAULT_QUERY_PARAMS: Dict[str, Any] = {
    "num_queries": CROSSFIT_Q_DEFAULT_NUM_QUERIES,
    "k_max": CROSSFIT_Q_DEFAULT_K_MAX,
    "mass_min": CROSSFIT_Q_DEFAULT_MASS_MIN,
    "mass_max": CROSSFIT_Q_DEFAULT_MASS_MAX,
    "mass_bins": CROSSFIT_Q_DEFAULT_MASS_BINS,
}

_CROSSFIT_Q_MLP_INIT_LOCK = Lock()
_CROSSFIT_Q_FEATURE_CACHE_CHUNK_SIZE = 262_144
_MONOTONE_CVAR95_ALPHA = 0.95
SUPPORTED_OPTIMIZATION_MODES = {"monotone", "monotone_cvar95"}


def _debug(enabled: bool, message: str):
    if enabled:
        log.debug(message)


def _sample_prior_by_run_batches(
    runner,
    counts_by_run,
    *,
    max_sampling_batch_size=None,
    generator=None,
):
    max_sampling_batch_size = normalize_max_sampling_batch_size(max_sampling_batch_size)
    counts_by_run = [int(count) for count in counts_by_run]
    num_runs = len(counts_by_run)
    if max_sampling_batch_size is None:
        counts_tensor = torch.as_tensor(
            counts_by_run, device=runner.device, dtype=torch.long
        )
        run_ids = torch.repeat_interleave(
            torch.arange(num_runs, device=runner.device, dtype=torch.long),
            counts_tensor,
        )
        return (
            runner.sample_prior(int(sum(counts_by_run)), generator=generator),
            run_ids,
        )

    parts_by_run = [[] for _ in range(num_runs)]
    for counts in split_counts_by_run_batches(counts_by_run, max_sampling_batch_size):
        total = int(sum(counts))
        if total <= 0:
            continue
        x = runner.sample_prior(total, generator=generator)
        _append_by_counts(parts_by_run, x, counts)
    x_by_run = [torch.cat(parts, dim=0) for parts in parts_by_run]
    counts_tensor = torch.as_tensor(
        counts_by_run, device=runner.device, dtype=torch.long
    )
    run_ids = torch.repeat_interleave(
        torch.arange(num_runs, device=runner.device, dtype=torch.long), counts_tensor
    )
    return torch.cat(x_by_run, dim=0), run_ids


def _sample_segment_by_run_batches(
    runner,
    x: torch.Tensor,
    run_ids: torch.Tensor,
    *,
    num_runs: int,
    start_time,
    end_time,
    max_sampling_batch_size=None,
    generator=None,
):
    return apply_to_run_batches(
        x,
        run_ids,
        num_runs=int(num_runs),
        max_count_per_run=max_sampling_batch_size,
        fn=lambda batch: runner.sample_segment(
            batch,
            start_time,
            end_time,
            generator=generator,
        ),
    )


# --- Cost-optimal monotone split-factor optimizers ---------------------------


def _objective_values(M, y):
    return M @ (1.0 / y)


def _weighted_pava_non_decreasing(values: np.ndarray, weights: np.ndarray):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("PAVA values and weights must be matching 1D arrays")
    if values.size == 0:
        raise ValueError("PAVA values must be non-empty")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("PAVA inputs must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("PAVA weights must be positive")

    blocks = []
    for idx, (value, weight) in enumerate(zip(values, weights)):
        weighted_sum = float(weight * value)
        blocks.append([idx, idx + 1, float(weight), weighted_sum, float(value)])
        while len(blocks) >= 2 and blocks[-2][4] > blocks[-1][4]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = left[2] + right[2]
            merged_sum = left[3] + right[3]
            blocks.append(
                [
                    left[0],
                    right[1],
                    merged_weight,
                    merged_sum,
                    merged_sum / merged_weight,
                ]
            )

    projected = np.empty_like(values, dtype=float)
    for start, end, _, _, value in blocks:
        projected[start:end] = value
    return projected


if njit is not None:

    @njit
    def _crossfit_q_project_sequences_numba(Q_raw, F_hat):
        num_times, obs_dim = Q_raw.shape
        Q_proj = np.empty_like(Q_raw)
        starts = np.empty(num_times, dtype=np.int64)
        ends = np.empty(num_times, dtype=np.int64)
        weights = np.empty(num_times, dtype=np.float32)
        sums = np.empty(num_times, dtype=np.float32)
        levels = np.empty(num_times, dtype=np.float32)
        clipped = np.empty(num_times, dtype=np.float32)

        for obs_idx in range(obs_dim):
            F = float(F_hat[obs_idx])
            if F < 0.0:
                F = 0.0
            elif F > 1.0:
                F = 1.0
            lower = F * F
            upper = F
            for time_idx in range(num_times):
                y = float(Q_raw[time_idx, obs_idx])
                if y < lower:
                    y = lower
                elif y > upper:
                    y = upper
                clipped[time_idx] = y
            clipped[num_times - 1] = F

            if lower == upper:
                for time_idx in range(num_times):
                    Q_proj[time_idx, obs_idx] = F
            else:
                block_count = 0
                for time_idx in range(num_times):
                    weight = 1e12 if time_idx == num_times - 1 else 1.0
                    value = clipped[time_idx]
                    starts[block_count] = time_idx
                    ends[block_count] = time_idx + 1
                    weights[block_count] = weight
                    sums[block_count] = weight * value
                    levels[block_count] = value
                    block_count += 1
                    while (
                        block_count >= 2
                        and levels[block_count - 2] > levels[block_count - 1]
                    ):
                        left = block_count - 2
                        right = block_count - 1
                        merged_weight = weights[left] + weights[right]
                        merged_sum = sums[left] + sums[right]
                        ends[left] = ends[right]
                        weights[left] = merged_weight
                        sums[left] = merged_sum
                        levels[left] = merged_sum / merged_weight
                        block_count -= 1
                for block_idx in range(block_count):
                    value = levels[block_idx]
                    if value < lower:
                        value = lower
                    elif value > upper:
                        value = upper
                    for time_idx in range(starts[block_idx], ends[block_idx]):
                        Q_proj[time_idx, obs_idx] = value
                Q_proj[num_times - 1, obs_idx] = F

        return Q_proj

else:
    _crossfit_q_project_sequences_numba = None


def _monotone_simplex_from_profile(
    profile: np.ndarray,
    cost_w: np.ndarray,
    *,
    allocation_floor=1e-12,
):
    profile = np.maximum(np.asarray(profile, dtype=float), 0.0)
    cost_w = np.asarray(cost_w, dtype=float)
    h = _weighted_pava_non_decreasing(profile / cost_w, cost_w)
    raw = np.sqrt(np.maximum(h, 0.0))
    if not np.isfinite(raw).all() or float(np.max(raw)) <= 0.0:
        raw = np.ones_like(cost_w, dtype=float)
    raw = np.maximum(raw, float(allocation_floor))
    allocation = raw / float(cost_w @ raw)
    return cost_w * allocation


def _monotone_dual_value_and_simplex(profile, cost_w):
    profile = np.maximum(np.asarray(profile, dtype=float), 0.0)
    y = _monotone_simplex_from_profile(profile, cost_w)
    value = float((profile * cost_w) @ (1.0 / y))
    return value, y


def _monotone_line_search(profile, target_profile, cost_w, *, max_iters=48):
    profile = np.asarray(profile, dtype=float)
    target_profile = np.asarray(target_profile, dtype=float)
    if np.allclose(profile, target_profile, rtol=0.0, atol=0.0):
        value, y = _monotone_dual_value_and_simplex(profile, cost_w)
        return 0.0, value, y

    def value_at(gamma):
        candidate = (1.0 - gamma) * profile + gamma * target_profile
        return _monotone_dual_value_and_simplex(candidate, cost_w)

    left, right = 0.0, 1.0
    inv_phi = (math.sqrt(5.0) - 1.0) * 0.5
    mid1 = right - inv_phi * (right - left)
    mid2 = left + inv_phi * (right - left)
    val1, _ = value_at(mid1)
    val2, _ = value_at(mid2)
    for _ in range(max_iters):
        if val1 < val2:
            left = mid1
            mid1 = mid2
            val1 = val2
            mid2 = left + inv_phi * (right - left)
            val2, _ = value_at(mid2)
        else:
            right = mid2
            mid2 = mid1
            val2 = val1
            mid1 = right - inv_phi * (right - left)
            val1, _ = value_at(mid1)

    candidates = [0.0, 1.0, 0.5 * (left + right)]
    best_gamma = 0.0
    best_value, best_y = value_at(0.0)
    for gamma in candidates[1:]:
        value, y = value_at(gamma)
        if value > best_value:
            best_gamma = float(gamma)
            best_value = value
            best_y = y
    return best_gamma, best_value, best_y


def _solve_frank_wolfe_monotone_allocation(
    M: np.ndarray,
    *,
    cost_w: np.ndarray,
    relative_tol: float = 1e-5,
    max_iters: int = 1000,
):
    M = np.asarray(M, dtype=float)
    num_points, num_levels = M.shape
    if num_points == 0 or num_levels == 0:
        raise ValueError("M must be non-empty")

    weighted_M = M * cost_w[None, :]
    row_scores = np.sqrt(np.maximum(weighted_M, 0.0)).sum(axis=1)
    profile = np.maximum(M[int(np.argmax(row_scores))].copy(), 0.0)

    best_y = np.asarray(cost_w, dtype=float).copy()
    best_upper = math.inf
    for iter_idx in range(max_iters):
        dual_lower, y = _monotone_dual_value_and_simplex(profile, cost_w)
        full_values = _objective_values(weighted_M, y)
        full_upper = float(np.max(full_values))
        if full_upper < best_upper:
            best_upper = full_upper
            best_y = y.copy()
        gap = max(full_upper - dual_lower, 0.0) / max(abs(full_upper), 1.0)
        if gap <= relative_tol:
            return y

        worst_idx = int(np.argmax(full_values))
        target_profile = np.maximum(M[worst_idx], 0.0)
        gamma, candidate_lower, candidate_y = _monotone_line_search(
            profile, target_profile, cost_w
        )
        if gamma <= 0.0 or candidate_lower <= dual_lower + 1e-14:
            step = 2.0 / float(iter_idx + 3.0)
            profile = (1.0 - step) * profile + step * target_profile
        else:
            profile = (1.0 - gamma) * profile + gamma * target_profile
            if candidate_lower > dual_lower:
                candidate_values = _objective_values(weighted_M, candidate_y)
                candidate_upper = float(np.max(candidate_values))
                if candidate_upper < best_upper:
                    best_upper = candidate_upper
                    best_y = candidate_y.copy()

    return best_y


def _weighted_cvar_profile(
    M: np.ndarray,
    losses: np.ndarray,
    row_weights: np.ndarray,
    *,
    alpha: float,
):
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("CVaR alpha must be in [0, 1)")
    losses = np.asarray(losses, dtype=float).reshape(-1)
    row_weights = _normalize_query_weights(row_weights, losses.shape[0])
    if M.shape[0] != losses.shape[0]:
        raise ValueError("CVaR losses must match allocation matrix rows")
    if not np.isfinite(losses).all():
        raise ValueError("CVaR losses must be finite")

    tail_mass = 1.0 - alpha
    caps = row_weights / tail_mass
    adversary = np.zeros_like(row_weights, dtype=float)
    remaining = 1.0
    for idx in np.argsort(losses)[::-1]:
        take = min(float(caps[idx]), remaining)
        if take > 0.0:
            adversary[idx] = take
            remaining -= take
        if remaining <= 1e-12:
            break
    total = float(adversary.sum())
    if total <= 0.0:
        raise ValueError("CVaR adversary weights have zero total")
    adversary /= total
    return adversary @ M, float(adversary @ losses)


def _query_weights_for_allocation(query_metadata, count: int):
    return _normalize_query_weights(
        None if query_metadata is None else query_metadata.get("weights"),
        int(count),
    )


def _solve_cvar_monotone_allocation(
    M: np.ndarray,
    *,
    cost_w: np.ndarray,
    row_weights: np.ndarray,
    alpha: float = _MONOTONE_CVAR95_ALPHA,
    relative_tol: float = 1e-5,
    max_iters: int = 1000,
):
    M = np.asarray(M, dtype=float)
    num_points, num_levels = M.shape
    if num_points == 0 or num_levels == 0:
        raise ValueError("M must be non-empty")
    row_weights = _normalize_query_weights(row_weights, num_points)
    weighted_M = M * cost_w[None, :]
    row_scores = np.sqrt(np.maximum(weighted_M, 0.0)).sum(axis=1)
    profile, _ = _weighted_cvar_profile(
        M,
        row_scores,
        row_weights,
        alpha=alpha,
    )

    best_y = np.asarray(cost_w, dtype=float).copy()
    best_upper = math.inf
    for iter_idx in range(max_iters):
        dual_lower, y = _monotone_dual_value_and_simplex(profile, cost_w)
        full_values = _objective_values(weighted_M, y)
        target_profile, full_upper = _weighted_cvar_profile(
            M,
            full_values,
            row_weights,
            alpha=alpha,
        )
        if full_upper < best_upper:
            best_upper = full_upper
            best_y = y.copy()
        gap = max(full_upper - dual_lower, 0.0) / max(abs(full_upper), 1.0)
        if gap <= relative_tol:
            return y

        gamma, candidate_lower, candidate_y = _monotone_line_search(
            profile, target_profile, cost_w
        )
        if gamma <= 0.0 or candidate_lower <= dual_lower + 1e-14:
            step = 2.0 / float(iter_idx + 3.0)
            profile = (1.0 - step) * profile + step * target_profile
        else:
            profile = (1.0 - gamma) * profile + gamma * target_profile
            if candidate_lower > dual_lower:
                candidate_values = _objective_values(weighted_M, candidate_y)
                _, candidate_upper = _weighted_cvar_profile(
                    M,
                    candidate_values,
                    row_weights,
                    alpha=alpha,
                )
                if candidate_upper < best_upper:
                    best_upper = candidate_upper
                    best_y = candidate_y.copy()

    return best_y


def _normalize_query_weights(weights, count: int):
    if weights is None:
        return np.full(int(count), 1.0 / float(count), dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape != (int(count),):
        raise ValueError(f"query weights shape {weights.shape} != ({int(count)},)")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("query weights must be finite and nonnegative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("query weights must have positive total")
    return weights / total


def _solve_optimal_split_factors(
    variance2_per_level: np.ndarray,
    tau2: np.ndarray,
    cost_weights: Sequence[float],
    optimization_mode: str = "monotone",
    *,
    query_metadata: Mapping[str, Any] | None = None,
) -> List[float]:
    """Cost-optimal monotone split factors from the per-level variance grids.

    `monotone` minimises the worst query's `sum_l c_l M[q,l] / y_l`;
    `monotone_cvar95` minimises the 95% CVaR over queries instead.
    """
    if optimization_mode not in SUPPORTED_OPTIMIZATION_MODES:
        raise ValueError(f"unknown optimization_mode '{optimization_mode}'")
    variance2_per_level = np.asarray(variance2_per_level, dtype=float)
    tau2 = np.asarray(tau2, dtype=float)
    if variance2_per_level.ndim != 2:
        raise ValueError("variance2_per_level must be (num_levels, grid_pts)")
    if not np.isfinite(variance2_per_level).all() or not np.isfinite(tau2).all():
        raise ValueError("variance/tau2 contain non-finite values")

    num_levels, grid_pts = variance2_per_level.shape
    if tau2.shape != (grid_pts,):
        raise ValueError(f"tau2 shape {tau2.shape} != ({grid_pts},)")

    cost_w = np.asarray(cost_weights, dtype=float)
    if cost_w.shape != (num_levels,):
        raise ValueError(f"cost_weights shape {cost_w.shape} != ({num_levels},)")
    if not np.isfinite(cost_w).all() or np.any(cost_w <= 0.0):
        raise ValueError("cost_weights must be finite and positive")
    cost_w = cost_w / float(cost_w.sum())

    M = np.maximum(variance2_per_level, 0.0).T.copy()
    M[:, 0] += np.maximum(tau2, 0.0)
    if optimization_mode == "monotone":
        simplex = _solve_frank_wolfe_monotone_allocation(M, cost_w=cost_w)
    else:
        simplex = _solve_cvar_monotone_allocation(
            M,
            cost_w=cost_w,
            row_weights=_query_weights_for_allocation(query_metadata, M.shape[0]),
            alpha=_MONOTONE_CVAR95_ALPHA,
        )
    simplex = np.asarray(simplex, dtype=float).reshape(-1)
    if simplex.shape != (num_levels,):
        raise ValueError(f"optimizer simplex shape {simplex.shape} != ({num_levels},)")
    if not np.isfinite(simplex).all() or float(simplex.sum()) <= 0.0:
        raise ValueError("optimizer produced invalid simplex")
    simplex /= float(simplex.sum())
    allocation = simplex / cost_w
    allocation /= float(cost_w @ allocation)
    if not np.isfinite(allocation).all() or np.any(allocation <= 0.0):
        raise ValueError("optimizer produced invalid allocation")
    if np.any(np.diff(allocation) < -1e-8 * np.maximum(1.0, np.abs(allocation[:-1]))):
        raise ValueError("optimizer produced non-monotone allocation")
    return (allocation[1:] / allocation[:-1]).tolist()


# --- Cross-fitted Q-projection regression -----------------------------------


def _crossfit_q_normalize_mlp_params(params: Mapping[str, Any] | None):
    merged = dict(_DEFAULT_CROSSFIT_Q_MLP_PARAMS)
    if params:
        unknown = sorted(set(params) - set(_DEFAULT_CROSSFIT_Q_MLP_PARAMS))
        if unknown:
            raise ValueError(f"Unknown crossfit_q_mlp_params: {unknown}")
        merged.update(dict(params))

    hidden_dims = merged.get("hidden_dims")
    if isinstance(hidden_dims, int):
        hidden_dims = [int(hidden_dims)]
    hidden_dims = [int(width) for width in (hidden_dims or [])]
    if not hidden_dims or any(width < 1 for width in hidden_dims):
        raise ValueError("crossfit_q_mlp_params.hidden_dims must be positive ints")
    merged["hidden_dims"] = hidden_dims

    merged["activation"] = str(merged.get("activation", "silu")).lower()
    if merged["activation"] not in {"relu", "silu", "tanh"}:
        raise ValueError("crossfit_q_mlp_params.activation must be relu, silu, or tanh")

    merged["epochs"] = int(merged.get("epochs", 100))
    if merged["epochs"] < 1:
        raise ValueError("crossfit_q_mlp_params.epochs must be >= 1")

    merged["batch_size"] = int(merged.get("batch_size", 256))
    if merged["batch_size"] < 1:
        raise ValueError("crossfit_q_mlp_params.batch_size must be >= 1")

    merged["lr"] = float(merged.get("lr", 1e-3))
    if not math.isfinite(merged["lr"]) or merged["lr"] <= 0.0:
        raise ValueError("crossfit_q_mlp_params.lr must be finite and positive")

    merged["weight_decay"] = float(merged.get("weight_decay", 1e-4))
    if not math.isfinite(merged["weight_decay"]) or merged["weight_decay"] < 0.0:
        raise ValueError(
            "crossfit_q_mlp_params.weight_decay must be finite and nonnegative"
        )


    merged["loss"] = str(merged.get("loss", "mse")).lower()
    if merged["loss"] not in {"bce", "mse"}:
        raise ValueError("crossfit_q_mlp_params.loss must be bce or mse")

    merged["device"] = str(merged.get("device", "runner"))

    num_threads = merged.get("num_threads", 2)
    if num_threads in {None, "", 0, "0", "none", "None"}:
        merged["num_threads"] = None
    else:
        merged["num_threads"] = int(num_threads)
        if merged["num_threads"] < 1:
            raise ValueError(
                "crossfit_q_mlp_params.num_threads must be positive or null"
            )

    return merged


def _crossfit_q_normalize_mlp_run_parallelism(value: Any = None) -> int:
    if value is None or value == "":
        return int(CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM)
    workers = int(value)
    if workers < 1:
        raise ValueError("crossfit_q_mlp_run_parallelism must be at least 1")
    return workers


class _TorchNumThreadsContext:
    def __init__(self, num_threads: int | None, enabled: bool):
        self._num_threads = num_threads
        self._enabled = bool(enabled and num_threads is not None)
        self._previous = None

    def __enter__(self):
        if self._enabled:
            self._previous = int(torch.get_num_threads())
            target = int(self._num_threads)
            if self._previous != target:
                torch.set_num_threads(target)
        return int(torch.get_num_threads())

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and self._previous is not None:
            if int(torch.get_num_threads()) != int(self._previous):
                torch.set_num_threads(int(self._previous))
        return False


def _crossfit_q_activation(name: str):
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"unknown MLP activation {name!r}")


def _crossfit_q_build_mlp(input_dim: int, output_dim: int, params: Mapping[str, Any]):
    layers: List[torch.nn.Module] = []
    prev = int(input_dim)
    for width in params["hidden_dims"]:
        layers.append(torch.nn.Linear(prev, int(width)))
        layers.append(_crossfit_q_activation(str(params["activation"])))
        prev = int(width)
    layers.append(torch.nn.Linear(prev, int(output_dim)))
    return torch.nn.Sequential(*layers)


def _crossfit_q_query_conditioning(query_spec: Mapping[str, Any]):
    """Per-query descriptor for the regressor: just the subset size and the mass.

    The state enters the network only through the margins to the query corner
    (see `triple_features`), so the descriptor carries no per-coordinate blocks:
    a mask, the raster coordinate index and the ranks are all either constant,
    arbitrary, or recoverable from these two scalars.
    """
    if not isinstance(query_spec, Mapping):
        raise ValueError("crossfit_q requires a query_spec")
    if query_spec.get("kind") != "sparse_subset_lower_orthant":
        raise ValueError(f"unknown query spec kind {query_spec.get('kind')!r}")

    ambient_dim = int(query_spec["ambient_dim"])
    if ambient_dim < 1:
        raise ValueError("query ambient_dim must be positive")
    active_dims = np.asarray(query_spec["active_dims"], dtype=np.int64)
    thresholds = np.asarray(query_spec["thresholds"], dtype=float)
    if active_dims.ndim != 2 or thresholds.shape != active_dims.shape:
        raise ValueError("query active_dims and thresholds must be matching 2D arrays")
    if np.any(active_dims >= ambient_dim):
        raise ValueError("query active dimension exceeds ambient dimension")
    active_sizes = np.asarray(query_spec["active_sizes"], dtype=float).reshape(-1)
    target_mass = np.asarray(query_spec["target_mass"], dtype=float).reshape(-1)
    num_queries = active_dims.shape[0]
    if active_sizes.shape != (num_queries,) or target_mass.shape != (num_queries,):
        raise ValueError("query metadata shapes do not match active_dims")

    active_mask = active_dims >= 0
    safe_dims = np.where(active_mask, active_dims, 0).astype(np.int64, copy=False)
    mass = np.clip(target_mass, 1e-6, 1.0 - 1e-6)
    features = np.column_stack(
        [active_sizes / float(ambient_dim), np.log(mass / (1.0 - mass))]
    )
    if not np.isfinite(features).all():
        raise ValueError("crossfit_q query features contain non-finite values")
    return {
        "features": features.astype(np.float32, copy=False),
        "active_dims": safe_dims,
        "active_mask": active_mask.astype(np.float32, copy=False),
        "thresholds": np.where(active_mask, thresholds, 0.0).astype(
            np.float32, copy=False
        ),
        "ambient_dim": ambient_dim,
        "max_active": int(active_dims.shape[1]),
    }


def _crossfit_q_mlp_device(params: Mapping[str, Any], runner_device: str | None):
    requested = str(params.get("device", "runner"))
    if requested in {"runner", ""}:
        device = torch.device(runner_device or "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested MLP device {device} but CUDA is unavailable")
    return device


def _crossfit_q_mlp_predict_all_levels(
    train_states_by_level,
    train_labels,
    test_states_by_level,
    query_conditioning,
    level_times,
    *,
    params: Mapping[str, Any] | None,
    runner_device: str | None,
    seed: int | None,
    manage_num_threads: bool = True,
):
    params = _crossfit_q_normalize_mlp_params(params)
    device = _crossfit_q_mlp_device(params, runner_device)
    with _TorchNumThreadsContext(
        params.get("num_threads"), manage_num_threads and device.type == "cpu"
    ):
        return _crossfit_q_mlp_predict_all_levels_impl(
            train_states_by_level,
            train_labels,
            test_states_by_level,
            query_conditioning,
            level_times,
            params=params,
            device=device,
            seed=seed,
        )


def _crossfit_q_mlp_predict_all_levels_impl(
    train_states_by_level,
    train_labels,
    test_states_by_level,
    query_conditioning,
    level_times,
    *,
    params: Mapping[str, Any],
    device: torch.device,
    seed: int | None,
):
    x_train_np = np.asarray(train_states_by_level, dtype=float)
    x_test_np = np.asarray(test_states_by_level, dtype=float)
    if x_train_np.ndim < 3 or x_test_np.ndim < 3:
        raise ValueError(
            "crossfit_q all-level states must be (levels, paths, features)"
        )
    num_levels = int(x_train_np.shape[0])
    x_train_np = x_train_np.reshape(num_levels, x_train_np.shape[1], -1)
    x_test_np = x_test_np.reshape(num_levels, x_test_np.shape[1], -1)
    if x_test_np.shape[0] != num_levels or x_test_np.shape[2] != x_train_np.shape[2]:
        raise ValueError("train/test all-level states have incompatible shapes")

    y_train_np = np.asarray(train_labels, dtype=float)
    if y_train_np.ndim != 2:
        raise ValueError("crossfit_q MLP labels must be 2D")
    if x_train_np.shape[1] < 1:
        raise ValueError("crossfit_q MLP needs at least one training path")
    if y_train_np.shape[0] != x_train_np.shape[1]:
        raise ValueError("crossfit_q all-level labels must match training paths")
    if y_train_np.shape[1] < 1:
        raise ValueError("crossfit_q MLP needs at least one output")
    if int(query_conditioning["ambient_dim"]) != int(x_train_np.shape[2]):
        raise ValueError(
            "query ambient dimension does not match flattened crossfit_q states"
        )
    if int(query_conditioning["features"].shape[0]) != int(y_train_np.shape[1]):
        raise ValueError("query feature count must match crossfit_q label columns")

    times_np = np.asarray(level_times, dtype=float).reshape(-1)
    if times_np.shape != (num_levels,):
        raise ValueError(f"level_times shape {times_np.shape} != ({num_levels},)")
    if not np.isfinite(times_np).all():
        raise ValueError("crossfit_q level_times must be finite")

    state_flat = x_train_np.reshape(-1, x_train_np.shape[2])
    center = state_flat.mean(axis=0)
    scale = state_flat.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x_train_np = (x_train_np - center.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
    x_test_np = (x_test_np - center.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)

    q_features_np = np.asarray(query_conditioning["features"], dtype=float)
    q_center = q_features_np.mean(axis=0)
    q_scale = q_features_np.std(axis=0)
    q_scale = np.where(q_scale > 1e-12, q_scale, 1.0)
    q_features_np = (q_features_np - q_center) / q_scale

    time_center = float(times_np.mean())
    time_scale = float(times_np.std())
    if time_scale <= 1e-12:
        time_scale = 1.0
    time_features_np = ((times_np - time_center) / time_scale).reshape(num_levels, 1)

    active_dims_np = np.asarray(query_conditioning["active_dims"], dtype=np.int64)
    active_mask_np = np.asarray(query_conditioning["active_mask"], dtype=float)
    # thresholds into the same units as the standardised state, so the margin
    # x_d - t_d is scale-free (the centre cancels in the subtraction)
    thresholds_np = np.asarray(query_conditioning["thresholds"], dtype=float)
    t_norm_np = np.where(
        active_mask_np > 0,
        (thresholds_np - center[active_dims_np]) / scale[active_dims_np],
        0.0,
    )

    x_train = torch.as_tensor(x_train_np, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train_np, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(x_test_np, dtype=torch.float32, device=device)
    q_features = torch.as_tensor(q_features_np, dtype=torch.float32, device=device)
    active_dims = torch.as_tensor(active_dims_np, dtype=torch.long, device=device)
    active_mask = torch.as_tensor(active_mask_np, dtype=torch.float32, device=device)
    t_norm = torch.as_tensor(t_norm_np, dtype=torch.float32, device=device)
    time_features = torch.as_tensor(
        time_features_np, dtype=torch.float32, device=device
    )
    num_queries = int(q_features.shape[0])
    max_active = int(active_dims.shape[1])
    train_paths = int(x_train.shape[1])

    def triple_features(
        states: torch.Tensor, pair_idx: torch.Tensor, paths_per_level: int
    ):
        """Inputs for one (level, path, query) triple.

        The raw state is deliberately absent: the network sees only how far the
        state sits from the query corner in the queried coordinates, plus the
        query's size, its mass and the level time.
        """
        level_path_idx = torch.div(pair_idx, num_queries, rounding_mode="floor")
        query_idx = pair_idx - level_path_idx * num_queries
        path_idx = (
            level_path_idx
            - torch.div(level_path_idx, paths_per_level, rounding_mode="floor")
            * paths_per_level
        )
        level_idx = torch.div(level_path_idx, paths_per_level, rounding_mode="floor")
        state_part = states[level_idx, path_idx]
        query_part = q_features[query_idx]
        dims = active_dims[query_idx]
        mask = active_mask[query_idx]
        margin = (torch.gather(state_part, 1, dims) - t_norm[query_idx]) * mask
        time_part = time_features[level_idx]
        return torch.cat([query_part, margin, time_part], dim=1)

    input_dim = int(q_features.shape[1]) + max_active + 1

    def materialize_triple_features(states: torch.Tensor, paths_per_level: int):
        pair_count = int(num_levels) * int(paths_per_level) * int(num_queries)
        features = torch.empty(
            (pair_count, input_dim),
            device=device,
            dtype=torch.float32,
        )
        chunk_size = int(_CROSSFIT_Q_FEATURE_CACHE_CHUNK_SIZE)
        for start in range(0, pair_count, chunk_size):
            end = min(start + chunk_size, pair_count)
            pair_idx = torch.arange(start, end, device=device, dtype=torch.long)
            features[start:end] = triple_features(states, pair_idx, paths_per_level)
        return features

    train_features = materialize_triple_features(x_train, train_paths)
    train_targets = y_train.unsqueeze(0).expand(num_levels, -1, -1).reshape(-1, 1)
    del x_train

    if seed is None:
        with _CROSSFIT_Q_MLP_INIT_LOCK:
            model = _crossfit_q_build_mlp(input_dim, 1, params).to(device=device)
    else:
        fork_devices = (
            list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        )
        with _CROSSFIT_Q_MLP_INIT_LOCK:
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(int(seed))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
                model = _crossfit_q_build_mlp(input_dim, 1, params).to(device=device)
    optimizer_params = {
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
    }
    if device.type == "cuda":
        optimizer_params["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    pair_count = int(num_levels) * int(train_paths) * int(num_queries)
    batch_size = min(int(params["batch_size"]), int(pair_count))
    epochs = int(params["epochs"])
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed) + 104729)

    loss_mode = str(params["loss"])

    def supervised_loss(logits: torch.Tensor, targets: torch.Tensor):
        if loss_mode == "bce":
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        return torch.nn.functional.mse_loss(torch.sigmoid(logits), targets)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(pair_count, device=device, generator=gen)
        for start in range(0, int(order.numel()), batch_size):
            idx = order[start : start + batch_size]
            xb = train_features[idx]
            yb = train_targets[idx]
            logits = model(xb)
            loss = supervised_loss(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    del train_features, train_targets
    model.eval()
    test_paths = int(x_test.shape[1])
    test_pair_count = int(num_levels) * int(test_paths) * int(num_queries)
    test_features = materialize_triple_features(x_test, test_paths)
    del x_test
    predictions_flat = np.empty(test_pair_count, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, test_pair_count, batch_size):
            end = min(start + batch_size, test_pair_count)
            predictions_flat[start:end] = (
                torch.sigmoid(model(test_features[start:end]))
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
    return predictions_flat.reshape(num_levels, test_paths, num_queries)


def _crossfit_q_predict_all_levels(
    states_by_level: np.ndarray,
    labels: np.ndarray,
    *,
    query_conditioning,
    level_times: np.ndarray,
    mlp_params: Mapping[str, Any] | None,
    mlp_device: str | None,
    seed: int | None,
    manage_mlp_num_threads: bool = True,
):
    """Fit Q on all pilot paths and predict on the same paths.

    The fit is in-sample. `Q_raw = mean(2*p*y - p^2)` equals `E[Q^2]` minus the
    regressor's mean-squared error for any `p`, so a restricted regressor biases
    it downward while an in-sample fit biases it upward; the projection in
    `_crossfit_q_project_sequences` re-imposes the structure the martingale
    guarantees.
    """
    predictions = _crossfit_q_mlp_predict_all_levels(
        states_by_level,
        labels,
        states_by_level,
        query_conditioning,
        level_times,
        params=mlp_params,
        runner_device=mlp_device,
        seed=seed,
        manage_num_threads=manage_mlp_num_threads,
    )
    finite = np.isfinite(predictions)
    if not finite.all():
        fallback = np.broadcast_to(labels.mean(axis=0), predictions.shape)
        predictions = np.where(finite, predictions, fallback)
    return np.clip(predictions, 0.0, 1.0)


def _crossfit_q_project_sequences_reference(Q_raw: np.ndarray, F_hat: np.ndarray):
    Q_raw = np.asarray(Q_raw, dtype=float)
    F_hat = np.asarray(F_hat, dtype=float)
    num_times, obs_dim = Q_raw.shape
    weights = np.ones(num_times, dtype=float)
    weights[-1] = 1e12
    Q_proj = np.empty_like(Q_raw, dtype=float)

    for obs_idx in range(obs_dim):
        F = float(np.clip(F_hat[obs_idx], 0.0, 1.0))
        lower = F * F
        upper = F
        y = np.clip(Q_raw[:, obs_idx], lower, upper)
        y[-1] = F
        if math.isclose(lower, upper, rel_tol=0.0, abs_tol=0.0):
            q = np.full(num_times, F, dtype=float)
        else:
            q = _weighted_pava_non_decreasing(y, weights)
            q = np.clip(q, lower, upper)
            q[-1] = F
        Q_proj[:, obs_idx] = q
    return Q_proj


def _crossfit_q_project_sequences(Q_raw: np.ndarray, F_hat: np.ndarray):
    Q_raw = np.asarray(Q_raw, dtype=float)
    F_hat = np.asarray(F_hat, dtype=float)
    if (
        _crossfit_q_project_sequences_numba is None
        or not np.isfinite(Q_raw).all()
        or not np.isfinite(F_hat).all()
    ):
        return _crossfit_q_project_sequences_reference(Q_raw, F_hat)
    Q_proj = _crossfit_q_project_sequences_numba(
        np.ascontiguousarray(Q_raw, dtype=np.float32),
        np.ascontiguousarray(F_hat, dtype=np.float32),
    )
    return Q_proj


def _estimate_crossfit_q_variance_payload(
    states_by_level: np.ndarray,
    labels: np.ndarray,
    *,
    query_spec,
    level_times,
    mlp_params: Mapping[str, Any] | None = None,
    mlp_device: str | None = None,
    seed: int | None = None,
    manage_mlp_num_threads: bool = True,
):
    labels = np.asarray(labels, dtype=float)
    if labels.ndim != 2:
        raise ValueError("crossfit_q labels must be a 2D array")
    num_paths, obs_dim = labels.shape
    num_levels = int(states_by_level.shape[0])
    if num_paths < 2:
        raise ValueError("crossfit_q needs at least two pilot paths")
    query_conditioning = _crossfit_q_query_conditioning(query_spec)
    if int(query_conditioning["features"].shape[0]) != int(obs_dim):
        raise ValueError("query feature count must match crossfit_q labels")
    level_times = np.asarray(level_times, dtype=float).reshape(-1)
    if level_times.shape != (num_levels,):
        raise ValueError(f"level_times shape {level_times.shape} != ({num_levels},)")

    mlp_params_normalized = _crossfit_q_normalize_mlp_params(mlp_params)

    F_hat = labels.mean(axis=0)
    Q_raw = np.empty((num_levels + 1, obs_dim), dtype=float)
    predictions_by_level = _crossfit_q_predict_all_levels(
        states_by_level,
        labels,
        query_conditioning=query_conditioning,
        level_times=level_times,
        mlp_params=mlp_params_normalized,
        mlp_device=mlp_device,
        seed=seed,
        manage_mlp_num_threads=manage_mlp_num_threads,
    )
    for level_idx in range(num_levels):
        level_predictions = predictions_by_level[level_idx]
        Q_raw[level_idx] = np.mean(
            2.0 * level_predictions * labels - level_predictions**2,
            axis=0,
        )

    Q_raw[-1] = F_hat
    Q_proj = _crossfit_q_project_sequences(Q_raw, F_hat)
    variance2 = np.maximum(Q_proj[1:] - Q_proj[:-1], 0.0)
    tau2 = np.maximum(Q_proj[0] - F_hat * F_hat, 0.0)
    return variance2, tau2

def _normalize_query_params(params: Mapping[str, Any] | None):
    """Validate the phase-1 query configuration.

    Six knobs, one dict. `k_max` caps the coordinate-subset size; queries are
    stored in a (Q, k_max) padded array, so it also bounds the regressor's input
    width.
    """
    merged = dict(_DEFAULT_QUERY_PARAMS)
    if params:
        unknown = sorted(set(params) - set(_DEFAULT_QUERY_PARAMS))
        if unknown:
            raise ValueError(f"Unknown query_params: {unknown}")
        merged.update(dict(params))

    merged["num_queries"] = int(merged["num_queries"])
    if merged["num_queries"] < 1:
        raise ValueError("query_params.num_queries must be >= 1")

    merged["k_max"] = int(merged["k_max"])
    if merged["k_max"] < 1:
        raise ValueError("query_params.k_max must be >= 1")

    merged["mass_min"] = float(merged["mass_min"])
    merged["mass_max"] = float(merged["mass_max"])
    if not (
        math.isfinite(merged["mass_min"])
        and math.isfinite(merged["mass_max"])
        and 0.0 < merged["mass_min"] < merged["mass_max"] < 1.0
    ):
        raise ValueError("query_params requires 0 < mass_min < mass_max < 1")

    merged["mass_bins"] = int(merged["mass_bins"])
    if merged["mass_bins"] < 1:
        raise ValueError("query_params.mass_bins must be >= 1")

    return merged


def _mass_grid(count: int, mass_min: float, mass_max: float):
    if int(count) < 1:
        return np.empty(0, dtype=float)
    return mass_min + (np.arange(int(count), dtype=float) + 0.5) / float(count) * (
        mass_max - mass_min
    )


def _empirical_query_weights(
    subset_sizes: np.ndarray, empirical_mass: np.ndarray, mass_bins: int
):
    """Balance subset sizes, then occupied empirical-CDF bins within each size."""
    subset_sizes = np.asarray(subset_sizes, dtype=np.int64)
    empirical_mass = np.asarray(empirical_mass, dtype=float)
    if subset_sizes.ndim != 1 or empirical_mass.shape != subset_sizes.shape:
        raise ValueError("query weight metadata must be 1D with matching shapes")
    if not np.isfinite(empirical_mass).all():
        raise ValueError("empirical query CDF masses must be finite")
    mass_bins = int(mass_bins)
    if mass_bins < 1:
        raise ValueError("query weights require at least one mass bin")
    bins = np.floor(
        np.clip(empirical_mass, 0.0, 1.0 - 1e-15) * mass_bins
    ).astype(int)
    bins = np.clip(bins, 0, int(mass_bins) - 1)
    weights = np.zeros(subset_sizes.shape[0], dtype=float)
    unique_sizes = np.unique(subset_sizes)
    if unique_sizes.size == 0:
        raise ValueError("query weights require at least one query")
    for subset_size in unique_sizes:
        size_mask = subset_sizes == subset_size
        occupied_bins = np.unique(bins[size_mask])
        for mass_bin in occupied_bins:
            mask = size_mask & (bins == mass_bin)
            weights[mask] = 1.0 / (
                int(unique_sizes.size) * int(occupied_bins.size) * int(mask.sum())
            )
    weights /= float(weights.sum())
    return weights


def _generate_grid_free_queries(samples, query_params, *, design_seed):
    """Sparse lower-orthant queries: a random coordinate subset per query, with
    the corner at per-coordinate empirical quantiles of the pilot samples.

    For each query draw a size ``k`` log-uniform on {1..min(D, k_max)}, a random
    subset ``S`` of ``k`` coordinates, a mass ``m`` from an even grid over
    [mass_min, mass_max], and simplex weights ``w ~ Dirichlet(1_k)``. The corner
    sits at the ``m ** w_j`` quantile of coordinate ``S[j]``.

    ``prod_j m**w_j == m`` exactly because ``sum_j w_j == 1``, and every rank
    lands in ``(m, 1)`` on its own -- so there is nothing to clip, rescale or
    validate. ``m`` is the orthant's mass only under independent coordinates; it
    is a spreading device that keeps queries non-degenerate at every ``k``, not
    a claim about the true mass.

    ``design_seed`` is the run's global index, so every run draws its own design
    and the estimator is never conditioned on one arbitrary query design.
    """
    query_params = _normalize_query_params(query_params)
    samples_np = coerce_samples_np(samples)
    if samples_np.shape[0] < 1:
        raise ValueError("phase 1 needs at least one terminal sample")
    dim = int(samples_np.shape[1])
    total = int(query_params["num_queries"])
    k_max = min(dim, int(query_params["k_max"]))
    rng = np.random.default_rng(int(design_seed) + 104729 * dim)

    sizes = np.clip(
        np.round(np.exp(rng.uniform(0.0, math.log(k_max), size=total))), 1, k_max
    ).astype(np.int64)
    masses = _mass_grid(
        total, float(query_params["mass_min"]), float(query_params["mass_max"])
    )

    width = int(sizes.max())
    active_dims = np.full((total, width), -1, dtype=np.int64)
    thresholds = np.full((total, width), np.nan, dtype=float)
    ranks_used = np.full((total, width), np.nan, dtype=float)

    for query_idx in range(total):
        k = int(sizes[query_idx])
        mass = float(masses[query_idx])
        dims = np.sort(rng.choice(dim, size=k, replace=False))
        ranks = np.exp(rng.dirichlet(np.ones(k)) * math.log(mass))
        active_dims[query_idx, :k] = dims
        ranks_used[query_idx, :k] = ranks
        for pos, dim_idx in enumerate(dims):
            thresholds[query_idx, pos] = np.quantile(samples_np[:, dim_idx], ranks[pos])

    return {
        "kind": "sparse_subset_lower_orthant",
        "ambient_dim": int(dim),
        "thresholds": thresholds,
        "active_dims": active_dims,
        "active_sizes": sizes,
        "target_mass": masses,
        "ranks": ranks_used,
    }


def _lower_orthant_labels(samples, query_points):
    values = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
    values = values.reshape(-1, values.shape[-1])
    if not isinstance(query_points, Mapping):
        raise ValueError("grid-free phase 1 requires sparse subset query specs")
    if query_points.get("kind") != "sparse_subset_lower_orthant":
        raise ValueError(f"unknown query spec kind {query_points.get('kind')!r}")
    active_dims_np = np.asarray(query_points["active_dims"], dtype=np.int64)
    thresholds_np = np.asarray(query_points["thresholds"], dtype=float)
    if active_dims_np.shape != thresholds_np.shape:
        raise ValueError("query active_dims and thresholds must have matching shapes")
    if int(query_points.get("ambient_dim", values.shape[1])) != int(values.shape[1]):
        raise ValueError("query ambient dimension does not match samples")
    if np.any(active_dims_np >= int(values.shape[1])):
        raise ValueError("query active dimension exceeds sample dimension")
    result = torch.ones(
        (values.shape[0], active_dims_np.shape[0]),
        device=values.device,
        dtype=torch.bool,
    )
    active_dims = torch.as_tensor(
        active_dims_np, device=values.device, dtype=torch.long
    )
    thresholds = torch.as_tensor(
        thresholds_np, device=values.device, dtype=values.dtype
    )
    for pos in range(active_dims.shape[1]):
        dims = active_dims[:, pos]
        active = dims >= 0
        if not bool(active.any()):
            continue
        query_cols = torch.nonzero(active, as_tuple=False).flatten()
        selected = values[:, dims[active]] <= thresholds[active, pos].reshape(1, -1)
        result[:, query_cols] &= selected
    return result


def _query_metadata(query_spec, labels: np.ndarray, *, mass_bins: int):
    labels = np.asarray(labels, dtype=bool)
    if labels.ndim != 2:
        raise ValueError("query labels must be a 2D array")
    if labels.shape[0] < 1:
        raise ValueError("query metadata requires at least one path")
    num_queries = labels.shape[1]
    if not isinstance(query_spec, Mapping):
        raise ValueError("query metadata requires sparse subset query specs")
    subset_size = np.asarray(query_spec["active_sizes"], dtype=np.int64)
    if subset_size.shape != (num_queries,):
        raise ValueError("query metadata and labels disagree on query count")
    empirical_mass = labels.mean(axis=0).astype(float)
    weights = _empirical_query_weights(subset_size, empirical_mass, int(mass_bins))
    return {
        "weights": weights,
        "empirical_mass": empirical_mass,
        "n_paths": int(labels.shape[0]),
    }


def _simulate_crossfit_q_trajectories(
    runner,
    *,
    chunk_size: int,
    paths_per_run: int,
    split_points: Sequence[Any],
    max_sampling_batch_size=None,
    generator=None,
):
    total_paths = int(chunk_size) * int(paths_per_run)
    x, run_ids = _sample_prior_by_run_batches(
        runner,
        [int(paths_per_run)] * int(chunk_size),
        max_sampling_batch_size=max_sampling_batch_size,
        generator=generator,
    )
    start_points = [runner.start_time] + list(split_points)
    end_points = list(split_points) + [runner.end_time]
    states_by_level = []
    for start_t, end_t in zip(start_points, end_points):
        states_by_level.append(x.detach().reshape(total_paths, -1).clone())
        x = _sample_segment_by_run_batches(
            runner,
            x,
            run_ids,
            num_runs=chunk_size,
            start_time=start_t,
            end_time=end_t,
            max_sampling_batch_size=max_sampling_batch_size,
            generator=generator,
        )

    x0 = runner.postprocess_samples(x)
    states = (
        torch.stack(states_by_level, dim=0)
        .cpu()
        .numpy()
        .astype(float, copy=False)
        .reshape(len(states_by_level), int(chunk_size), int(paths_per_run), -1)
    )
    return states, x0


def _grid_free_query_payload(samples, query_params, *, design_seed):
    query_spec = _generate_grid_free_queries(
        samples, query_params, design_seed=design_seed
    )
    labels = _lower_orthant_labels(samples, query_spec)
    labels_np = labels.detach().cpu().numpy().astype(np.bool_, copy=False)
    query_metadata = _query_metadata(
        query_spec,
        labels_np,
        mass_bins=int(_normalize_query_params(query_params)["mass_bins"]),
    )
    return query_spec, labels, query_metadata


def _estimate_crossfit_q_variance_for_run(
    run_idx: int,
    states_by_level: np.ndarray,
    spec: Mapping[str, Any],
    *,
    level_times,
    mlp_params: Mapping[str, Any],
    mlp_device: str | None,
    seed: int | None,
    manage_mlp_num_threads: bool,
):
    return int(run_idx), _estimate_crossfit_q_variance_payload(
        states_by_level,
        spec["labels"],
        query_spec=spec["query_spec"],
        level_times=level_times,
        mlp_params=mlp_params,
        mlp_device=mlp_device,
        seed=seed,
        manage_mlp_num_threads=manage_mlp_num_threads,
    )


# --- Phase 1 builders -------------------------------------------------------


def _run_crossfit_q_phase1_sampling_batch(
    runner,
    *,
    B1,
    split_percentages,
    crossfit_q_mlp_run_parallelism=1,
    crossfit_q_mlp_params=None,
    query_params=None,
    run_index_offset=0,
    reuse_phase1_samples,
    chunk_size,
    debug=False,
    max_sampling_batch_size=None,
    generator=None,
):
    query_params = _normalize_query_params(query_params)
    mlp_params_normalized = _crossfit_q_normalize_mlp_params(crossfit_q_mlp_params)
    mlp_run_parallelism = _crossfit_q_normalize_mlp_run_parallelism(
        crossfit_q_mlp_run_parallelism
    )
    mlp_device = _crossfit_q_mlp_device(mlp_params_normalized, str(runner.device))

    _, split_points = runner.resolve_split_percentages(split_percentages)
    level_times = [runner.start_time] + list(split_points)
    segment_costs = runner.segment_costs(split_points)
    full_path_cost = int(np.sum(segment_costs))
    if full_path_cost <= 0:
        raise ValueError("crossfit_q requires positive full-path simulation cost")
    paths_per_run = int(B1) // full_path_cost
    if paths_per_run < 2:
        raise ValueError(
            f"B1={B1} is too small for crossfit_q; need at least "
            f"{2 * full_path_cost} for two unsplit pilot paths"
        )
    used_B1 = int(paths_per_run * full_path_cost)
    _debug(
        debug,
        f"Phase 1 crossfit_q: chunk={chunk_size} paths_per_run={paths_per_run} "
        f"mlp_workers={min(mlp_run_parallelism, int(chunk_size))} "
        f"mlp_device={mlp_device} mlp_num_threads={mlp_params_normalized.get('num_threads')}",
    )

    states_by_run, x0 = _simulate_crossfit_q_trajectories(
        runner,
        chunk_size=chunk_size,
        paths_per_run=paths_per_run,
        split_points=split_points,
        max_sampling_batch_size=max_sampling_batch_size,
        generator=generator,
    )
    phase1_x0_by_run: List[Any] = [None] * chunk_size
    x0_by_run = list(torch.split(x0, int(paths_per_run)))
    if reuse_phase1_samples:
        x0_np = coerce_samples_np(x0).reshape(int(chunk_size), int(paths_per_run), -1)
        phase1_x0_by_run = [x0_np[idx] for idx in range(int(chunk_size))]

    run_specs = []
    for run_idx in range(chunk_size):
        query_spec, labels_tensor, query_metadata = _grid_free_query_payload(
            x0_by_run[run_idx],
            query_params,
            design_seed=int(run_index_offset) + int(run_idx),
        )
        labels = labels_tensor.detach().cpu().numpy().astype(np.bool_, copy=False)
        run_specs.append(
            {
                "query_spec": query_spec,
                "labels": labels,
                "query_metadata": query_metadata,
            }
        )

    estimates_by_run: List[Any] = [None] * int(chunk_size)
    effective_mlp_workers = min(int(mlp_run_parallelism), int(chunk_size))
    if effective_mlp_workers <= 1:
        for run_idx, spec in enumerate(run_specs):
            _, estimates_by_run[run_idx] = _estimate_crossfit_q_variance_for_run(
                run_idx,
                states_by_run[:, run_idx, :, :],
                spec,
                level_times=level_times,
                mlp_params=mlp_params_normalized,
                mlp_device=str(runner.device),
                seed=7919 + int(run_idx),
                manage_mlp_num_threads=True,
            )
    else:
        _debug(
            debug,
            f"Phase 1 crossfit_q MLP run parallelism enabled: "
            f"workers={effective_mlp_workers} chunk={chunk_size}",
        )
        with _TorchNumThreadsContext(
            mlp_params_normalized.get("num_threads"), mlp_device.type == "cpu"
        ):
            with ThreadPoolExecutor(max_workers=effective_mlp_workers) as mlp_executor:
                futures = []
                for run_idx, spec in enumerate(run_specs):
                    futures.append(
                        (
                            run_idx,
                            mlp_executor.submit(
                                _estimate_crossfit_q_variance_for_run,
                                run_idx,
                                states_by_run[:, run_idx, :, :],
                                spec,
                                level_times=level_times,
                                mlp_params=mlp_params_normalized,
                                mlp_device=str(runner.device),
                                seed=7919 + int(run_idx),
                                manage_mlp_num_threads=False,
                            ),
                        )
                    )
                for run_idx, future in futures:
                    _, estimates_by_run[run_idx] = future.result()

    payloads = []
    for run_idx, spec in enumerate(run_specs):
        variance2, tau2 = estimates_by_run[run_idx]
        payloads.append(
            {
                "split_points": list(split_points),
                "variance2_per_level": variance2,
                "tau2": tau2,
                "used_B1": int(used_B1),
                "phase1_x0_samples": phase1_x0_by_run[run_idx],
                "query_metadata": spec.get("query_metadata"),
            }
        )
    return payloads


# --- Phase 1 → Phase 2 ------------------------------------------------------


def _floor_split_sampling_cost(runner, split_points, split_factors, n0: int) -> float:
    n0 = int(n0)
    if n0 < 1:
        raise ValueError("n0 must be at least 1")
    if not split_points:
        return n0 * int(runner.segment_cost(runner.start_time, runner.end_time))

    if len(split_factors) != len(split_points):
        raise ValueError("split_factors must match split_points length")
    starts = [runner.start_time] + list(split_points)
    ends = list(split_points) + [runner.end_time]

    current_count = n0
    cost = current_count * int(runner.segment_cost(starts[0], ends[0]))
    for idx, split_factor in enumerate(split_factors):
        current_count = floor_split_total_count(current_count, float(split_factor))
        cost += current_count * int(runner.segment_cost(starts[idx + 1], ends[idx + 1]))
    return float(cost)


def _max_floor_split_roots_for_budget(
    runner,
    split_points,
    split_factors,
    *,
    budget: int,
    expected_cost_per_root: float,
) -> int:
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if expected_cost_per_root <= 0.0 or not math.isfinite(expected_cost_per_root):
        raise ValueError("expected_cost_per_root must be finite and positive")

    cost_for_one = _floor_split_sampling_cost(runner, split_points, split_factors, 1)
    if cost_for_one > budget:
        raise ValueError(
            f"B2={budget} too small; floor split cost for one root is {cost_for_one:.6f}"
        )

    upper = max(1, int(budget // expected_cost_per_root))
    while (
        _floor_split_sampling_cost(runner, split_points, split_factors, upper) > budget
    ):
        upper //= 2
        if upper < 1:
            raise ValueError(
                f"B2={budget} too small; floor split cost for one root is {cost_for_one:.6f}"
            )

    lower = upper
    probe = max(upper * 2, 2)
    while (
        _floor_split_sampling_cost(runner, split_points, split_factors, probe) <= budget
    ):
        lower = probe
        probe *= 2

    high = probe - 1
    while lower < high:
        mid = (lower + high + 1) // 2
        if (
            _floor_split_sampling_cost(runner, split_points, split_factors, mid)
            <= budget
        ):
            lower = mid
        else:
            high = mid - 1
    return int(lower)


def _solve_phase1_allocation(
    payload,
    runner,
    *,
    B,
    optimization_mode="monotone",
):
    split_points = payload["split_points"]
    seg_costs = runner.segment_costs(split_points)
    total = float(np.sum(seg_costs))
    if total <= 0.0:
        raise ValueError("segment costs must sum to a positive value")
    cost_weights = [c / total for c in seg_costs]

    split_factors = _solve_optimal_split_factors(
        payload["variance2_per_level"],
        payload["tau2"],
        cost_weights,
        optimization_mode=optimization_mode,
        query_metadata=payload.get("query_metadata"),
    )
    cost_per_root = runner.expected_cost_per_root(split_points, split_factors)
    B2 = B - int(payload["used_B1"])
    n0 = _max_floor_split_roots_for_budget(
        runner,
        split_points,
        split_factors,
        budget=B2,
        expected_cost_per_root=cost_per_root,
    )
    return {
        "split_points": split_points,
        "N_i": split_factors,
        "n0": int(n0),
        "used_B1": int(payload["used_B1"]),
        "phase1_x0_samples": payload.get("phase1_x0_samples"),
    }


def run_estimate_and_sample(
    runner,
    comparison_state: Any,
    *,
    comparison_mode: str,
    metrics: Sequence[str] = ("ks",),
    B: int,
    B1: int,
    split_percentages,
    reuse_phase1_samples: bool,
    n_runs: int,
    seed: int | None,
    crossfit_q_mlp_run_parallelism: int = CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM,
    crossfit_q_mlp_params: Mapping[str, Any] | None = None,
    query_params: Mapping[str, Any] | None = None,
    optimization_mode: str = "monotone",
    debug: bool = False,
    n_parallel: int = 1,
    run_offset: int = 0,
    return_trial_results: bool = False,
    max_sampling_batch_size=None,
):
    if not 0 <= B1 < B:
        raise ValueError("require 0 <= B1 < B")
    if optimization_mode not in SUPPORTED_OPTIMIZATION_MODES:
        raise ValueError(f"unknown optimization_mode '{optimization_mode}'")

    phase1_extra = {
        "query_params": dict(_normalize_query_params(query_params)),
        "crossfit_q_mlp_run_parallelism": int(
            _crossfit_q_normalize_mlp_run_parallelism(crossfit_q_mlp_run_parallelism)
        ),
        "crossfit_q_mlp_params": dict(
            _crossfit_q_normalize_mlp_params(crossfit_q_mlp_params)
        ),
    }

    _debug(
        debug,
        f"Starting estimate_and_sample B={B} B1={B1} runner={runner.runner_name} "
        f"optimizer={optimization_mode}",
    )

    chunks = list(iter_run_chunks(n_runs, n_parallel))
    trial_results: List[Any] = [None] * n_runs
    workers = max(1, min(n_runs, n_parallel))

    with ThreadPoolExecutor(max_workers=workers) as alloc_executor:
        allocation_futures = []
        for start, end in tqdm(chunks, desc="Phase 1", leave=False):
            chunk_size = end - start
            run_seed = None if seed is None else seed + run_offset + start
            payloads = _run_crossfit_q_phase1_sampling_batch(
                runner,
                B1=B1,
                run_index_offset=run_offset + start,
                split_percentages=split_percentages,
                reuse_phase1_samples=reuse_phase1_samples,
                chunk_size=chunk_size,
                debug=debug,
                max_sampling_batch_size=max_sampling_batch_size,
                generator=make_torch_generator(run_seed, runner.device),
                **phase1_extra,
            )
            for local_idx, payload in enumerate(payloads):
                run_idx = start + local_idx
                future = alloc_executor.submit(
                    _solve_phase1_allocation,
                    payload,
                    runner,
                    B=B,
                    optimization_mode=optimization_mode,
                )
                allocation_futures.append((run_idx, future))

        for run_idx, future in tqdm(
            allocation_futures, desc="Optimize N_i", leave=False
        ):
            trial_results[run_idx] = future.result()

    if any(t is None for t in trial_results):
        raise RuntimeError("Phase 1 allocation did not produce all trial results")

    metric_workers = max(1, min(int(n_parallel), n_runs))
    metric_futures: List[Tuple[int, Any]] = []
    with ThreadPoolExecutor(max_workers=metric_workers) as metric_executor:
        for start, end in tqdm(chunks, desc="Phase 2", leave=False):
            chunk = trial_results[start:end]
            run_seed = (
                None if seed is None else seed + PHASE2_SEED_OFFSET + run_offset + start
            )
            samples_by_run, _, _ = runner.run_split_batch(
                n0_by_run=[int(t["n0"]) for t in chunk],
                split_points=chunk[0]["split_points"],
                split_factors_by_run=[t["N_i"] for t in chunk],
                generator=make_torch_generator(run_seed, runner.device),
                max_sampling_batch_size=max_sampling_batch_size,
            )
            submit_sampling_trial_result_futures(
                executor=metric_executor,
                futures=metric_futures,
                run_indices=range(start, end),
                samples_by_run=samples_by_run,
                runner=runner,
                comparison_mode=comparison_mode,
                metric_states=comparison_state,
                metrics=metrics,
                phase1_x0_samples_by_run=[t.get("phase1_x0_samples") for t in chunk],
            )
        for run_idx, future in tqdm(metric_futures, desc="Metrics", leave=False):
            trial_results[run_idx].update(future.result())

    result = {
        "mode": "estimate_and_sample",
        "B": int(B),
        "B1": int(B1),
        **summarize_sampling_trials(trial_results, metrics),
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result
