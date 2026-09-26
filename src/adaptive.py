import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from numba import njit

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only PyTorch installations do not ship Triton.
    triton = None
    tl = None

from runners.splitting import (
    InsufficientSplitBudgetError,
    apply_to_run_batches,
    append_by_counts as _append_by_counts,
    normalize_max_sampling_batch_size,
    run_mixture_batch,
    split_counts_by_run_batches,
)
from runners.trees import (
    InsufficientTreeBudgetError,
    design_cost,
    design_mixture,
    mean_split_factors,
)
from uniform_c import max_feasible_c, solve_uniform_c
from metrics.utils import coerce_samples_np
from trials import (
    PHASE2_SEED_OFFSET,
    iter_run_chunks,
    make_torch_generator,
    submit_sampling_trial_result_futures,
    summarize_sampling_trials,
)

log = logging.getLogger(__name__)

CROSSFIT_Q_DEFAULT_FOLDS = 1
CROSSFIT_Q_DEFAULT_MLP_HIDDEN_DIMS = [128, 64]
CROSSFIT_Q_DEFAULT_MLP_ACTIVATION = "silu"
CROSSFIT_Q_DEFAULT_MLP_EPOCHS = 5
CROSSFIT_Q_DEFAULT_MLP_BATCH_SIZE = 24000
CROSSFIT_Q_DEFAULT_MLP_LR = 5e-3
CROSSFIT_Q_DEFAULT_MLP_WEIGHT_DECAY = 3e-4
CROSSFIT_Q_DEFAULT_MLP_LOSS = "mse"
CROSSFIT_Q_DEFAULT_MLP_DEVICE = "runner"
CROSSFIT_Q_DEFAULT_MLP_NUM_THREADS = 2
CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM = 25
CROSSFIT_Q_DEFAULT_NUM_QUERIES = 1024
CROSSFIT_Q_DEFAULT_K_MAX = 64
CROSSFIT_Q_DEFAULT_MASS_MIN = 0.05
CROSSFIT_Q_DEFAULT_MASS_MAX = 0.95

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
}

_CROSSFIT_Q_MLP_INIT_LOCK = Lock()
_FAST_FW_RELATIVE_TOL = 3e-4
_FAST_FW_MAX_ITERS = 2_000
_FAST_FW_LINE_SEARCH_ITERS = 6
_QUERY_LABEL_MAX_TEMPORARY_BYTES = 512 * 1024 * 1024
_QUERY_LABEL_MAX_TILE_ELEMENTS = 4096
_MONOTONE_CVAR95_ALPHA = 0.95
SUPPORTED_OPTIMIZATION_MODES = {"monotone", "monotone_cvar95", "learned_c"}
_INFEASIBLE_ALLOCATION_RETRY_SEED_OFFSET = 1 << 40


def _debug(enabled: bool, message: str):
    if enabled:
        log.debug(message)


def _synchronize_runner_device(runner):
    device = torch.device(runner.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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

    @njit(cache=True, nogil=True)
    def _weighted_pava_non_decreasing_numba(values, weights):
        count = values.size
        starts = np.empty(count, dtype=np.int64)
        ends = np.empty(count, dtype=np.int64)
        block_weights = np.empty(count, dtype=np.float64)
        block_sums = np.empty(count, dtype=np.float64)
        blocks = 0
        for index in range(count):
            starts[blocks] = index
            ends[blocks] = index + 1
            block_weights[blocks] = weights[index]
            block_sums[blocks] = weights[index] * values[index]
            blocks += 1
            while blocks >= 2:
                left = blocks - 2
                right = blocks - 1
                if (
                    block_sums[left] / block_weights[left]
                    <= block_sums[right] / block_weights[right]
                ):
                    break
                ends[left] = ends[right]
                block_weights[left] += block_weights[right]
                block_sums[left] += block_sums[right]
                blocks -= 1

        projected = np.empty_like(values)
        for block in range(blocks):
            level = block_sums[block] / block_weights[block]
            for index in range(starts[block], ends[block]):
                projected[index] = level
        return projected

    @njit(cache=True, nogil=True)
    def _monotone_dual_numba(profile, cost_w):
        pooled = _weighted_pava_non_decreasing_numba(profile / cost_w, cost_w)
        raw = np.sqrt(np.maximum(pooled, 0.0))
        maximum = 0.0
        valid = True
        for level in range(raw.size):
            valid = valid and np.isfinite(raw[level])
            maximum = max(maximum, raw[level])
        if not valid or maximum <= 0.0:
            raw[:] = 1.0
        raw = np.maximum(raw, 1e-12)

        normalizer = 0.0
        for level in range(raw.size):
            normalizer += cost_w[level] * raw[level]
        simplex = np.empty_like(raw)
        value = 0.0
        for level in range(raw.size):
            simplex[level] = cost_w[level] * raw[level] / normalizer
            value += profile[level] * cost_w[level] / simplex[level]
        return value, simplex

    @njit(cache=True, nogil=True)
    def _monotone_losses_numba(weighted_M, simplex):
        losses = np.empty(weighted_M.shape[0], dtype=np.float64)
        inverse = 1.0 / simplex
        for row in range(weighted_M.shape[0]):
            value = 0.0
            for level in range(weighted_M.shape[1]):
                value += weighted_M[row, level] * inverse[level]
            losses[row] = value
        return losses

    @njit(cache=True, nogil=True)
    def _monotone_directional_derivative(profile, target, cost_w, simplex):
        value = 0.0
        for level in range(profile.size):
            value += (
                (target[level] - profile[level])
                * cost_w[level]
                / simplex[level]
            )
        return value

    @njit(cache=True, nogil=True)
    def _monotone_derivative_line_search_numba(
        profile, target, cost_w, line_search_iters
    ):
        base_value, base_simplex = _monotone_dual_numba(profile, cost_w)
        if np.array_equal(profile, target):
            return 0.0, base_value
        if (
            _monotone_directional_derivative(
                profile, target, cost_w, base_simplex
            )
            <= 0.0
        ):
            return 0.0, base_value

        target_value, target_simplex = _monotone_dual_numba(target, cost_w)
        if (
            _monotone_directional_derivative(
                profile, target, cost_w, target_simplex
            )
            >= 0.0
        ):
            return 1.0, target_value

        left = 0.0
        right = 1.0
        best_gamma = 0.0
        best_value = base_value
        if target_value > best_value:
            best_gamma = 1.0
            best_value = target_value
        for _ in range(line_search_iters):
            midpoint = 0.5 * (left + right)
            candidate = (1.0 - midpoint) * profile + midpoint * target
            value, simplex = _monotone_dual_numba(candidate, cost_w)
            derivative = _monotone_directional_derivative(
                profile, target, cost_w, simplex
            )
            if value > best_value:
                best_gamma = midpoint
                best_value = value
            if derivative > 0.0:
                left = midpoint
            else:
                right = midpoint
        return best_gamma, best_value

    @njit(cache=True, nogil=True)
    def _solve_frank_wolfe_monotone_allocation_numba(
        M, cost_w, relative_tol, max_iters, line_search_iters
    ):
        weighted_M = M * cost_w.reshape(1, -1)
        best_row = 0
        best_score = -1.0
        for row in range(M.shape[0]):
            score = 0.0
            for level in range(M.shape[1]):
                score += math.sqrt(max(weighted_M[row, level], 0.0))
            if score > best_score:
                best_score = score
                best_row = row

        profile = M[best_row].copy()
        best_simplex = cost_w.copy()
        best_upper = math.inf
        best_lower = 0.0
        converged = False
        for iteration in range(max_iters):
            lower, simplex = _monotone_dual_numba(profile, cost_w)
            best_lower = max(best_lower, lower)
            losses = _monotone_losses_numba(weighted_M, simplex)
            worst = int(np.argmax(losses))
            upper = losses[worst]
            if upper < best_upper:
                best_upper = upper
                best_simplex = simplex.copy()
            gap = max(best_upper - best_lower, 0.0) / max(
                abs(best_upper), np.finfo(np.float64).tiny
            )
            if gap <= relative_tol:
                converged = True
                break

            target = M[worst]
            gamma, candidate_lower = _monotone_derivative_line_search_numba(
                profile, target, cost_w, line_search_iters
            )
            if gamma <= 0.0 or candidate_lower <= lower + 1e-14:
                gamma = 2.0 / float(iteration + 3.0)
            profile = (1.0 - gamma) * profile + gamma * target

        if not converged:
            lower, simplex = _monotone_dual_numba(profile, cost_w)
            best_lower = max(best_lower, lower)
            losses = _monotone_losses_numba(weighted_M, simplex)
            upper = float(np.max(losses))
            if upper < best_upper:
                best_upper = upper
                best_simplex = simplex.copy()
            converged = (
                max(best_upper - best_lower, 0.0)
                / max(abs(best_upper), np.finfo(np.float64).tiny)
                <= relative_tol
            )
        return best_simplex, best_upper, best_lower, converged

    @njit(cache=True, nogil=True)
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
    _weighted_pava_non_decreasing_numba = None
    _crossfit_q_project_sequences_numba = None
    _solve_frank_wolfe_monotone_allocation_numba = None


def _monotone_simplex_from_profile(
    profile: np.ndarray,
    cost_w: np.ndarray,
    *,
    allocation_floor=1e-12,
):
    profile = np.maximum(np.asarray(profile, dtype=np.float64), 0.0)
    cost_w = np.asarray(cost_w, dtype=np.float64)
    values = np.ascontiguousarray(profile / cost_w)
    weights = np.ascontiguousarray(cost_w)
    h = (
        _weighted_pava_non_decreasing_numba(values, weights)
        if _weighted_pava_non_decreasing_numba is not None
        else _weighted_pava_non_decreasing(values, weights)
    )
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
    relative_tol: float = _FAST_FW_RELATIVE_TOL,
    max_iters: int = _FAST_FW_MAX_ITERS,
):
    M = np.ascontiguousarray(M, dtype=np.float64)
    cost_w = np.ascontiguousarray(cost_w, dtype=np.float64)
    num_points, num_levels = M.shape
    if num_points == 0 or num_levels == 0:
        raise ValueError("M must be non-empty")
    if cost_w.shape != (num_levels,) or np.any(cost_w <= 0.0):
        raise ValueError("cost_w must be positive and match M columns")
    if not np.isfinite(M).all() or not np.isfinite(cost_w).all():
        raise ValueError("optimizer inputs must be finite")
    M = np.maximum(M, 0.0)

    if _solve_frank_wolfe_monotone_allocation_numba is None:
        raise RuntimeError("the monotone optimizer requires numba")
    simplex, upper, lower, converged = (
        _solve_frank_wolfe_monotone_allocation_numba(
            M,
            cost_w,
            float(relative_tol),
            int(max_iters),
            int(_FAST_FW_LINE_SEARCH_ITERS),
        )
    )
    if not converged:
        gap = max(float(upper) - float(lower), 0.0) / max(
            abs(float(upper)), np.finfo(np.float64).tiny
        )
        raise RuntimeError(
            f"monotone optimizer did not certify tolerance {relative_tol:g}; "
            f"final relative gap was {gap:.6g}"
        )
    return simplex


def _cvar_profile(M: np.ndarray, losses: np.ndarray, *, alpha: float):
    """Average of ``M`` over the worst ``1 - alpha`` fraction of queries.

    Queries carry equal weight, so the CVaR adversary just fills the highest
    losses first, each capped at ``1 / (n * (1 - alpha))``.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("CVaR alpha must be in [0, 1)")
    losses = np.asarray(losses, dtype=float).reshape(-1)
    count = losses.shape[0]
    if M.shape[0] != count:
        raise ValueError("CVaR losses must match allocation matrix rows")
    if not np.isfinite(losses).all():
        raise ValueError("CVaR losses must be finite")

    cap = 1.0 / (count * (1.0 - alpha))
    adversary = np.zeros(count, dtype=float)
    remaining = 1.0
    for idx in np.argsort(losses)[::-1]:
        adversary[idx] = min(cap, remaining)
        remaining -= adversary[idx]
        if remaining <= 1e-12:
            break
    adversary /= float(adversary.sum())
    return adversary @ M, float(adversary @ losses)


def _solve_cvar_monotone_allocation(
    M: np.ndarray,
    *,
    cost_w: np.ndarray,
    alpha: float = _MONOTONE_CVAR95_ALPHA,
    relative_tol: float = 1e-5,
    max_iters: int = 1000,
):
    M = np.asarray(M, dtype=float)
    num_points, num_levels = M.shape
    if num_points == 0 or num_levels == 0:
        raise ValueError("M must be non-empty")
    weighted_M = M * cost_w[None, :]
    row_scores = np.sqrt(np.maximum(weighted_M, 0.0)).sum(axis=1)
    profile, _ = _cvar_profile(M, row_scores, alpha=alpha)

    best_y = np.asarray(cost_w, dtype=float).copy()
    best_upper = math.inf
    for iter_idx in range(max_iters):
        dual_lower, y = _monotone_dual_value_and_simplex(profile, cost_w)
        full_values = _objective_values(weighted_M, y)
        target_profile, full_upper = _cvar_profile(M, full_values, alpha=alpha)
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
                _, candidate_upper = _cvar_profile(M, candidate_values, alpha=alpha)
                if candidate_upper < best_upper:
                    best_upper = candidate_upper
                    best_y = candidate_y.copy()

    return best_y


def _query_variance_matrix(variance2_per_level, tau2) -> np.ndarray:
    """``M[q, i]``: per-level variance contribution of query ``q``."""
    M = np.maximum(np.asarray(variance2_per_level, dtype=float), 0.0).T.copy()
    M[:, 0] += np.maximum(np.asarray(tau2, dtype=float), 0.0)
    return M


def _solve_optimal_split_factors(
    variance2_per_level: np.ndarray,
    tau2: np.ndarray,
    cost_weights: Sequence[float],
    optimization_mode: str = "monotone",
    c_max: float | None = None,
) -> List[float]:
    """Cost-optimal monotone split factors from the per-level variance grids.

    `monotone` minimises the worst query's `sum_l c_l M[q,l] / y_l`;
    `monotone_cvar95` minimises the 95% CVaR over queries instead;
    `learned_c` minimises the same worst-query objective restricted to one
    branching factor repeated at every split, which is convex in `log c` (see
    `uniform_c`).  That mode requires `c_max`, which `_solve_phase1_allocation`
    derives from the phase-2 budget.
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

    M = _query_variance_matrix(variance2_per_level, tau2)
    if optimization_mode == "learned_c":
        if c_max is None:
            raise ValueError("learned_c requires c_max")
        factor = float(solve_uniform_c(M, cost_w, c_min=1.0, c_max=float(c_max))["c"])
        return [factor] * (num_levels - 1)
    if optimization_mode == "monotone":
        simplex = _solve_frank_wolfe_monotone_allocation(M, cost_w=cost_w)
    else:
        simplex = _solve_cvar_monotone_allocation(
            M, cost_w=cost_w, alpha=_MONOTONE_CVAR95_ALPHA
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
        """Inputs for one (level, path, query) triple, built on demand.

        The raw state is deliberately absent: the network sees only how far the
        state sits from the query corner in the queried coordinates, plus the
        query's size, its mass and the level time.

        The margins are gathered by flat index rather than by first slicing out
        `states[level, path]`. That intermediate would be (batch, state_dim)
        wide just to read `max_active` columns from it -- for a latent diffusion
        state it is hundreds of megabytes per batch, and for a target-space one
        far more. Materialising every triple up front costs
        `levels * paths * queries * max_active` floats, which is tens of
        gigabytes once the split schedule is long, so each batch is built as it
        is consumed instead.
        """
        level_path_idx = torch.div(pair_idx, num_queries, rounding_mode="floor")
        query_idx = pair_idx - level_path_idx * num_queries
        level_idx = torch.div(level_path_idx, paths_per_level, rounding_mode="floor")
        state_dim = int(states.shape[-1])
        flat_dims = level_path_idx.unsqueeze(1) * state_dim + active_dims[query_idx]
        gathered = states.reshape(-1)[flat_dims]
        margin = (gathered - t_norm[query_idx]) * active_mask[query_idx]
        return torch.cat(
            [q_features[query_idx], margin, time_features[level_idx]], dim=1
        )

    input_dim = int(q_features.shape[1]) + max_active + 1

    train_targets = y_train.unsqueeze(0).expand(num_levels, -1, -1).reshape(-1, 1)

    # Module parameters are initialized on CPU. Serialize that short section so
    # concurrent fits cannot race through PyTorch's process-global CPU RNG; no
    # CUDA RNG snapshot is needed before moving the initialized model.
    with _CROSSFIT_Q_MLP_INIT_LOCK:
        if seed is None:
            model = _crossfit_q_build_mlp(input_dim, 1, params)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                model = _crossfit_q_build_mlp(input_dim, 1, params)
    model = model.to(device=device)
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
            xb = triple_features(x_train, idx, train_paths)
            yb = train_targets[idx]
            logits = model(xb)
            loss = supervised_loss(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    del x_train, train_targets
    model.eval()
    test_paths = int(x_test.shape[1])
    test_pair_count = int(num_levels) * int(test_paths) * int(num_queries)
    predictions_flat = np.empty(test_pair_count, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, test_pair_count, batch_size):
            end = min(start + batch_size, test_pair_count)
            pair_idx = torch.arange(start, end, device=device, dtype=torch.long)
            predictions_flat[start:end] = (
                torch.sigmoid(model(triple_features(x_test, pair_idx, test_paths)))
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
    return predictions_flat.reshape(num_levels, test_paths, num_queries)


def _crossfit_q_fold_indices(num_paths: int, n_folds: int):
    num_paths = int(num_paths)
    n_folds = int(n_folds)
    if n_folds < 1:
        raise ValueError("crossfit_q_folds must be at least 1")
    if n_folds > num_paths:
        raise ValueError(
            f"crossfit_q_folds={n_folds} exceeds the {num_paths} pilot paths"
        )
    if n_folds == 1:
        all_idx = np.arange(num_paths, dtype=np.int64)
        return [(all_idx, all_idx)]
    fold_ids = np.arange(num_paths, dtype=np.int64) % n_folds
    return [
        (np.flatnonzero(fold_ids != fold_idx), np.flatnonzero(fold_ids == fold_idx))
        for fold_idx in range(n_folds)
    ]


def _crossfit_q_predict_all_levels(
    states_by_level: np.ndarray,
    labels: np.ndarray,
    fold_indices,
    *,
    query_conditioning,
    level_times: np.ndarray,
    mlp_params: Mapping[str, Any] | None,
    mlp_device: str | None,
    seed: int | None,
    manage_mlp_num_threads: bool = True,
):
    """Fit one MLP per fold and assemble predictions in pilot-path order.

    With one fold, the sole training and test sets both contain every path,
    exactly reproducing the in-sample estimator. With multiple folds, every
    path is predicted by a model whose training set excluded that path.
    """
    num_levels, num_paths, obs_dim = (
        int(states_by_level.shape[0]),
        int(labels.shape[0]),
        int(labels.shape[1]),
    )
    predictions_by_level = np.empty((num_levels, num_paths, obs_dim), dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        train_labels = labels[train_idx]
        fold_seed = None if seed is None else int(seed) + int(fold_idx)
        predictions = _crossfit_q_mlp_predict_all_levels(
            states_by_level[:, train_idx, :],
            train_labels,
            states_by_level[:, test_idx, :],
            query_conditioning,
            level_times,
            params=mlp_params,
            runner_device=mlp_device,
            seed=fold_seed,
            manage_num_threads=manage_mlp_num_threads,
        )
        finite = np.isfinite(predictions)
        if not finite.all():
            fallback = np.broadcast_to(train_labels.mean(axis=0), predictions.shape)
            predictions = np.where(finite, predictions, fallback)
        predictions_by_level[:, test_idx, :] = np.clip(predictions, 0.0, 1.0)
    return predictions_by_level


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
    n_folds: int = CROSSFIT_Q_DEFAULT_FOLDS,
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

    fold_indices = _crossfit_q_fold_indices(num_paths, int(n_folds))
    mlp_params_normalized = _crossfit_q_normalize_mlp_params(mlp_params)

    F_hat = labels.mean(axis=0)
    Q_raw = np.empty((num_levels + 1, obs_dim), dtype=float)
    predictions_by_level = _crossfit_q_predict_all_levels(
        states_by_level,
        labels,
        fold_indices,
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

    return merged


def _mass_grid(count: int, mass_min: float, mass_max: float):
    if int(count) < 1:
        return np.empty(0, dtype=float)
    return mass_min + (np.arange(int(count), dtype=float) + 0.5) / float(count) * (
        mass_max - mass_min
    )


def _draw_grid_free_query_design(dim, query_params, *, seed, run_index):
    """Draw a query design without waiting for the pilot simulation."""
    query_params = _normalize_query_params(query_params)
    dim = int(dim)
    total = int(query_params["num_queries"])
    k_max = min(dim, int(query_params["k_max"]))
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed or 0), int(run_index), dim])
    )
    sizes = rng.integers(1, k_max + 1, size=total).astype(np.int64)
    masses = _mass_grid(
        total, float(query_params["mass_min"]), float(query_params["mass_max"])
    )
    width = int(sizes.max())
    active = np.arange(width)[None, :] < sizes[:, None]

    keys = rng.random((total, dim))
    active_dims = np.full((total, width), -1, dtype=np.int64)
    if dim == 2 and width == 2:
        first = (keys[:, 1] < keys[:, 0]).astype(np.int64)
        selected = np.stack((first, 1 - first), axis=1)
        selected = np.where(active, selected, dim)
        selected.sort(axis=1)
        active_dims[:] = np.where(active, selected, -1)
    elif width < dim:
        for size in np.unique(sizes):
            rows = np.flatnonzero(sizes == int(size))
            selected = np.argpartition(keys[rows], int(size) - 1, axis=1)[
                :, : int(size)
            ]
            selected.sort(axis=1)
            active_dims[rows, : int(size)] = selected
    else:
        selected = np.argsort(keys, axis=1)[:, :width]
        selected = np.where(active, selected, dim)
        selected.sort(axis=1)
        active_dims[:] = np.where(active, selected, -1)

    exponentials = rng.exponential(size=(total, width)) * active
    weights = exponentials / exponentials.sum(axis=1, keepdims=True)
    ranks = np.full((total, width), np.nan, dtype=float)
    ranks[active] = np.exp(weights[active] * np.repeat(np.log(masses), sizes))
    return {
        "ambient_dim": dim,
        "active_dims": active_dims,
        "active_sizes": sizes,
        "target_mass": masses,
        "ranks": ranks,
    }


def _query_spec_from_design(design, thresholds):
    return {
        "kind": "sparse_subset_lower_orthant",
        **design,
        "thresholds": thresholds,
    }


def _finish_grid_free_query_design(samples, design):
    samples_np = coerce_samples_np(samples)
    if samples_np.shape[0] < 1:
        raise ValueError("phase 1 needs at least one terminal sample")
    if samples_np.shape[1] != int(design["ambient_dim"]):
        raise ValueError("sample and query-design dimensions differ")
    active_dims = np.asarray(design["active_dims"], dtype=np.int64)
    ranks = np.asarray(design["ranks"], dtype=float)
    active = active_dims >= 0
    rows, positions = np.nonzero(active)
    dims = active_dims[rows, positions]
    virtual = (int(samples_np.shape[0]) - 1) * ranks[rows, positions]
    lower = np.floor(virtual).astype(np.int64)
    upper = np.ceil(virtual).astype(np.int64)
    weight = virtual - lower
    ordered = np.sort(samples_np, axis=0)
    left = ordered[lower, dims]
    right = ordered[upper, dims]
    delta = right - left
    interpolated = np.where(
        weight < 0.5,
        left + delta * weight,
        right - delta * (1.0 - weight),
    )
    thresholds = np.full(active_dims.shape, np.nan, dtype=float)
    thresholds[rows, positions] = interpolated
    return _query_spec_from_design(design, thresholds)


def _finish_grid_free_query_designs_cuda(samples_by_run, designs):
    """Finish all exact marginal quantiles in one native-dtype GPU sort."""
    values = samples_by_run.reshape(len(designs), samples_by_run.shape[1], -1)
    if not values.is_cuda:
        raise ValueError("CUDA query completion requires CUDA samples")
    widths = [int(np.asarray(design["active_dims"]).shape[1]) for design in designs]
    width = max(widths)
    queries = int(np.asarray(designs[0]["active_dims"]).shape[0])
    dims_np = np.full((len(designs), queries, width), -1, dtype=np.int64)
    ranks_np = np.full((len(designs), queries, width), np.nan, dtype=float)
    for run, design in enumerate(designs):
        dims_np[run, :, : widths[run]] = design["active_dims"]
        ranks_np[run, :, : widths[run]] = design["ranks"]

    dims = torch.as_tensor(dims_np, device=values.device, dtype=torch.long)
    ranks = torch.as_tensor(ranks_np, device=values.device, dtype=torch.float64)
    active = dims >= 0
    safe_dims = dims.clamp_min(0)
    virtual = (int(values.shape[1]) - 1) * torch.where(
        active, ranks, torch.zeros_like(ranks)
    )
    lower = torch.floor(virtual).to(torch.long)
    upper = torch.ceil(virtual).to(torch.long)
    weight = virtual - lower
    ordered = torch.sort(values, dim=1).values
    run_ids = torch.arange(len(designs), device=values.device)[:, None, None]
    left = ordered[run_ids, lower, safe_dims].to(torch.float64)
    right = ordered[run_ids, upper, safe_dims].to(torch.float64)
    delta = right - left
    thresholds = torch.where(
        weight < 0.5,
        left + delta * weight,
        right - delta * (1.0 - weight),
    )
    thresholds = torch.where(active, thresholds, torch.full_like(thresholds, torch.nan))
    thresholds_np = thresholds.cpu().numpy()
    return [
        _query_spec_from_design(design, thresholds_np[run, :, : widths[run]])
        for run, design in enumerate(designs)
    ]


def _generate_grid_free_queries(samples, query_params, *, seed, run_index):
    """Sparse lower-orthant queries at pilot empirical marginal quantiles.

    For each query draw a size ``k`` uniform on {1..min(D, k_max)}, a random
    subset ``S`` of ``k`` coordinates, a mass ``m`` from an even grid over
    [mass_min, mass_max], and simplex weights ``w ~ Dirichlet(1_k)``. The corner
    sits at the ``m ** w_j`` quantile of coordinate ``S[j]``.

    Uniform ``k`` makes the design its own reference measure, so the allocator
    weights every query equally and nothing has to be reweighted afterwards.

    ``prod_j m**w_j == m`` exactly because ``sum_j w_j == 1``, and every rank
    lands in ``(m, 1)`` on its own -- so there is nothing to clip, rescale or
    validate. ``m`` is the orthant's mass only under independent coordinates; it
    is a spreading device that keeps queries non-degenerate at every ``k``, not
    a claim about the true mass.

    The design is drawn from the sweep ``seed`` and the run's global index, so
    runs never share a design and re-running with a new seed draws new ones --
    the estimator is never conditioned on one arbitrary query design.
    """
    values = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
    dim = int(values.reshape(values.shape[0], -1).shape[1])
    design = _draw_grid_free_query_design(
        dim, query_params, seed=seed, run_index=run_index
    )
    if values.is_cuda:
        return _finish_grid_free_query_designs_cuda(
            values.reshape(1, values.shape[0], -1), [design]
        )[0]
    return _finish_grid_free_query_design(samples, design)


if triton is not None:

    @triton.jit
    def _lower_orthant_labels_kernel(
        values_ptr,
        dims_ptr,
        thresholds_ptr,
        output_ptr,
        pair_count: tl.constexpr,
        paths: tl.constexpr,
        dimension: tl.constexpr,
        queries: tl.constexpr,
        width: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        pairs = tl.program_id(0) * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
        pair_mask = pairs < pair_count
        query = pairs % queries
        remainder = pairs // queries
        path = remainder % paths
        run = remainder // paths
        positions = tl.arange(0, BLOCK_WIDTH)
        position_mask = positions[None, :] < width
        spec_offsets = (
            (run[:, None] * queries + query[:, None]) * width + positions[None, :]
        )
        load_mask = pair_mask[:, None] & position_mask
        dims = tl.load(dims_ptr + spec_offsets, mask=load_mask, other=-1)
        active = (dims >= 0) & position_mask
        sample_offsets = (
            (run[:, None] * paths + path[:, None]) * dimension
            + tl.maximum(dims, 0)
        )
        values = tl.load(
            values_ptr + sample_offsets, mask=load_mask & active, other=0.0
        )
        thresholds = tl.load(
            thresholds_ptr + spec_offsets, mask=load_mask & active, other=0.0
        )
        failures = active & (values > thresholds)
        matches = tl.sum(failures.to(tl.int32), axis=1) == 0
        tl.store(output_ptr + pairs, matches, mask=pair_mask)

else:
    _lower_orthant_labels_kernel = None


def _stack_query_specs(query_points, *, device, dtype):
    queries = int(np.asarray(query_points[0]["active_dims"]).shape[0])
    width = max(np.asarray(spec["active_dims"]).shape[1] for spec in query_points)
    dims = np.full((len(query_points), queries, width), -1, dtype=np.int32)
    thresholds = np.full((len(query_points), queries, width), np.nan, dtype=float)
    for run, spec in enumerate(query_points):
        active_dims = np.asarray(spec["active_dims"], dtype=np.int32)
        query_thresholds = np.asarray(spec["thresholds"], dtype=float)
        if active_dims.shape != query_thresholds.shape:
            raise ValueError("query active_dims and thresholds must match")
        if active_dims.shape[0] != queries:
            raise ValueError("all runs must use the same number of queries")
        dims[run, :, : active_dims.shape[1]] = active_dims
        thresholds[run, :, : active_dims.shape[1]] = query_thresholds
    return (
        torch.as_tensor(dims, device=device, dtype=torch.int32).contiguous(),
        torch.as_tensor(thresholds, device=device, dtype=dtype).contiguous(),
    )


def _lower_orthant_labels_batched(samples_by_run, query_points):
    values = samples_by_run.reshape(
        len(query_points), samples_by_run.shape[1], -1
    ).contiguous()
    dimension = int(values.shape[2])
    for spec in query_points:
        if not isinstance(spec, Mapping) or spec.get("kind") != "sparse_subset_lower_orthant":
            raise ValueError("grid-free phase 1 requires sparse lower-orthant specs")
        if int(spec.get("ambient_dim", dimension)) != dimension:
            raise ValueError("query ambient dimension does not match samples")
        if np.any(np.asarray(spec["active_dims"]) >= dimension):
            raise ValueError("query active dimension exceeds sample dimension")
    dims, thresholds = _stack_query_specs(
        query_points, device=values.device, dtype=values.dtype
    )
    runs, paths, _ = values.shape
    queries, width = int(dims.shape[1]), int(dims.shape[2])
    output = torch.empty((runs, paths, queries), device=values.device, dtype=torch.bool)

    if values.is_cuda and _lower_orthant_labels_kernel is not None:
        block_width = triton.next_power_of_2(width)
        block_pairs = 1024 if width <= 4 and paths >= 1000 else (512 if width <= 4 else 64)
        block_pairs = max(
            1, min(block_pairs, _QUERY_LABEL_MAX_TILE_ELEMENTS // block_width)
        )
        pair_count = int(output.numel())
        _lower_orthant_labels_kernel[(triton.cdiv(pair_count, block_pairs),)](
            values,
            dims,
            thresholds,
            output,
            pair_count=pair_count,
            paths=paths,
            dimension=dimension,
            queries=queries,
            width=width,
            BLOCK_PAIRS=block_pairs,
            BLOCK_WIDTH=block_width,
            num_warps=4,
        )
        return output

    active = dims >= 0
    safe_dims = dims.clamp_min(0).to(torch.long)
    block = max(
        1,
        min(
            queries,
            _QUERY_LABEL_MAX_TEMPORARY_BYTES
            // max(1, runs * paths * width * values.element_size() * 2),
        ),
    )
    for start in range(0, queries, block):
        end = min(queries, start + block)
        source = values[:, :, None, :].expand(-1, -1, end - start, -1)
        indices = safe_dims[:, None, start:end, :].expand(-1, paths, -1, -1)
        comparisons = torch.gather(source, 3, indices) <= thresholds[
            :, None, start:end, :
        ]
        comparisons |= ~active[:, None, start:end, :]
        output[:, :, start:end] = comparisons.all(dim=-1)
    return output


def _lower_orthant_labels(samples, query_points):
    values = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
    values = values.reshape(1, values.shape[0], -1)
    return _lower_orthant_labels_batched(values, [query_points])[0]


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
    return states, x0, x.detach().reshape(total_paths, -1)


def _grid_free_query_payload(samples, query_params, *, seed, run_index):
    query_spec = _generate_grid_free_queries(
        samples, query_params, seed=seed, run_index=run_index
    )
    labels = _lower_orthant_labels(samples, query_spec)
    return query_spec, labels


def _estimate_crossfit_q_variance_for_run(
    run_idx: int,
    states_by_level: np.ndarray,
    spec: Mapping[str, Any],
    *,
    level_times,
    n_folds: int,
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
        n_folds=int(n_folds),
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
    crossfit_q_folds=CROSSFIT_Q_DEFAULT_FOLDS,
    crossfit_q_mlp_run_parallelism=CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM,
    crossfit_q_mlp_params=None,
    query_params=None,
    seed=None,
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
    effective_folds = int(crossfit_q_folds)
    if effective_folds < 1:
        raise ValueError("crossfit_q_folds must be at least 1")
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
    if effective_folds > paths_per_run:
        raise ValueError(
            f"crossfit_q_folds={effective_folds} exceeds the {paths_per_run} "
            "pilot paths available per run"
        )
    used_B1 = int(paths_per_run * full_path_cost)
    _debug(
        debug,
        f"Phase 1 crossfit_q: chunk={chunk_size} paths_per_run={paths_per_run} "
        f"folds={effective_folds} "
        f"mlp_workers={min(mlp_run_parallelism, int(chunk_size))} "
        f"mlp_device={mlp_device} mlp_num_threads={mlp_params_normalized.get('num_threads')}",
    )

    run_indices = [
        int(run_index_offset) + run_idx for run_idx in range(int(chunk_size))
    ]

    def draw_designs():
        started = time.perf_counter()
        designs = [
            _draw_grid_free_query_design(
                runner.input_dim,
                query_params,
                seed=seed,
                run_index=run_index,
            )
            for run_index in run_indices
        ]
        return designs, time.perf_counter() - started

    # Query designs do not depend on pilot samples, so hide their CPU work
    # behind the Phase-1 simulation without introducing another user option.
    with ThreadPoolExecutor(max_workers=1) as query_executor:
        design_future = query_executor.submit(draw_designs)
        _synchronize_runner_device(runner)
        simulation_started = time.perf_counter()
        states_by_run, x0, terminal_states = _simulate_crossfit_q_trajectories(
            runner,
            chunk_size=chunk_size,
            paths_per_run=paths_per_run,
            split_points=split_points,
            max_sampling_batch_size=max_sampling_batch_size,
            generator=generator,
        )
        _synchronize_runner_device(runner)
        simulation_seconds = time.perf_counter() - simulation_started
        designs, design_seconds = design_future.result()
    phase1_x0_by_run: List[Any] = [None] * chunk_size
    if reuse_phase1_samples:
        x0_np = coerce_samples_np(x0).reshape(int(chunk_size), int(paths_per_run), -1)
        phase1_x0_by_run = [x0_np[idx] for idx in range(int(chunk_size))]

    # Queries are drawn at runner.input_dim, so they index the model state. A
    # runner whose postprocessing changes dimension -- a latent diffusion model
    # decoding to pixels -- must label on the model-space terminal state, or the
    # query coordinates address a different space from the state the regressor
    # conditions on. x0 stays decoded for the reused phase-1 samples and the metric.
    query_samples = (
        terminal_states
        if str(getattr(runner, "phase1_query_space", "target")) == "model"
        else x0
    )
    query_by_run = list(torch.split(query_samples, int(paths_per_run)))

    _synchronize_runner_device(runner)
    query_started = time.perf_counter()
    if query_samples.is_cuda:
        samples_by_run = query_samples.reshape(int(chunk_size), int(paths_per_run), -1)
        query_specs = _finish_grid_free_query_designs_cuda(samples_by_run, designs)
        labels_by_run = _lower_orthant_labels_batched(samples_by_run, query_specs)
        run_specs = [
            {
                "query_spec": query_specs[run_idx],
                "labels": labels_by_run[run_idx]
                .cpu()
                .numpy()
                .astype(np.bool_, copy=False),
            }
            for run_idx in range(int(chunk_size))
        ]
    else:
        def finish_run(run_idx):
            query_spec = _finish_grid_free_query_design(
                query_by_run[run_idx], designs[run_idx]
            )
            labels = _lower_orthant_labels(query_by_run[run_idx], query_spec)
            return {
                "query_spec": query_spec,
                "labels": labels.numpy().astype(np.bool_, copy=False),
            }

        query_workers = min(int(chunk_size), 4)
        with ThreadPoolExecutor(max_workers=query_workers) as query_executor:
            run_specs = list(query_executor.map(finish_run, range(int(chunk_size))))
    _synchronize_runner_device(runner)
    query_seconds = design_seconds + (time.perf_counter() - query_started)

    estimates_by_run: List[Any] = [None] * int(chunk_size)
    estimation_seconds_by_run = np.zeros(int(chunk_size), dtype=float)
    effective_mlp_workers = min(int(mlp_run_parallelism), int(chunk_size))
    if effective_mlp_workers <= 1:
        for run_idx, spec in enumerate(run_specs):
            _synchronize_runner_device(runner)
            estimation_started = time.perf_counter()
            _, estimates_by_run[run_idx] = _estimate_crossfit_q_variance_for_run(
                run_idx,
                states_by_run[:, run_idx, :, :],
                spec,
                level_times=level_times,
                n_folds=effective_folds,
                mlp_params=mlp_params_normalized,
                mlp_device=str(runner.device),
                seed=7919 + int(run_idx),
                manage_mlp_num_threads=True,
            )
            _synchronize_runner_device(runner)
            estimation_seconds_by_run[run_idx] = (
                time.perf_counter() - estimation_started
            )
    else:
        _debug(
            debug,
            "Phase 1 crossfit_q MLP run parallelism enabled: "
            f"workers={effective_mlp_workers} chunk={chunk_size}",
        )
        _synchronize_runner_device(runner)
        estimation_started = time.perf_counter()
        with _TorchNumThreadsContext(
            mlp_params_normalized.get("num_threads"), mlp_device.type == "cpu"
        ):
            with ThreadPoolExecutor(max_workers=effective_mlp_workers) as mlp_executor:
                futures = [
                    mlp_executor.submit(
                        _estimate_crossfit_q_variance_for_run,
                        run_idx,
                        states_by_run[:, run_idx, :, :],
                        spec,
                        level_times=level_times,
                        n_folds=effective_folds,
                        mlp_params=mlp_params_normalized,
                        mlp_device=str(runner.device),
                        seed=7919 + int(run_idx),
                        manage_mlp_num_threads=False,
                    )
                    for run_idx, spec in enumerate(run_specs)
                ]
                for run_idx, future in enumerate(futures):
                    _, estimates_by_run[run_idx] = future.result()
        _synchronize_runner_device(runner)
        amortized_seconds = (time.perf_counter() - estimation_started) / int(chunk_size)
        estimation_seconds_by_run.fill(amortized_seconds)

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
                "phase1_simulation_seconds": simulation_seconds / int(chunk_size),
                "query_seconds": query_seconds / int(chunk_size),
                "estimation_seconds": float(estimation_seconds_by_run[run_idx]),
            }
        )
    return payloads


# --- Phase 1 → Phase 2 ------------------------------------------------------


def _solve_phase1_allocation(
    payload,
    runner,
    *,
    B,
    optimization_mode="monotone",
):
    optimization_started = time.perf_counter()
    split_points = payload["split_points"]
    seg_costs = runner.segment_costs(split_points)
    total = float(np.sum(seg_costs))
    if total <= 0.0:
        raise ValueError("segment costs must sum to a positive value")
    cost_weights = [c / total for c in seg_costs]

    B2 = B - int(payload["used_B1"])
    # One branching factor grows the path count like c^L, so the search needs a
    # budget-aware ceiling; the free allocation is bounded by the monotone
    # simplex instead and needs none.
    c_max = (
        max_feasible_c(runner, split_points, budget=B2)
        if optimization_mode == "learned_c"
        else None
    )
    split_factors = _solve_optimal_split_factors(
        payload["variance2_per_level"],
        payload["tau2"],
        cost_weights,
        optimization_mode=optimization_mode,
        c_max=c_max,
    )
    # Every learned mode has phase-1 variances, so the mixture weights come from
    # the finite-query minimax LP.
    variances = _query_variance_matrix(
        payload["variance2_per_level"], payload["tau2"]
    )
    try:
        design = design_mixture(
            split_factors, seg_costs, B2, variances=variances
        )
    except InsufficientTreeBudgetError as exc:
        raise InsufficientSplitBudgetError(str(exc)) from exc
    result = {
        "split_points": split_points,
        "design": design,
        "N_i": mean_split_factors(design),
        "n0": int(sum(tree.roots for tree in design)),
        "used_B1": int(payload["used_B1"]),
        "phase1_x0_samples": payload.get("phase1_x0_samples"),
        "phase1_simulation_seconds": float(payload["phase1_simulation_seconds"]),
        "query_seconds": float(payload["query_seconds"]),
        "estimation_seconds": float(payload["estimation_seconds"]),
    }
    result["optimization_seconds"] = time.perf_counter() - optimization_started
    return result


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
    crossfit_q_folds: int = CROSSFIT_Q_DEFAULT_FOLDS,
    crossfit_q_mlp_run_parallelism: int = CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM,
    crossfit_q_mlp_params: Mapping[str, Any] | None = None,
    query_params: Mapping[str, Any] | None = None,
    optimization_mode: str = "monotone",
    debug: bool = False,
    n_parallel: int = 1,
    run_offset: int = 0,
    return_trial_results: bool = False,
    max_sampling_batch_size=None,
    max_paths_in_flight=None,
    phase1: Mapping[str, Any] | None = None,
):
    """``phase1`` selects the MMD sibling-pilot Phase 1, which reads
    ``comparison_state["mmd_design"]``; without it the crossfit query MLP runs."""
    if not 0 <= B1 < B:
        raise ValueError("require 0 <= B1 < B")
    if int(crossfit_q_folds) < 1:
        raise ValueError("crossfit_q_folds must be at least 1")
    if optimization_mode not in SUPPORTED_OPTIMIZATION_MODES:
        raise ValueError(f"unknown optimization_mode '{optimization_mode}'")

    phase1_extra = {
        "query_params": dict(_normalize_query_params(query_params)),
        "crossfit_q_folds": int(crossfit_q_folds),
        "crossfit_q_mlp_run_parallelism": int(
            _crossfit_q_normalize_mlp_run_parallelism(
                crossfit_q_mlp_run_parallelism
            )
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

    trial_results: List[Any] = []
    workers = max(1, min(n_runs, n_parallel))
    additional_attempts = 0
    discarded_attempts = 0
    first_round = True
    last_budget_error: InsufficientSplitBudgetError | None = None

    with ThreadPoolExecutor(max_workers=workers) as alloc_executor:
        while len(trial_results) < n_runs:
            missing = n_runs - len(trial_results)
            if first_round:
                attempt_count = missing
                attempt_seed = seed
                attempt_offset = run_offset
                progress_label = "Phase 1"
                first_round = False
            else:
                remaining_retry_budget = n_runs - additional_attempts
                if remaining_retry_budget <= 0:
                    raise RuntimeError(
                        f"Could not produce {n_runs} feasible allocations after "
                        f"{n_runs + additional_attempts} attempts; discarded "
                        f"{discarded_attempts} allocations"
                    ) from last_budget_error
                attempt_count = min(missing, remaining_retry_budget)
                attempt_seed = (
                    None
                    if seed is None
                    else seed + _INFEASIBLE_ALLOCATION_RETRY_SEED_OFFSET
                )
                attempt_offset = run_offset + additional_attempts
                additional_attempts += attempt_count
                progress_label = "Phase 1 replacements"

            allocation_futures = []
            attempt_chunks = list(iter_run_chunks(attempt_count, n_parallel))
            for start, end in tqdm(
                attempt_chunks, desc=progress_label, leave=False
            ):
                chunk_size = end - start
                run_seed = (
                    None
                    if attempt_seed is None
                    else attempt_seed + attempt_offset + start
                )
                generator = make_torch_generator(run_seed, runner.device)
                if phase1 is not None:
                    from phase1_mmd import run_mmd_sibling_phase1_batch

                    payloads = run_mmd_sibling_phase1_batch(
                        runner,
                        B=B,
                        B1=B1,
                        split_percentages=split_percentages,
                        phase1=phase1,
                        design_state=comparison_state["mmd_design"],
                        optimization_mode=optimization_mode,
                        reuse_phase1_samples=reuse_phase1_samples,
                        chunk_size=chunk_size,
                        max_sampling_batch_size=max_sampling_batch_size,
                        generator=generator,
                    )
                else:
                    payloads = _run_crossfit_q_phase1_sampling_batch(
                        runner,
                        B1=B1,
                        seed=attempt_seed,
                        run_index_offset=attempt_offset + start,
                        split_percentages=split_percentages,
                        reuse_phase1_samples=reuse_phase1_samples,
                        chunk_size=chunk_size,
                        debug=debug,
                        max_sampling_batch_size=max_sampling_batch_size,
                        generator=generator,
                        **phase1_extra,
                    )
                for payload in payloads:
                    allocation_futures.append(
                        alloc_executor.submit(
                            _solve_phase1_allocation,
                            payload,
                            runner,
                            B=B,
                            optimization_mode=optimization_mode,
                        )
                    )

            for future in tqdm(
                allocation_futures, desc="Optimize N_i", leave=False
            ):
                try:
                    trial_results.append(future.result())
                except InsufficientSplitBudgetError as exc:
                    discarded_attempts += 1
                    last_budget_error = exc

    if discarded_attempts:
        log.warning(
            "Discarded %d infeasible allocation attempts at B=%d, B1=%d; "
            "generated replacements",
            discarded_attempts,
            B,
            B1,
        )

    chunks = list(iter_run_chunks(n_runs, n_parallel))
    metric_workers = max(1, min(int(n_parallel), n_runs))
    metric_futures: List[Tuple[int, Any]] = []
    with ThreadPoolExecutor(max_workers=metric_workers) as metric_executor:
        for start, end in tqdm(chunks, desc="Phase 2", leave=False):
            chunk = trial_results[start:end]
            run_seed = (
                None if seed is None else seed + PHASE2_SEED_OFFSET + run_offset + start
            )
            _synchronize_runner_device(runner)
            phase2_started = time.perf_counter()
            samples_by_run = run_mixture_batch(
                runner,
                designs_by_run=[t["design"] for t in chunk],
                split_points=chunk[0]["split_points"],
                generator=make_torch_generator(run_seed, runner.device),
                max_sampling_batch_size=max_sampling_batch_size,
                max_paths_in_flight=max_paths_in_flight,
            )
            _synchronize_runner_device(runner)
            phase2_seconds = (time.perf_counter() - phase2_started) / len(chunk)
            for trial in chunk:
                trial["phase2_seconds"] = float(phase2_seconds)
                trial["total_seconds"] = float(
                    trial["phase1_simulation_seconds"]
                    + trial["query_seconds"]
                    + trial["estimation_seconds"]
                    + trial["optimization_seconds"]
                    + trial["phase2_seconds"]
                )
            parts_by_run = []
            weights_by_run = []
            for trial, parts in zip(chunk, samples_by_run):
                design_weights = [tree.weight for tree in trial["design"]]
                pilots = trial.get("phase1_x0_samples")
                if pilots is None or len(pilots) == 0:
                    parts_by_run.append(list(parts))
                    weights_by_run.append(design_weights)
                    continue
                # Reused pilots are the all-ones tree; they carry their realized
                # budget share B1 / B, and the designed types share the rest.
                pilot_weight = float(trial["used_B1"]) / float(B)
                parts_by_run.append([pilots, *parts])
                weights_by_run.append(
                    [pilot_weight]
                    + [(1.0 - pilot_weight) * w for w in design_weights]
                )
            submit_sampling_trial_result_futures(
                executor=metric_executor,
                futures=metric_futures,
                run_indices=range(start, end),
                samples_by_run=parts_by_run,
                runner=runner,
                comparison_mode=comparison_mode,
                metric_states=comparison_state,
                metrics=metrics,
                weights_by_run=weights_by_run,
            )
        for run_idx, future in tqdm(metric_futures, desc="Metrics", leave=False):
            trial_results[run_idx].update(future.result())

    result = {
        "mode": "estimate_and_sample",
        "B": int(B),
        "B1": int(B1),
        **summarize_sampling_trials(trial_results, metrics),
    }
    for trial in trial_results:
        trial.pop("design", None)
    if return_trial_results:
        result["trial_results"] = trial_results
    return result
