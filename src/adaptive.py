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
    balanced_branch_counts_by_group,
    floor_split_total_count,
    repeat_by_counts,
)
from ks import coerce_samples_np
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
CROSSFIT_Q_DEFAULT_MLP_EPOCHS = 10
CROSSFIT_Q_DEFAULT_MLP_BATCH_SIZE = 16392
CROSSFIT_Q_DEFAULT_MLP_LR = 1e-3
CROSSFIT_Q_DEFAULT_MLP_WEIGHT_DECAY = 3e-4
CROSSFIT_Q_DEFAULT_MLP_LOSS = "bce"
CROSSFIT_Q_DEFAULT_MLP_DEVICE = "runner"
CROSSFIT_Q_DEFAULT_MLP_NUM_THREADS = 2
CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM = 5
CROSSFIT_Q_DEFAULT_NUM_QUERIES = 1024
CROSSFIT_Q_DEFAULT_TAIL_EPS = 1e-3
CROSSFIT_Q_DEFAULT_SUBSET_SIZES = [8, 16, 32, 64]
CROSSFIT_Q_DEFAULT_MASS_MIN = 0.05
CROSSFIT_Q_DEFAULT_MASS_MAX = 0.95
CROSSFIT_Q_DEFAULT_RANK_SPREAD = 0.4
CROSSFIT_Q_DEFAULT_SUBSET_SEED = 0
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

_DEFAULT_GRID_FREE_PARAMS: Dict[str, Any] = {
    "num_queries": CROSSFIT_Q_DEFAULT_NUM_QUERIES,
    "tail_eps": CROSSFIT_Q_DEFAULT_TAIL_EPS,
}

_DEFAULT_PHASE1_QUERY_PARAMS: Dict[str, Any] = {
    "subset_sizes": list(CROSSFIT_Q_DEFAULT_SUBSET_SIZES),
    "mass_min": CROSSFIT_Q_DEFAULT_MASS_MIN,
    "mass_max": CROSSFIT_Q_DEFAULT_MASS_MAX,
    "rank_spread": CROSSFIT_Q_DEFAULT_RANK_SPREAD,
    "subset_seed": CROSSFIT_Q_DEFAULT_SUBSET_SEED,
    "mass_bins": CROSSFIT_Q_DEFAULT_MASS_BINS,
}

_ALLOCATION_VARIANCE_PRIOR_STRENGTH = 8.0
_ALLOCATION_SHARE_PRIOR_STRENGTH = 4.0
_CROSSFIT_Q_MLP_INIT_LOCK = Lock()
_MONOTONE_CVAR95_ALPHA = 0.95
SUPPORTED_OPTIMIZATION_MODES = {"monotone", "monotone_cvar95"}


def _debug(enabled: bool, message: str):
    if enabled:
        log.debug(message)


# --- Joint-tree shape derivation --------------------------------------------


def _joint_tree_floor_cost(
    runner,
    split_points: Sequence[Any],
    joint_roots: int,
    joint_m: float,
):
    start_points = [runner.start_time] + list(split_points)
    end_points = list(split_points) + [runner.end_time]
    current_count = int(joint_roots)
    cost = 0.0
    for start_t, end_t in zip(start_points, end_points):
        current_count = floor_split_total_count(current_count, joint_m)
        cost += current_count * int(runner.segment_cost(start_t, end_t))
    return cost


def _derive_joint_tree_shape(
    runner, split_points: Sequence[Any], B1: int, joint_m: float
):
    joint_m = float(joint_m)
    if joint_m <= 1.0 or not math.isfinite(joint_m):
        raise ValueError("joint_m must be finite and > 1")

    cost_for_one = _joint_tree_floor_cost(runner, split_points, 1, joint_m)
    if B1 < cost_for_one:
        raise ValueError(
            f"B1={B1} is too small for joint with joint_m={joint_m:g}; "
            f"need at least {cost_for_one:.6f}"
        )

    lower, upper = 1, 2
    while _joint_tree_floor_cost(runner, split_points, upper, joint_m) <= B1:
        lower = upper
        upper *= 2

    high = upper - 1
    while lower < high:
        mid = (lower + high + 1) // 2
        if _joint_tree_floor_cost(runner, split_points, mid, joint_m) <= B1:
            lower = mid
        else:
            high = mid - 1
    return int(lower), float(joint_m)


# --- Independent-mode shape derivation --------------------------------------


def _independent_variance_cost(
    runner, t_curr, t_next, outer_count, middle_count, inner_count
):
    return (
        outer_count * runner.segment_cost(runner.start_time, t_curr)
        + outer_count * middle_count * runner.segment_cost(t_curr, t_next)
        + outer_count
        * middle_count
        * inner_count
        * runner.segment_cost(t_next, runner.end_time)
    )


def _derive_independent_counts(
    runner,
    variance_times: Sequence[Any],
    B1: int,
    independent_n2: int,
) -> Tuple[float, List[Dict[str, int]], int]:
    if not variance_times:
        raise ValueError("variance_times must be non-empty")
    if independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")

    budget_per_variance = B1 / len(variance_times)
    if budget_per_variance <= 0:
        raise ValueError("B1 must be positive for independent variance estimation")

    counts_by_variance = []
    total_used_B1 = 0
    next_times = list(variance_times[1:]) + [runner.end_time]

    for t_curr, t_next in zip(variance_times, next_times):
        segment1 = runner.segment_cost(runner.start_time, t_curr)
        segment2 = runner.segment_cost(t_curr, t_next)
        segment3 = runner.segment_cost(t_next, runner.end_time)
        min_inner = 2 if segment3 > 0 else 1
        min_cost = (
            segment1 + independent_n2 * segment2 + independent_n2 * min_inner * segment3
        )
        if budget_per_variance < min_cost:
            raise ValueError(
                f"B1 too small for equal per-variance independent estimation; "
                f"budget_per_variance={budget_per_variance:.6f}, min required={min_cost:.6f} "
                f"at t={t_curr}"
            )

        def cost_for_y(y):
            return (y * y) * (segment1 + independent_n2 * segment2) + (y**3) * (
                independent_n2 * segment3
            )

        lower, upper = 1.0, 2.0
        while cost_for_y(upper) <= budget_per_variance:
            upper *= 2.0
        for _ in range(80):
            mid = 0.5 * (lower + upper)
            if cost_for_y(mid) <= budget_per_variance:
                lower = mid
            else:
                upper = mid

        outer_count = max(1, int(lower * lower))
        if segment3 == 0:
            inner_count = 1
        else:
            inner_count = max(2, int(lower))
        used_budget = _independent_variance_cost(
            runner, t_curr, t_next, outer_count, independent_n2, inner_count
        )
        counts_by_variance.append(
            {
                "outer_count": int(outer_count),
                "middle_count": int(independent_n2),
                "inner_count": int(inner_count),
            }
        )
        total_used_B1 += used_budget

    return budget_per_variance, counts_by_variance, int(total_used_B1)


# --- Joint tree construction + variance estimation --------------------------


def _build_joint_tree_batch(
    runner,
    chunk_size: int,
    joint_roots: int,
    split_points: Sequence[Any],
    joint_m: float,
    generator=None,
):
    x = runner.sample_prior(chunk_size * joint_roots, generator=generator)
    run_ids = torch.repeat_interleave(
        torch.arange(chunk_size, device=runner.device, dtype=torch.long), joint_roots
    )

    child_counts_by_level: List[torch.Tensor] = []
    parent_run_ids_by_level: List[torch.Tensor] = []
    used_B1_by_run = torch.zeros(chunk_size, device=runner.device, dtype=torch.long)

    start_points = [runner.start_time] + list(split_points)
    end_points = list(split_points) + [runner.end_time]
    for start_t, end_t in zip(start_points, end_points):
        child_counts = balanced_branch_counts_by_group(
            run_ids,
            torch.full(
                (chunk_size,),
                float(joint_m),
                device=runner.device,
                dtype=x.dtype,
            ),
            num_groups=chunk_size,
            generator=generator,
        )
        child_counts_by_level.append(child_counts)
        parent_run_ids_by_level.append(run_ids)
        x = repeat_by_counts(x, child_counts)
        run_ids = run_ids.repeat_interleave(child_counts, dim=0)
        x = runner.sample_segment(x, start_t, end_t, generator=generator)
        used_B1_by_run = used_B1_by_run + torch.bincount(
            run_ids, minlength=chunk_size
        ) * int(runner.segment_cost(start_t, end_t))

    return (
        x,
        run_ids,
        child_counts_by_level,
        parent_run_ids_by_level,
        used_B1_by_run.cpu().tolist(),
    )


def _segment_mean_and_unbiased_var(values: torch.Tensor, counts: torch.Tensor):
    counts = counts.to(device=values.device, dtype=torch.long)
    means = torch.segment_reduce(values, reduce="mean", lengths=counts)
    sums = torch.segment_reduce(values, reduce="sum", lengths=counts)
    sums_sq = torch.segment_reduce(values * values, reduce="sum", lengths=counts)
    count_values = counts.to(dtype=values.dtype).unsqueeze(1)
    denom = torch.clamp(count_values - 1.0, min=1.0)
    variances = torch.clamp((sums_sq - (sums * sums) / count_values) / denom, min=0.0)
    return means, variances, counts >= 2


def _grouped_mean(values: torch.Tensor, group_ids: torch.Tensor, num_groups: int):
    group_ids = group_ids.to(device=values.device, dtype=torch.long)
    sums = torch.zeros(
        num_groups, values.shape[1], device=values.device, dtype=values.dtype
    )
    sums.index_add_(0, group_ids, values)
    counts = torch.bincount(group_ids, minlength=num_groups).to(
        device=values.device, dtype=values.dtype
    )
    return sums / counts.clamp_min(1.0).unsqueeze(1), counts


def _grouped_unbiased_var(values, group_ids, num_groups):
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
    safe = counts.clamp_min(1.0).unsqueeze(1)
    denom = torch.clamp(safe - 1.0, min=1.0)
    variances = (sums_sq - (sums * sums) / safe) / denom
    return torch.where(
        (counts >= 2).unsqueeze(1),
        torch.clamp(variances, min=0.0),
        torch.zeros_like(variances),
    )


def _estimate_variances_from_tree_batch(
    leaf_observables: torch.Tensor,
    num_levels: int,
    child_counts_by_level: Sequence[torch.Tensor],
    parent_run_ids_by_level: Sequence[torch.Tensor],
    chunk_size: int,
):
    """Return (variance2[chunk, num_levels, obs_dim], tau2[chunk, obs_dim])."""
    if (
        len(child_counts_by_level) != num_levels
        or len(parent_run_ids_by_level) != num_levels
    ):
        raise ValueError("level arrays must have length num_levels")

    current = leaf_observables
    obs_dim = current.shape[1]
    raw_var_by_level: List[torch.Tensor] = [None] * num_levels  # type: ignore
    valid_mask_by_level: List[torch.Tensor] = [None] * num_levels  # type: ignore

    for rev_idx in range(num_levels - 1, -1, -1):
        counts = child_counts_by_level[rev_idx].to(
            device=current.device, dtype=torch.long
        )
        child_means, child_vars, valid_mask = _segment_mean_and_unbiased_var(
            current, counts
        )
        raw_var_by_level[rev_idx] = child_vars
        valid_mask_by_level[rev_idx] = valid_mask
        current = child_means
    root_run_ids = parent_run_ids_by_level[0].to(
        device=current.device, dtype=torch.long
    )

    variance2 = torch.zeros(
        chunk_size, num_levels, obs_dim, device=current.device, dtype=current.dtype
    )
    for level_idx in range(num_levels):
        corrected = raw_var_by_level[level_idx]
        if level_idx + 1 < num_levels:
            lower_counts = child_counts_by_level[level_idx + 1].to(
                device=corrected.device, dtype=corrected.dtype
            )
            lower_noise_by_child = raw_var_by_level[
                level_idx + 1
            ] / lower_counts.unsqueeze(1)
            parent_counts = child_counts_by_level[level_idx].to(
                device=corrected.device, dtype=torch.long
            )
            corrected = corrected - torch.segment_reduce(
                lower_noise_by_child, reduce="mean", lengths=parent_counts
            )
        valid = valid_mask_by_level[level_idx]
        parents = parent_run_ids_by_level[level_idx].to(
            device=corrected.device, dtype=torch.long
        )
        variance2_flat, _ = _grouped_mean(corrected[valid], parents[valid], chunk_size)
        variance2[:, level_idx, :] = torch.clamp(variance2_flat, min=0.0)

    tau2 = _grouped_unbiased_var(current, root_run_ids, chunk_size)
    return variance2.cpu().numpy(), tau2.cpu().numpy()


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


def _prepare_allocation_matrix(
    variance2_per_level: np.ndarray,
    tau2: np.ndarray,
    *,
    query_metadata: Mapping[str, Any] | None,
):
    M = np.maximum(variance2_per_level, 0.0).T.copy()
    M[:, 0] += np.maximum(tau2, 0.0)
    num_points, num_levels = M.shape
    weights = _normalize_query_weights(
        None if query_metadata is None else query_metadata.get("weights"), num_points
    )
    if query_metadata is None:
        return M

    empirical_mass = np.asarray(
        query_metadata.get("empirical_mass"), dtype=float
    ).reshape(-1)
    target_mass = np.asarray(query_metadata.get("target_mass"), dtype=float).reshape(-1)
    if empirical_mass.shape != (num_points,) or target_mass.shape != (num_points,):
        raise ValueError("query mass metadata must match allocation matrix rows")
    n_paths = int(query_metadata.get("n_paths", 0))
    if n_paths < 1:
        raise ValueError("query_metadata.n_paths must be positive")

    empirical_mass = np.clip(empirical_mass, 0.0, 1.0)
    target_mass = np.clip(target_mass, 1e-12, 1.0 - 1e-12)
    prior = float(_ALLOCATION_VARIANCE_PRIOR_STRENGTH)
    alpha = n_paths * empirical_mass + prior * target_mass
    beta = n_paths * (1.0 - empirical_mass) + prior * (1.0 - target_mass)
    posterior_total = (alpha * beta) / ((alpha + beta) * (alpha + beta + 1.0))
    posterior_total = np.maximum(posterior_total, 0.0)

    row_totals = M.sum(axis=1)
    positive = row_totals > 0.0
    if bool(np.any(positive)):
        shares = np.zeros_like(M)
        shares[positive] = M[positive] / row_totals[positive, None]
        pooled = (weights[positive, None] * shares[positive]).sum(axis=0)
        pooled_total = float(pooled.sum())
        if pooled_total > 0.0:
            pooled = pooled / pooled_total
        else:
            pooled = np.full(num_levels, 1.0 / num_levels, dtype=float)
    else:
        shares = np.full_like(M, 1.0 / num_levels)
        pooled = np.full(num_levels, 1.0 / num_levels, dtype=float)
    shares[~positive] = pooled
    share_prior = float(_ALLOCATION_SHARE_PRIOR_STRENGTH)
    if share_prior > 0.0:
        data_weight = n_paths / (n_paths + share_prior)
        shares = data_weight * shares + (1.0 - data_weight) * pooled.reshape(1, -1)
    shares = np.maximum(shares, 0.0)
    share_totals = shares.sum(axis=1)
    shares = np.divide(
        shares,
        share_totals[:, None],
        out=np.full_like(shares, 1.0 / num_levels),
        where=share_totals[:, None] > 0.0,
    )
    return shares * posterior_total[:, None]


def _solve_optimal_split_factors(
    variance2_per_level: np.ndarray,
    tau2: np.ndarray,
    cost_weights: Sequence[float],
    optimization_mode: str = "monotone",
    *,
    query_metadata: Mapping[str, Any] | None = None,
) -> List[float]:
    """Solve for optimal split factors given per-level variance grids and tau^2."""
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

    M = _prepare_allocation_matrix(
        variance2_per_level,
        tau2,
        query_metadata=query_metadata,
    )
    row_weights = _query_weights_for_allocation(query_metadata, M.shape[0])
    if optimization_mode == "monotone":
        simplex = _solve_frank_wolfe_monotone_allocation(M, cost_w=cost_w)
    elif optimization_mode == "monotone_cvar95":
        simplex = _solve_cvar_monotone_allocation(
            M,
            cost_w=cost_w,
            row_weights=row_weights,
            alpha=_MONOTONE_CVAR95_ALPHA,
        )
    else:
        raise ValueError(f"unknown optimization_mode '{optimization_mode}'")
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

    merged["loss"] = str(merged.get("loss", "bce")).lower()
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
    if not isinstance(query_spec, Mapping):
        raise ValueError("crossfit_q requires query_spec for MLP query features")
    if query_spec.get("kind") != "sparse_subset_lower_orthant":
        raise ValueError(f"unknown query spec kind {query_spec.get('kind')!r}")

    ambient_dim = int(query_spec["ambient_dim"])
    if ambient_dim < 1:
        raise ValueError("query ambient_dim must be positive")
    active_dims = np.asarray(query_spec["active_dims"], dtype=np.int64)
    thresholds = np.asarray(query_spec["thresholds"], dtype=float)
    ranks = np.asarray(query_spec.get("ranks"), dtype=float)
    active_sizes = np.asarray(query_spec["active_sizes"], dtype=float).reshape(-1)
    target_mass = np.asarray(query_spec["target_mass"], dtype=float).reshape(-1)
    if active_dims.ndim != 2 or thresholds.shape != active_dims.shape:
        raise ValueError("query active_dims and thresholds must be matching 2D arrays")
    if ranks.shape != active_dims.shape:
        ranks = np.full(active_dims.shape, np.nan, dtype=float)
    num_queries, max_active = active_dims.shape
    if active_sizes.shape != (num_queries,) or target_mass.shape != (num_queries,):
        raise ValueError("query metadata shapes do not match active_dims")
    if np.any(active_dims >= ambient_dim):
        raise ValueError("query active dimension exceeds ambient dimension")

    active_mask = active_dims >= 0
    safe_dims = np.where(active_mask, active_dims, 0).astype(np.int64, copy=False)
    dim_scale = max(ambient_dim - 1, 1)
    dim_features = np.where(active_mask, safe_dims / float(dim_scale), 0.0)
    threshold_features = np.where(active_mask, thresholds, 0.0)
    rank_features = np.where(active_mask, np.nan_to_num(ranks, nan=0.0), 0.0)
    mass = np.clip(target_mass, 1e-6, 1.0 - 1e-6)
    scalar_features = np.column_stack(
        [
            active_sizes / float(ambient_dim),
            np.log(mass / (1.0 - mass)),
        ]
    )
    features = np.concatenate(
        [
            active_mask.astype(float),
            dim_features,
            threshold_features,
            rank_features,
            scalar_features,
        ],
        axis=1,
    )
    if not np.isfinite(features).all():
        raise ValueError("crossfit_q MLP query features contain non-finite values")
    return {
        "features": features.astype(np.float32, copy=False),
        "active_dims": safe_dims.astype(np.int64, copy=False),
        "active_mask": active_mask.astype(np.float32, copy=False),
        "thresholds": threshold_features.astype(np.float32, copy=False),
        "ambient_dim": int(ambient_dim),
        "max_active": int(max_active),
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

    x_train = torch.as_tensor(x_train_np, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train_np, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(x_test_np, dtype=torch.float32, device=device)
    q_features = torch.as_tensor(q_features_np, dtype=torch.float32, device=device)
    active_dims = torch.as_tensor(active_dims_np, dtype=torch.long, device=device)
    active_mask = torch.as_tensor(active_mask_np, dtype=torch.float32, device=device)
    time_features = torch.as_tensor(
        time_features_np, dtype=torch.float32, device=device
    )
    num_queries = int(q_features.shape[0])
    max_active = int(active_dims.shape[1])
    train_paths = int(x_train.shape[1])

    def triple_features(
        states: torch.Tensor, pair_idx: torch.Tensor, paths_per_level: int
    ):
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
        gathered = torch.gather(state_part, 1, dims) * mask
        time_part = time_features[level_idx]
        return torch.cat([state_part, query_part, gathered, time_part], dim=1)

    input_dim = int(x_train.shape[2]) + int(q_features.shape[1]) + max_active + 1
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    pair_count = int(num_levels) * int(train_paths) * int(num_queries)
    batch_size = min(int(params["batch_size"]), int(pair_count))
    epochs = int(params["epochs"])
    loss_mode = str(params["loss"])
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed) + 104729)

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
            level_path_idx = torch.div(idx, num_queries, rounding_mode="floor")
            query_idx = idx - level_path_idx * num_queries
            path_idx = (
                level_path_idx
                - torch.div(level_path_idx, train_paths, rounding_mode="floor")
                * train_paths
            )
            yb = y_train[path_idx, query_idx].reshape(-1, 1)
            logits = model(xb)
            loss = supervised_loss(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    test_paths = int(x_test.shape[1])
    test_pair_count = int(num_levels) * int(test_paths) * int(num_queries)
    predictions_flat = np.empty(test_pair_count, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, test_pair_count, batch_size):
            end = min(start + batch_size, test_pair_count)
            pair_idx = torch.arange(start, end, device=device, dtype=torch.long)
            xb = triple_features(x_test, pair_idx, test_paths)
            predictions_flat[start:end] = (
                torch.sigmoid(model(xb)).reshape(-1).detach().cpu().numpy()
            )
    return predictions_flat.reshape(num_levels, test_paths, num_queries)


def _crossfit_q_fold_indices(num_paths: int, n_folds: int):
    if int(n_folds) == 1:
        all_idx = np.arange(int(num_paths), dtype=np.int64)
        return [(all_idx, all_idx)]
    fold_ids = np.arange(int(num_paths), dtype=np.int64) % int(n_folds)
    return [
        (np.flatnonzero(fold_ids != fold_idx), np.flatnonzero(fold_ids == fold_idx))
        for fold_idx in range(int(n_folds))
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
    num_levels, num_paths, obs_dim = (
        states_by_level.shape[0],
        labels.shape[0],
        labels.shape[1],
    )
    predictions_by_level = np.empty((num_levels, num_paths, obs_dim), dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        train_labels = labels[train_idx]
        fold_seed = None if seed is None else int(seed) + fold_idx
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
    n_folds: int,
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

    n_folds = min(int(n_folds), num_paths)
    if n_folds < 1:
        raise ValueError("crossfit_q_folds must be at least 1")
    fold_indices = _crossfit_q_fold_indices(num_paths, n_folds)
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


def _normalize_grid_free_params(params: Mapping[str, Any] | None):
    merged = dict(_DEFAULT_GRID_FREE_PARAMS)
    if params:
        unknown = sorted(set(params) - set(_DEFAULT_GRID_FREE_PARAMS))
        if unknown:
            raise ValueError(f"Unknown grid_free_params: {unknown}")
        merged.update(dict(params))
    merged["num_queries"] = int(merged["num_queries"])
    if merged["num_queries"] < 1:
        raise ValueError("grid_free_params.num_queries must be >= 1")
    merged["tail_eps"] = float(merged.get("tail_eps", 1e-3))
    if not math.isfinite(merged["tail_eps"]) or not (0.0 <= merged["tail_eps"] < 0.5):
        raise ValueError("grid_free_params.tail_eps must be in [0, 0.5)")
    return {
        "num_queries": int(merged["num_queries"]),
        "tail_eps": float(merged["tail_eps"]),
    }


def _coerce_positive_int_list(value, *, name: str):
    if isinstance(value, int):
        values = [int(value)]
    else:
        values = [int(item) for item in (value or [])]
    if not values or any(item < 1 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _normalize_phase1_query_params(params: Mapping[str, Any] | None):
    merged = dict(_DEFAULT_PHASE1_QUERY_PARAMS)
    if params:
        unknown = sorted(set(params) - set(_DEFAULT_PHASE1_QUERY_PARAMS))
        if unknown:
            raise ValueError(f"Unknown phase1_query_params: {unknown}")
        merged.update(dict(params))
    merged["subset_sizes"] = _coerce_positive_int_list(
        merged.get("subset_sizes"),
        name="phase1_query_params.subset_sizes",
    )
    merged["mass_min"] = float(merged.get("mass_min", 0.02))
    merged["mass_max"] = float(merged.get("mass_max", 0.98))
    if not (
        math.isfinite(merged["mass_min"])
        and math.isfinite(merged["mass_max"])
        and 0.0 < merged["mass_min"] < merged["mass_max"] < 1.0
    ):
        raise ValueError("phase1_query_params requires 0 < mass_min < mass_max < 1")
    merged["rank_spread"] = float(merged.get("rank_spread", 0.25))
    if not math.isfinite(merged["rank_spread"]) or merged["rank_spread"] < 0.0:
        raise ValueError("phase1_query_params.rank_spread must be nonnegative")
    merged["subset_seed"] = int(merged.get("subset_seed", 0))
    merged["mass_bins"] = int(merged.get("mass_bins", 16))
    if merged["mass_bins"] < 1:
        raise ValueError("phase1_query_params.mass_bins must be >= 1")
    return merged


def _effective_sparse_subset_sizes(dim: int, params: Mapping[str, Any]):
    dim = int(dim)
    if dim < 1:
        raise ValueError("query dimension must be positive")
    if dim == 1:
        return [1]
    sizes = sorted({min(int(size), dim) for size in params["subset_sizes"]})
    if not sizes:
        raise ValueError(
            "phase1_query_params.subset_sizes produced no valid subset sizes"
        )
    return sizes


def _allocate_query_counts(total: int, subset_sizes: Sequence[int]):
    total = int(total)
    if total < 1:
        raise ValueError("num_queries must be positive")
    sizes = list(subset_sizes)
    base = total // len(sizes)
    remainder = total - base * len(sizes)
    return {int(size): int(base + (idx < remainder)) for idx, size in enumerate(sizes)}


def _mass_grid(count: int, mass_min: float, mass_max: float):
    if int(count) < 1:
        return np.empty(0, dtype=float)
    return mass_min + (np.arange(int(count), dtype=float) + 0.5) / float(count) * (
        mass_max - mass_min
    )


def _sparse_query_weights(
    subset_sizes: np.ndarray, target_mass: np.ndarray, mass_bins: int
):
    subset_sizes = np.asarray(subset_sizes, dtype=np.int64)
    target_mass = np.asarray(target_mass, dtype=float)
    if subset_sizes.ndim != 1 or target_mass.shape != subset_sizes.shape:
        raise ValueError("query weight metadata must be 1D with matching shapes")
    bins = np.floor(np.clip(target_mass, 0.0, 1.0 - 1e-15) * int(mass_bins)).astype(int)
    bins = np.clip(bins, 0, int(mass_bins) - 1)
    strata = list(zip(subset_sizes.tolist(), bins.tolist()))
    unique = sorted(set(strata))
    weights = np.zeros(subset_sizes.shape[0], dtype=float)
    if not unique:
        raise ValueError("query weights require at least one query")
    for stratum in unique:
        mask = np.array([item == stratum for item in strata], dtype=bool)
        weights[mask] = 1.0 / (len(unique) * int(mask.sum()))
    weights /= float(weights.sum())
    return weights


def _generate_grid_free_queries(
    samples,
    params: Mapping[str, Any] | None,
    *,
    phase1_query_params: Mapping[str, Any] | None = None,
):
    params = _normalize_grid_free_params(params)
    phase1_query_params = _normalize_phase1_query_params(phase1_query_params)
    samples_np = coerce_samples_np(samples)
    if samples_np.shape[0] < 1:
        raise ValueError("grid-free phase 1 needs at least one terminal sample")
    dim = int(samples_np.shape[1])
    total = int(params["num_queries"])
    subset_sizes = _effective_sparse_subset_sizes(dim, phase1_query_params)
    query_counts = _allocate_query_counts(total, subset_sizes)
    max_active = max(subset_sizes)
    active_dims = np.full((total, max_active), -1, dtype=np.int64)
    thresholds = np.full((total, max_active), np.nan, dtype=float)
    active_sizes = np.empty(total, dtype=np.int64)
    target_mass = np.empty(total, dtype=float)
    ranks_used = np.full((total, max_active), np.nan, dtype=float)
    rng = np.random.default_rng(int(phase1_query_params["subset_seed"]) + 104729 * dim)
    eps = float(params["tail_eps"])
    query_idx = 0
    for subset_size, count in query_counts.items():
        masses = _mass_grid(
            count,
            float(phase1_query_params["mass_min"]),
            float(phase1_query_params["mass_max"]),
        )
        for mass in masses:
            dims = np.sort(rng.choice(dim, size=int(subset_size), replace=False))
            base_rank = float(mass) ** (1.0 / float(subset_size))
            if subset_size > 1 and float(phase1_query_params["rank_spread"]) > 0.0:
                offsets = rng.normal(size=int(subset_size))
                offsets = offsets - float(offsets.mean())
                max_abs = float(np.max(np.abs(offsets)))
                if max_abs > 0.0:
                    offsets = (
                        offsets / max_abs * float(phase1_query_params["rank_spread"])
                    )
                ranks = np.exp(np.log(base_rank) + offsets)
            else:
                ranks = np.full(int(subset_size), base_rank, dtype=float)
            ranks = np.clip(ranks, eps, 1.0 - eps)
            active_dims[query_idx, : int(subset_size)] = dims
            active_sizes[query_idx] = int(subset_size)
            target_mass[query_idx] = float(mass)
            ranks_used[query_idx, : int(subset_size)] = ranks
            for pos, dim_idx in enumerate(dims):
                thresholds[query_idx, pos] = np.quantile(
                    samples_np[:, dim_idx], ranks[pos]
                )
            query_idx += 1
    if query_idx != total:
        raise RuntimeError(
            "internal query allocation did not produce requested query count"
        )
    weights = _sparse_query_weights(
        active_sizes, target_mass, int(phase1_query_params["mass_bins"])
    )
    spec = {
        "kind": "sparse_subset_lower_orthant",
        "ambient_dim": int(dim),
        "thresholds": thresholds,
        "active_dims": active_dims,
        "active_sizes": active_sizes,
        "target_mass": target_mass,
        "query_weights": weights,
        "ranks": ranks_used,
    }
    return spec


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
    empirical_mass = labels.mean(axis=0).astype(float)
    if not isinstance(query_spec, Mapping):
        raise ValueError("query metadata requires sparse subset query specs")
    target_mass = np.asarray(query_spec["target_mass"], dtype=float)
    subset_size = np.asarray(query_spec["active_sizes"], dtype=np.int64)
    weights = np.asarray(query_spec["query_weights"], dtype=float)
    if (
        target_mass.shape != empirical_mass.shape
        or subset_size.shape != empirical_mass.shape
    ):
        raise ValueError("query metadata and labels disagree on query count")
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if (
        weights.shape != empirical_mass.shape
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
    ):
        weights = _sparse_query_weights(subset_size, target_mass, int(mass_bins))
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        weights = np.full(
            empirical_mass.shape[0], 1.0 / empirical_mass.shape[0], dtype=float
        )
    else:
        weights = weights / total_weight
    return {
        "weights": weights,
        "empirical_mass": empirical_mass,
        "target_mass": target_mass,
        "n_paths": int(labels.shape[0]),
    }


def _simulate_crossfit_q_trajectories(
    runner,
    *,
    chunk_size: int,
    paths_per_run: int,
    split_points: Sequence[Any],
    generator=None,
):
    total_paths = int(chunk_size) * int(paths_per_run)
    x = runner.sample_prior(total_paths, generator=generator)
    start_points = [runner.start_time] + list(split_points)
    end_points = list(split_points) + [runner.end_time]
    states_by_level = []
    for start_t, end_t in zip(start_points, end_points):
        states_by_level.append(x.detach().reshape(total_paths, -1).clone())
        x = runner.sample_segment(x, start_t, end_t, generator=generator)

    x0 = runner.postprocess_samples(x)
    states = (
        torch.stack(states_by_level, dim=0)
        .cpu()
        .numpy()
        .astype(float, copy=False)
        .reshape(len(states_by_level), int(chunk_size), int(paths_per_run), -1)
    )
    return states, x0


def _grid_free_query_payload(samples, grid_free_params, phase1_query_params):
    query_spec = _generate_grid_free_queries(
        samples,
        grid_free_params,
        phase1_query_params=phase1_query_params,
    )
    labels = _lower_orthant_labels(samples, query_spec)
    labels_np = labels.detach().cpu().numpy().astype(np.bool_, copy=False)
    query_metadata = _query_metadata(
        query_spec,
        labels_np,
        mass_bins=int(phase1_query_params["mass_bins"]),
    )
    return query_spec, labels, query_metadata


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


def _run_joint_phase1_sampling_batch(
    runner,
    *,
    B1,
    joint_m,
    split_percentages,
    grid_free_params,
    phase1_query_params,
    reuse_phase1_samples,
    chunk_size,
    debug=False,
    generator=None,
):
    grid_free_params = _normalize_grid_free_params(grid_free_params)
    phase1_query_params = _normalize_phase1_query_params(phase1_query_params)
    _, split_points = runner.resolve_split_percentages(split_percentages)
    num_levels = len(split_points) + 1
    joint_roots, joint_m = _derive_joint_tree_shape(runner, split_points, B1, joint_m)
    _debug(
        debug,
        f"Phase 1 joint: chunk={chunk_size} "
        f"joint_roots={joint_roots} joint_m={joint_m:.4f}",
    )

    (
        leaf_x,
        leaf_run_ids,
        child_counts_by_level,
        parent_run_ids_by_level,
        used_B1_by_run,
    ) = _build_joint_tree_batch(
        runner,
        chunk_size,
        joint_roots,
        split_points,
        joint_m,
        generator=generator,
    )
    leaf_x0 = runner.postprocess_samples(leaf_x)
    counts = torch.bincount(leaf_run_ids, minlength=chunk_size).cpu().tolist()
    leaf_x0_by_run = list(torch.split(leaf_x0, counts))
    labels_by_run = []
    query_metadata_by_run = []
    for run_samples in leaf_x0_by_run:
        _, labels, query_metadata = _grid_free_query_payload(
            run_samples,
            grid_free_params,
            phase1_query_params,
        )
        labels_by_run.append(labels.to(device=leaf_x0.device, dtype=leaf_x0.dtype))
        query_metadata_by_run.append(query_metadata)
    leaf_observables = torch.cat(labels_by_run, dim=0)

    variance2_arr, tau2_arr = _estimate_variances_from_tree_batch(
        leaf_observables,
        num_levels,
        child_counts_by_level,
        parent_run_ids_by_level,
        chunk_size,
    )

    phase1_x0_by_run: List[Any] = [None] * chunk_size
    if reuse_phase1_samples:
        phase1_x0_by_run = [coerce_samples_np(s) for s in leaf_x0_by_run]

    return [
        {
            "split_points": list(split_points),
            "variance2_per_level": variance2_arr[run_idx],
            "tau2": tau2_arr[run_idx],
            "used_B1": int(used_B1_by_run[run_idx]),
            "phase1_x0_samples": phase1_x0_by_run[run_idx],
            "query_metadata": query_metadata_by_run[run_idx],
        }
        for run_idx in range(chunk_size)
    ]


def _run_independent_phase1_sampling_batch(
    runner,
    *,
    B1,
    split_percentages,
    grid_free_params,
    phase1_query_params,
    independent_n2,
    reuse_phase1_samples,
    chunk_size,
    debug=False,
    generator=None,
):
    grid_free_params = _normalize_grid_free_params(grid_free_params)
    phase1_query_params = _normalize_phase1_query_params(phase1_query_params)
    _, split_points = runner.resolve_split_percentages(split_percentages)
    variance_times = [runner.start_time] + list(split_points)
    num_levels = len(variance_times)
    _, counts_by_variance, _ = _derive_independent_counts(
        runner, variance_times, B1, independent_n2
    )
    _debug(
        debug, f"Phase 1 independent: chunk={chunk_size} counts={counts_by_variance}"
    )

    next_times = list(variance_times[1:]) + [runner.end_time]

    variance2_per_level: np.ndarray | None = None
    tau2: np.ndarray | None = None
    obs_dim: int | None = None
    phase1_samples_by_run: List[list] = [[] for _ in range(chunk_size)]
    query_specs_by_run: List[Any] = [None] * chunk_size
    query_metadata_by_run: List[Any] = [None] * chunk_size
    used_B1 = 0

    for idx, ((t_curr, t_next), spec) in enumerate(
        zip(zip(variance_times, next_times), counts_by_variance)
    ):
        outer = int(spec["outer_count"])
        middle = int(spec["middle_count"])
        inner = int(spec["inner_count"])
        leaf_per_run = outer * middle * inner

        root = runner.sample_prior(chunk_size * outer, generator=generator)
        x_curr = runner.sample_segment(
            root, runner.start_time, t_curr, generator=generator
        )
        x_curr_rep = repeat_by_counts(
            x_curr,
            torch.full(
                (x_curr.shape[0],), middle, device=runner.device, dtype=torch.long
            ),
        )
        x_next = runner.sample_segment(x_curr_rep, t_curr, t_next, generator=generator)
        x_next_rep = repeat_by_counts(
            x_next,
            torch.full(
                (x_next.shape[0],), inner, device=runner.device, dtype=torch.long
            ),
        )
        x_0 = runner.sample_segment(
            x_next_rep, t_next, runner.end_time, generator=generator
        )
        x_0 = runner.postprocess_samples(x_0)
        x0_by_run = list(torch.split(x_0, leaf_per_run))

        if reuse_phase1_samples:
            for run_idx in range(chunk_size):
                phase1_samples_by_run[run_idx].append(x0_by_run[run_idx])

        used_B1 += _independent_variance_cost(
            runner, t_curr, t_next, outer, middle, inner
        )

        labels_by_run = []
        for run_idx, run_samples in enumerate(x0_by_run):
            if idx == 0:
                query_spec, labels, query_metadata = _grid_free_query_payload(
                    run_samples,
                    grid_free_params,
                    phase1_query_params,
                )
                query_specs_by_run[run_idx] = query_spec
                query_metadata_by_run[run_idx] = query_metadata
            else:
                labels = _lower_orthant_labels(run_samples, query_specs_by_run[run_idx])
            labels_by_run.append(labels.to(device=x_0.device, dtype=x_0.dtype))
        observables = torch.cat(labels_by_run, dim=0)
        if obs_dim is None:
            obs_dim = observables.shape[1]
            variance2_per_level = np.zeros(
                (chunk_size, num_levels, obs_dim), dtype=float
            )
            tau2 = np.zeros((chunk_size, obs_dim), dtype=float)
        indicators = observables.reshape(chunk_size, outer, middle, inner, obs_dim)
        middle_means = indicators.mean(dim=3)
        if middle < 2:
            raise ValueError("Independent variance estimation needs middle_count >= 2")
        outer_vars = torch.clamp(middle_means.var(dim=2, unbiased=True), min=0.0)
        raw_variance2 = outer_vars.mean(dim=1)
        if inner >= 2 and runner.segment_cost(t_next, runner.end_time) > 0:
            inner_vars = torch.clamp(indicators.var(dim=3, unbiased=True), min=0.0)
            variance2_level = raw_variance2 - inner_vars.mean(dim=(1, 2)) / float(inner)
        else:
            variance2_level = raw_variance2
        variance2_level = torch.clamp(variance2_level, min=0.0)
        variance2_per_level[:, idx, :] = variance2_level.cpu().numpy()

        if idx == 0:
            outer_means = middle_means.mean(dim=2)
            if outer >= 2:
                tau2_level = torch.clamp(outer_means.var(dim=1, unbiased=True), min=0.0)
            else:
                tau2_level = torch.zeros(
                    chunk_size, obs_dim, device=runner.device, dtype=x_0.dtype
                )
            tau2[:] = tau2_level.cpu().numpy()

    if variance2_per_level is None or tau2 is None:
        raise RuntimeError("Independent phase-1 produced no variance estimates")

    return [
        {
            "split_points": list(split_points),
            "variance2_per_level": variance2_per_level[run_idx],
            "tau2": tau2[run_idx],
            "used_B1": int(used_B1),
            "phase1_x0_samples": (
                coerce_samples_np(torch.cat(phase1_samples_by_run[run_idx], dim=0))
                if reuse_phase1_samples and phase1_samples_by_run[run_idx]
                else None
            ),
            "query_metadata": query_metadata_by_run[run_idx],
        }
        for run_idx in range(chunk_size)
    ]


def _run_crossfit_q_phase1_sampling_batch(
    runner,
    *,
    B1,
    split_percentages,
    crossfit_q_folds,
    crossfit_q_mlp_run_parallelism=1,
    crossfit_q_mlp_params=None,
    grid_free_params=None,
    phase1_query_params=None,
    reuse_phase1_samples,
    chunk_size,
    debug=False,
    generator=None,
):
    grid_free_params = _normalize_grid_free_params(grid_free_params)
    phase1_query_params = _normalize_phase1_query_params(phase1_query_params)
    mlp_params_normalized = _crossfit_q_normalize_mlp_params(crossfit_q_mlp_params)
    mlp_run_parallelism = _crossfit_q_normalize_mlp_run_parallelism(
        crossfit_q_mlp_run_parallelism
    )
    effective_folds = int(crossfit_q_folds)
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
    if paths_per_run < int(effective_folds):
        raise ValueError(
            f"crossfit_q_folds={effective_folds} exceeds the {paths_per_run} "
            "pilot paths available per run"
        )

    used_B1 = int(paths_per_run * full_path_cost)
    _debug(
        debug,
        f"Phase 1 crossfit_q: chunk={chunk_size} paths_per_run={paths_per_run} "
        f"folds={effective_folds} mlp_workers={min(mlp_run_parallelism, int(chunk_size))} "
        f"mlp_device={mlp_device} mlp_num_threads={mlp_params_normalized.get('num_threads')}",
    )

    states_by_run, x0 = _simulate_crossfit_q_trajectories(
        runner,
        chunk_size=chunk_size,
        paths_per_run=paths_per_run,
        split_points=split_points,
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
            grid_free_params,
            phase1_query_params,
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
                n_folds=int(effective_folds),
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
                                n_folds=int(effective_folds),
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
    free_pilot=False,
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
    B2 = int(B) if free_pilot else B - int(payload["used_B1"])
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
    B: int,
    B1: int,
    split_percentages,
    independent_n2: int,
    joint_m: float,
    variance_estimation_mode: str,
    reuse_phase1_samples: bool,
    n_runs: int,
    seed: int | None,
    crossfit_q_folds: int = CROSSFIT_Q_DEFAULT_FOLDS,
    crossfit_q_mlp_run_parallelism: int = CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM,
    crossfit_q_mlp_params: Mapping[str, Any] | None = None,
    grid_free_params: Mapping[str, Any] | None = None,
    phase1_query_params: Mapping[str, Any] | None = None,
    optimization_mode: str = "monotone",
    free_pilot: bool = False,
    debug: bool = False,
    n_parallel: int = 1,
    run_offset: int = 0,
    return_trial_results: bool = False,
):
    if B1 < 0 or (not free_pilot and B1 >= B):
        raise ValueError("require 0 <= B1, and B1 < B unless free_pilot")
    if variance_estimation_mode == "independent" and independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")
    if variance_estimation_mode == "joint" and (
        joint_m <= 1.0 or not math.isfinite(float(joint_m))
    ):
        raise ValueError("joint_m must be finite and > 1")
    grid_free_params_normalized = _normalize_grid_free_params(grid_free_params)
    phase1_query_params_normalized = _normalize_phase1_query_params(phase1_query_params)
    crossfit_q_mlp_params_normalized = None
    crossfit_q_mlp_run_parallelism_normalized = int(
        CROSSFIT_Q_DEFAULT_MLP_RUN_PARALLELISM
    )
    if variance_estimation_mode == "crossfit_q":
        if int(crossfit_q_folds) < 1:
            raise ValueError("crossfit_q_folds must be at least 1")
        crossfit_q_mlp_run_parallelism_normalized = (
            _crossfit_q_normalize_mlp_run_parallelism(crossfit_q_mlp_run_parallelism)
        )
        crossfit_q_mlp_params_normalized = _crossfit_q_normalize_mlp_params(
            crossfit_q_mlp_params
        )
    if optimization_mode not in SUPPORTED_OPTIMIZATION_MODES:
        raise ValueError(f"unknown optimization_mode '{optimization_mode}'")

    common_phase1_extra = {
        "grid_free_params": dict(grid_free_params_normalized),
        "phase1_query_params": dict(phase1_query_params_normalized),
    }
    if variance_estimation_mode == "joint":
        phase1_fn = _run_joint_phase1_sampling_batch
        phase1_extra: Dict[str, Any] = {
            **common_phase1_extra,
            "joint_m": float(joint_m),
        }
    elif variance_estimation_mode == "independent":
        phase1_fn = _run_independent_phase1_sampling_batch
        phase1_extra = {
            **common_phase1_extra,
            "independent_n2": int(independent_n2),
        }
    elif variance_estimation_mode == "crossfit_q":
        phase1_fn = _run_crossfit_q_phase1_sampling_batch
        phase1_extra = {
            **common_phase1_extra,
            "crossfit_q_folds": int(crossfit_q_folds),
            "crossfit_q_mlp_run_parallelism": int(
                crossfit_q_mlp_run_parallelism_normalized
            ),
            "crossfit_q_mlp_params": dict(crossfit_q_mlp_params_normalized or {}),
        }
    else:
        raise ValueError(
            f"unknown variance_estimation_mode '{variance_estimation_mode}'"
        )

    _debug(
        debug,
        f"Starting estimate_and_sample B={B} B1={B1} runner={runner.runner_name} "
        f"mode={variance_estimation_mode} optimizer={optimization_mode} "
        f"free_pilot={free_pilot}",
    )

    chunks = list(iter_run_chunks(n_runs, n_parallel))
    trial_results: List[Any] = [None] * n_runs
    workers = max(1, min(n_runs, n_parallel))

    with ThreadPoolExecutor(max_workers=workers) as alloc_executor:
        allocation_futures = []
        for start, end in tqdm(chunks, desc="Phase 1", leave=False):
            chunk_size = end - start
            run_seed = None if seed is None else seed + run_offset + start
            payloads = phase1_fn(
                runner,
                B1=B1,
                split_percentages=split_percentages,
                reuse_phase1_samples=reuse_phase1_samples,
                chunk_size=chunk_size,
                debug=debug,
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
                    free_pilot=free_pilot,
                    optimization_mode=optimization_mode,
                )
                allocation_futures.append((run_idx, future))

        for run_idx, future in tqdm(
            allocation_futures, desc="Optimize N_i", leave=False
        ):
            trial_results[run_idx] = future.result()

    if any(t is None for t in trial_results):
        raise RuntimeError("Phase 1 allocation did not produce all trial results")

    ks_workers = max(1, min(int(n_parallel), n_runs))
    ks_futures: List[Tuple[int, Any]] = []
    with ThreadPoolExecutor(max_workers=ks_workers) as ks_executor:
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
            )
            submit_sampling_trial_result_futures(
                executor=ks_executor,
                futures=ks_futures,
                run_indices=range(start, end),
                samples_by_run=samples_by_run,
                runner=runner,
                comparison_mode=comparison_mode,
                comparison_state=comparison_state,
                phase1_x0_samples_by_run=[t.get("phase1_x0_samples") for t in chunk],
            )
        for run_idx, future in tqdm(ks_futures, desc="KS", leave=False):
            trial_results[run_idx].update(future.result())

    result = {
        "mode": "estimate_and_sample",
        "B": int(B),
        "B1": int(B1),
        **summarize_sampling_trials(trial_results),
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result
