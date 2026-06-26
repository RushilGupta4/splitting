import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

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


def _debug(enabled: bool, message: str):
    if enabled:
        log.debug(message)


# --- Pilot-tree shape derivation --------------------------------------------


def _pilot_tree_floor_cost(
    runner,
    split_points: Sequence[Any],
    pilot_roots: int,
    m_pilot: float,
):
    start_points = [runner.start_time] + list(split_points)
    end_points = list(split_points) + [runner.end_time]
    current_count = int(pilot_roots)
    cost = 0.0
    for start_t, end_t in zip(start_points, end_points):
        current_count = floor_split_total_count(current_count, m_pilot)
        cost += current_count * int(runner.segment_cost(start_t, end_t))
    return cost


def _derive_pilot_tree_shape(
    runner, split_points: Sequence[Any], B1: int, m_pilot: float
):
    m_pilot = float(m_pilot)
    if m_pilot <= 1.0 or not math.isfinite(m_pilot):
        raise ValueError("m_pilot must be finite and > 1")

    cost_for_one = _pilot_tree_floor_cost(runner, split_points, 1, m_pilot)
    if B1 < cost_for_one:
        raise ValueError(
            f"B1={B1} is too small for pilot_tree with m_pilot={m_pilot:g}; "
            f"need at least {cost_for_one:.6f}"
        )

    lower, upper = 1, 2
    while _pilot_tree_floor_cost(runner, split_points, upper, m_pilot) <= B1:
        lower = upper
        upper *= 2

    high = upper - 1
    while lower < high:
        mid = (lower + high + 1) // 2
        if _pilot_tree_floor_cost(runner, split_points, mid, m_pilot) <= B1:
            lower = mid
        else:
            high = mid - 1
    return int(lower), float(m_pilot)


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


# --- Pilot tree construction + variance estimation --------------------------


def _build_pilot_tree_batch(
    runner,
    chunk_size: int,
    pilot_roots: int,
    split_points: Sequence[Any],
    m_pilot: float,
    generator=None,
):
    x = runner.sample_prior(chunk_size * pilot_roots, generator=generator)
    run_ids = torch.repeat_interleave(
        torch.arange(chunk_size, device=runner.device, dtype=torch.long), pilot_roots
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
                float(m_pilot),
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


def _normalize_simplex_with_floor(weights: np.ndarray, floor: float):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("simplex weights must be 1D")
    n = weights.shape[0]
    if n == 0:
        raise ValueError("simplex weights must be non-empty")

    floor = min(max(float(floor), 0.0), 0.5 / float(n))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(n, 1.0 / n, dtype=float)
    weights = weights / total
    if floor <= 0.0 or np.all(weights >= floor):
        return weights
    above = np.maximum(weights - floor, 0.0)
    above_sum = float(above.sum())
    if above_sum <= 0.0:
        return np.full(n, 1.0 / n, dtype=float)
    return floor + max(1.0 - floor * n, 0.0) * above / above_sum


def _objective_values(M, y):
    return M @ (1.0 / y)


def _top_indices(values, count):
    count = min(max(int(count), 0), int(values.shape[0]))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == values.shape[0]:
        return np.arange(values.shape[0], dtype=np.int64)
    return np.argpartition(values, -count)[-count:].astype(np.int64, copy=False)


def _softmax_from_values(values, temperature):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    temperature = max(float(temperature), 1e-300)
    shifted = (values - float(np.max(values))) / temperature
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(values.shape[0], 1.0 / values.shape[0], dtype=float)
    return weights / total


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


def _monotone_value_and_simplex_from_weights(M, cost_w, w):
    w = _normalize_simplex_with_floor(w, 0.0)
    profile = np.maximum(np.asarray(M, dtype=float).T @ w, 0.0)
    y = _monotone_simplex_from_profile(profile, cost_w)
    return float(w @ _objective_values(M * cost_w[None, :], y)), y


def _maximize_monotone_profile(active_M, cost_w, initial_w=None, *, max_iters=300):
    num_active = int(active_M.shape[0])
    if initial_w is None:
        w = np.full(num_active, 1.0 / num_active, dtype=float)
    else:
        w = _normalize_simplex_with_floor(initial_w, 0.0)
    best_obj, best_y = _monotone_value_and_simplex_from_weights(active_M, cost_w, w)
    best_w = w.copy()
    no_improvement = 0

    active_weighted_M = active_M * cost_w[None, :]
    for _ in range(max_iters):
        _, y = _monotone_value_and_simplex_from_weights(active_M, cost_w, w)
        gradient = _objective_values(active_weighted_M, y)
        scale = float(np.max(np.abs(gradient)))
        if scale <= 0.0 or not np.isfinite(scale):
            break
        step = 0.5 / scale
        accepted = False
        for _ in range(16):
            expo = step * gradient
            expo = np.clip(expo - float(np.max(expo)), -50.0, 50.0)
            cand = _normalize_simplex_with_floor(w * np.exp(expo), 0.0)
            cand_obj, cand_y = _monotone_value_and_simplex_from_weights(
                active_M, cost_w, cand
            )
            if cand_obj >= best_obj * (1.0 - 1e-12):
                improvement = cand_obj - best_obj
                w = cand
                accepted = True
                if cand_obj > best_obj:
                    best_w = cand.copy()
                    best_y = cand_y
                    best_obj = cand_obj
                no_improvement = (
                    no_improvement + 1
                    if improvement <= max(1e-12, 1e-10 * max(abs(best_obj), 1.0))
                    else 0
                )
                break
            step *= 0.5
        if not accepted or no_improvement >= 25:
            break
    return best_y, best_w


def _solve_active_set_monotone_allocation(
    M: np.ndarray,
    y_initial: np.ndarray,
    *,
    cost_w: np.ndarray,
    relative_tol: float = 1e-5,
    max_outer_iters: int = 250,
    max_dual_iters: int = 1000,
):
    num_points, num_levels = M.shape
    if num_points == 0 or num_levels == 0:
        raise ValueError("M must be non-empty")

    weighted_M = M * cost_w[None, :]
    initial_active = min(num_points, max(32, 4 * num_levels))
    add_count = min(num_points, max(8, num_levels))
    max_active = min(num_points, max(256, 16 * num_levels))

    row_scores = np.sqrt(np.maximum(weighted_M, 0.0)).sum(axis=1)
    active_indices = set(int(idx) for idx in _top_indices(row_scores, initial_active))

    uniform_y = np.full(num_levels, 1.0 / num_levels, dtype=float)
    for probe_y in (y_initial, uniform_y):
        probe = _normalize_simplex_with_floor(probe_y, 1e-12)
        active_indices.update(
            int(idx)
            for idx in _top_indices(_objective_values(weighted_M, probe), add_count)
        )

    y_best = np.asarray(cost_w, dtype=float).copy()
    for _ in range(max_outer_iters):
        active_array = np.array(sorted(active_indices), dtype=np.int64)
        active_M = M[active_array]
        active_values = _objective_values(weighted_M[active_array], y_best)
        temperature = max(float(np.max(active_values)) * 1e-4, 1e-300)
        initial_w = _softmax_from_values(active_values, temperature)
        y_cand, active_dual = _maximize_monotone_profile(
            active_M, cost_w, initial_w, max_iters=max_dual_iters
        )
        full_values = _objective_values(weighted_M, y_cand)
        full_upper = float(np.max(full_values))
        full_dual = np.zeros(num_points, dtype=float)
        full_dual[active_array] = active_dual
        dual_lower, _ = _monotone_value_and_simplex_from_weights(M, cost_w, full_dual)
        gap = max(full_upper - dual_lower, 0.0) / max(abs(full_upper), 1.0)
        y_best = y_cand
        if gap <= relative_tol:
            break
        worst = _top_indices(full_values, add_count)
        prev_size = len(active_indices)
        active_indices.update(int(idx) for idx in worst)
        if len(active_indices) > max_active:
            keep = _top_indices(full_values, max_active)
            active_indices = set(int(idx) for idx in keep) | set(
                int(idx) for idx in active_array
            )
            if len(active_indices) > max_active:
                ranked = sorted(
                    active_indices, key=lambda idx: full_values[idx], reverse=True
                )
                active_indices = set(ranked[:max_active])
        if len(active_indices) == prev_size:
            break

    return y_best


def _monotone_fw_dual_value_and_simplex(profile, cost_w):
    profile = np.maximum(np.asarray(profile, dtype=float), 0.0)
    y = _monotone_simplex_from_profile(profile, cost_w)
    value = float((profile * cost_w) @ (1.0 / y))
    return value, y


def _monotone_fw_line_search(profile, target_profile, cost_w, *, max_iters=48):
    profile = np.asarray(profile, dtype=float)
    target_profile = np.asarray(target_profile, dtype=float)
    if np.allclose(profile, target_profile, rtol=0.0, atol=0.0):
        value, y = _monotone_fw_dual_value_and_simplex(profile, cost_w)
        return 0.0, value, y

    def value_at(gamma):
        candidate = (1.0 - gamma) * profile + gamma * target_profile
        return _monotone_fw_dual_value_and_simplex(candidate, cost_w)

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
        dual_lower, y = _monotone_fw_dual_value_and_simplex(profile, cost_w)
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
        gamma, candidate_lower, candidate_y = _monotone_fw_line_search(
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


def _solve_optimal_split_factors(
    variance2_per_level: np.ndarray,
    tau2: np.ndarray,
    cost_weights: Sequence[float],
    optimization_mode: str = "monotone",
) -> List[float]:
    """Solve for optimal split factors given per-level variance grids and tau^2."""
    if optimization_mode not in {"monotone", "monotone_fw"}:
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

    # (grid_pts, num_levels) — collapse leaf-level tau into level 0.
    M = np.maximum(variance2_per_level, 0.0).T.copy()
    M[:, 0] += np.maximum(tau2, 0.0)

    cost_w = np.asarray(cost_weights, dtype=float)
    if cost_w.shape != (num_levels,):
        raise ValueError(f"cost_weights shape {cost_w.shape} != ({num_levels},)")
    if not np.isfinite(cost_w).all() or np.any(cost_w <= 0.0):
        raise ValueError("cost_weights must be finite and positive")
    cost_w = cost_w / float(cost_w.sum())

    weighted_M = M * cost_w[None, :]
    root_matrix = np.sqrt(weighted_M)
    row_sums = root_matrix.sum(axis=1)
    j_star = int(np.argmax(row_sums))
    roots_at_j = root_matrix[j_star]
    total = float(roots_at_j.sum())
    y_initial = (
        roots_at_j / total
        if total > 0.0
        else np.full(num_levels, 1.0 / num_levels, dtype=float)
    )

    if optimization_mode == "monotone":
        simplex = _solve_active_set_monotone_allocation(M, y_initial, cost_w=cost_w)
    else:
        simplex = _solve_frank_wolfe_monotone_allocation(M, cost_w=cost_w)
    simplex = np.asarray(simplex, dtype=float)
    simplex /= float(simplex.sum())
    allocation = simplex / cost_w
    allocation /= float(cost_w @ allocation)
    return (allocation[1:] / allocation[:-1]).tolist()


# --- Phase 1 builders -------------------------------------------------------


def _run_pilot_tree_phase1_sampling_batch(
    runner,
    *,
    B1,
    pilot_m,
    comparison_mode,
    x_grid,
    split_percentages,
    reuse_phase1_samples,
    chunk_size,
    debug=False,
    generator=None,
):
    _, split_points = runner.resolve_split_percentages(split_percentages)
    num_levels = len(split_points) + 1
    pilot_roots, m_pilot = _derive_pilot_tree_shape(runner, split_points, B1, pilot_m)
    _debug(
        debug,
        f"Phase 1 pilot_tree: chunk={chunk_size} "
        f"pilot_roots={pilot_roots} m_pilot={m_pilot:.4f}",
    )

    (
        leaf_x,
        leaf_run_ids,
        child_counts_by_level,
        parent_run_ids_by_level,
        used_B1_by_run,
    ) = _build_pilot_tree_batch(
        runner,
        chunk_size,
        pilot_roots,
        split_points,
        m_pilot,
        generator=generator,
    )
    leaf_x0 = runner.postprocess_samples(leaf_x)
    leaf_observables = runner.observable_values(
        leaf_x0, comparison_mode=comparison_mode, x_grid=x_grid
    )

    variance2_arr, tau2_arr = _estimate_variances_from_tree_batch(
        leaf_observables,
        num_levels,
        child_counts_by_level,
        parent_run_ids_by_level,
        chunk_size,
    )

    phase1_x0_by_run: List[Any] = [None] * chunk_size
    if reuse_phase1_samples:
        counts = torch.bincount(leaf_run_ids, minlength=chunk_size).cpu().tolist()
        phase1_x0_by_run = [coerce_samples_np(s) for s in torch.split(leaf_x0, counts)]

    return [
        {
            "split_points": list(split_points),
            "variance2_per_level": variance2_arr[run_idx],
            "tau2": tau2_arr[run_idx],
            "used_B1": int(used_B1_by_run[run_idx]),
            "phase1_x0_samples": phase1_x0_by_run[run_idx],
        }
        for run_idx in range(chunk_size)
    ]


def _run_independent_phase1_sampling_batch(
    runner,
    *,
    B1,
    comparison_mode,
    x_grid,
    split_percentages,
    independent_n2,
    reuse_phase1_samples,
    chunk_size,
    debug=False,
    generator=None,
):
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

        if reuse_phase1_samples:
            x0_by_run = x_0.reshape(chunk_size, leaf_per_run, x_0.shape[-1])
            for run_idx in range(chunk_size):
                phase1_samples_by_run[run_idx].append(x0_by_run[run_idx])

        used_B1 += _independent_variance_cost(
            runner, t_curr, t_next, outer, middle, inner
        )

        observables = runner.observable_values(
            x_0,
            comparison_mode=comparison_mode,
            x_grid=x_grid,
        )
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
        }
        for run_idx in range(chunk_size)
    ]


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
    payload, runner, *, B, free_pilot=False, optimization_mode="monotone"
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
    x_grid,
    independent_n2: int,
    pilot_m: float,
    variance_estimation_mode: str,
    optimization_mode: str = "monotone",
    reuse_phase1_samples: bool,
    free_pilot: bool = False,
    n_runs: int,
    seed: int | None,
    debug: bool = False,
    n_parallel: int = 1,
    run_offset: int = 0,
    return_trial_results: bool = False,
):
    if B1 < 0 or (not free_pilot and B1 >= B):
        raise ValueError("require 0 <= B1, and B1 < B unless free_pilot")
    if variance_estimation_mode == "independent" and independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")
    if variance_estimation_mode == "pilot_tree" and (
        pilot_m <= 1.0 or not math.isfinite(float(pilot_m))
    ):
        raise ValueError("pilot_m must be finite and > 1")
    if optimization_mode not in {"monotone", "monotone_fw"}:
        raise ValueError(f"unknown optimization_mode '{optimization_mode}'")

    if variance_estimation_mode == "pilot_tree":
        phase1_fn = _run_pilot_tree_phase1_sampling_batch
        phase1_extra: Dict[str, Any] = {"pilot_m": float(pilot_m)}
    elif variance_estimation_mode == "independent":
        phase1_fn = _run_independent_phase1_sampling_batch
        phase1_extra = {"independent_n2": int(independent_n2)}
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
                comparison_mode=comparison_mode,
                x_grid=x_grid,
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
