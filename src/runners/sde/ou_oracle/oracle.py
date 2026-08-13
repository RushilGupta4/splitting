"""Finite-query minimax oracle for the Euler-discretized 2D OU benchmark.

The query set contains a dense Cartesian grid of finite lower-orthant
thresholds and both one-dimensional boundary families.  Gaussian CDFs give
each query's exact Euler-chain variance profile up to numerical CDF integration
error.  A local copy of the monotone Frank--Wolfe solver from
``src/adaptive.py`` then optimizes over the convex hull of those profiles, as
required by the minimax lemma.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy.stats import multivariate_normal, norm

REFERENCE_SCHEDULE = tuple(round(value / 20, 2) for value in range(19, 0, -1))
QUERY_GRID_SIZE = 320
QUERY_PROBABILITY_MIN = 0.01
QUERY_PROBABILITY_MAX = 0.99
QUERY_PROBABILITIES = tuple(
    float(value)
    for value in np.linspace(
        QUERY_PROBABILITY_MIN,
        QUERY_PROBABILITY_MAX,
        QUERY_GRID_SIZE,
    )
)
QUERY_COUNT = QUERY_GRID_SIZE**2 + 2 * QUERY_GRID_SIZE
CDF_SEED = 20260812
CDF_MAX_POINTS = 250_000
CDF_TOLERANCE = 2e-6
CDF_QUERY_BATCH_SIZE = 4_096
DEFAULT_RELATIVE_TOLERANCE = 1e-7
DEFAULT_MAX_ITERATIONS = 2_000

OU_RATES = (1.35, 1.60)
OU_SIGMA = 0.65


# This optimizer block is copied from src/adaptive.py so the paper scripts are
# standalone.  It additionally retains the convex-combination profile and the
# final primal-dual gap, which the production allocator does not need to expose.
def _objective_values(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    return M @ (1.0 / y)


def _weighted_pava_non_decreasing(
    values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
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

    blocks: list[list[float | int]] = []
    for idx, (value, weight) in enumerate(zip(values, weights)):
        weighted_sum = float(weight * value)
        blocks.append([idx, idx + 1, float(weight), weighted_sum, float(value)])
        while len(blocks) >= 2 and blocks[-2][4] > blocks[-1][4]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left[2]) + float(right[2])
            merged_sum = float(left[3]) + float(right[3])
            blocks.append(
                [
                    int(left[0]),
                    int(right[1]),
                    merged_weight,
                    merged_sum,
                    merged_sum / merged_weight,
                ]
            )

    projected = np.empty_like(values, dtype=float)
    for start, end, _, _, value in blocks:
        projected[int(start) : int(end)] = float(value)
    return projected


def _monotone_simplex_from_profile(
    profile: np.ndarray,
    cost_w: np.ndarray,
    *,
    allocation_floor: float = 1e-12,
) -> np.ndarray:
    profile = np.maximum(np.asarray(profile, dtype=float), 0.0)
    cost_w = np.asarray(cost_w, dtype=float)
    h = _weighted_pava_non_decreasing(profile / cost_w, cost_w)
    raw = np.sqrt(np.maximum(h, 0.0))
    if not np.isfinite(raw).all() or float(np.max(raw)) <= 0.0:
        raw = np.ones_like(cost_w, dtype=float)
    raw = np.maximum(raw, float(allocation_floor))
    allocation = raw / float(cost_w @ raw)
    return cost_w * allocation


def _monotone_dual_value_and_simplex(
    profile: np.ndarray, cost_w: np.ndarray
) -> tuple[float, np.ndarray]:
    profile = np.maximum(np.asarray(profile, dtype=float), 0.0)
    y = _monotone_simplex_from_profile(profile, cost_w)
    value = float((profile * cost_w) @ (1.0 / y))
    return value, y


def _monotone_line_search(
    profile: np.ndarray,
    target_profile: np.ndarray,
    cost_w: np.ndarray,
    *,
    max_iters: int = 48,
) -> tuple[float, float, np.ndarray]:
    profile = np.asarray(profile, dtype=float)
    target_profile = np.asarray(target_profile, dtype=float)
    if np.allclose(profile, target_profile, rtol=0.0, atol=0.0):
        value, y = _monotone_dual_value_and_simplex(profile, cost_w)
        return 0.0, value, y

    def value_at(gamma: float) -> tuple[float, np.ndarray]:
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
    relative_tol: float = DEFAULT_RELATIVE_TOLERANCE,
    max_iters: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, float]:
    M = np.asarray(M, dtype=float)
    num_points, num_levels = M.shape
    if num_points == 0 or num_levels == 0:
        raise ValueError("M must be non-empty")

    weighted_M = M * cost_w[None, :]
    row_scores = np.sqrt(np.maximum(weighted_M, 0.0)).sum(axis=1)
    profile = np.maximum(M[int(np.argmax(row_scores))].copy(), 0.0)

    best_y = np.asarray(cost_w, dtype=float).copy()
    best_profile = profile.copy()
    best_upper = math.inf
    for iter_idx in range(max_iters):
        dual_lower, y = _monotone_dual_value_and_simplex(profile, cost_w)
        full_values = _objective_values(weighted_M, y)
        full_upper = float(np.max(full_values))
        if full_upper < best_upper:
            best_upper = full_upper
            best_y = y.copy()
            best_profile = profile.copy()
        gap = max(full_upper - dual_lower, 0.0) / max(abs(full_upper), 1.0)
        if gap <= relative_tol:
            return y, profile, gap

        worst_idx = int(np.argmax(full_values))
        target_profile = np.maximum(M[worst_idx], 0.0)
        gamma, candidate_lower, candidate_y = _monotone_line_search(
            profile, target_profile, cost_w
        )
        if gamma <= 0.0 or candidate_lower <= dual_lower + 1e-14:
            step = 2.0 / float(iter_idx + 3.0)
            profile = (1.0 - step) * profile + step * target_profile
        else:
            candidate_profile = (1.0 - gamma) * profile + gamma * target_profile
            profile = candidate_profile
            if candidate_lower > dual_lower:
                candidate_values = _objective_values(weighted_M, candidate_y)
                candidate_upper = float(np.max(candidate_values))
                if candidate_upper < best_upper:
                    best_upper = candidate_upper
                    best_y = candidate_y.copy()
                    best_profile = candidate_profile.copy()

    best_lower, _ = _monotone_dual_value_and_simplex(best_profile, cost_w)
    best_gap = max(best_upper - best_lower, 0.0) / max(abs(best_upper), 1.0)
    return best_y, best_profile, best_gap


def _mode_variance(rate: float, step: int, *, steps: int) -> float:
    delta = 1.0 / steps
    decay = 1.0 - rate * delta
    decay_power = decay ** (2 * step)
    return decay_power + OU_SIGMA**2 * delta * (1.0 - decay_power) / (
        1.0 - decay**2
    )


def _coordinate_covariance(common: float, difference: float) -> np.ndarray:
    diagonal = 0.5 * (common + difference)
    off_diagonal = 0.5 * (common - difference)
    return np.asarray(
        [[diagonal, off_diagonal], [off_diagonal, diagonal]], dtype=float
    )


def _multivariate_normal_cdf_batched(
    points: np.ndarray,
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate Gaussian CDFs in bounded batches using one RNG stream."""
    points = np.asarray(points, dtype=float)
    values = np.empty(points.shape[0], dtype=float)
    for start in range(0, points.shape[0], CDF_QUERY_BATCH_SIZE):
        end = min(start + CDF_QUERY_BATCH_SIZE, points.shape[0])
        batch = multivariate_normal.cdf(
            points[start:end],
            mean=mean,
            cov=covariance,
            maxpts=CDF_MAX_POINTS,
            abseps=CDF_TOLERANCE,
            releps=CDF_TOLERANCE,
            rng=rng,
        )
        values[start:end] = np.asarray(batch, dtype=float).reshape(-1)
    return values


@functools.lru_cache(maxsize=None)
def _query_q_values(
    steps: int = 280,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return terminal probabilities and shared-descendant Q values."""
    terminal_modes = tuple(_mode_variance(rate, steps, steps=steps) for rate in OU_RATES)
    terminal_covariance = _coordinate_covariance(*terminal_modes)
    terminal_std = math.sqrt(float(terminal_covariance[0, 0]))

    probabilities = np.asarray(QUERY_PROBABILITIES, dtype=float)
    z_values = norm.ppf(probabilities)
    finite_z = np.asarray(
        [(first, second) for first in z_values for second in z_values],
        dtype=float,
    )
    finite_thresholds = terminal_std * finite_z
    rng = np.random.default_rng(CDF_SEED)
    finite_terminal_probabilities = _multivariate_normal_cdf_batched(
        finite_thresholds,
        mean=np.zeros(2),
        covariance=terminal_covariance,
        rng=rng,
    )
    terminal_probabilities = np.concatenate(
        (finite_terminal_probabilities, probabilities, probabilities)
    )

    split_steps = np.asarray(
        [round((1.0 - split) * steps) for split in REFERENCE_SCHEDULE],
        dtype=int,
    )
    q_values = np.empty((terminal_probabilities.size, split_steps.size), dtype=float)
    paired_finite_thresholds = np.column_stack((finite_thresholds, finite_thresholds))
    paired_marginal_thresholds = np.column_stack((z_values, z_values))

    delta = 1.0 / steps
    for column, split_step in enumerate(split_steps):
        retained_modes = []
        for rate in OU_RATES:
            decay = 1.0 - rate * delta
            retained_modes.append(
                decay ** (2 * (steps - int(split_step)))
                * _mode_variance(rate, int(split_step), steps=steps)
            )
        retained_covariance = _coordinate_covariance(*retained_modes)
        descendant_covariance = np.block(
            [
                [terminal_covariance, retained_covariance],
                [retained_covariance, terminal_covariance],
            ]
        )
        finite_q = _multivariate_normal_cdf_batched(
            paired_finite_thresholds,
            mean=np.zeros(4),
            covariance=descendant_covariance,
            rng=rng,
        )

        marginal_correlation = float(
            retained_covariance[0, 0] / terminal_covariance[0, 0]
        )
        marginal_q = _multivariate_normal_cdf_batched(
            paired_marginal_thresholds,
            mean=np.zeros(2),
            covariance=np.asarray(
                [[1.0, marginal_correlation], [marginal_correlation, 1.0]]
            ),
            rng=rng,
        )
        q_values[:, column] = np.concatenate((finite_q, marginal_q, marginal_q))

    if not np.isfinite(terminal_probabilities).all() or not np.isfinite(q_values).all():
        raise RuntimeError("Non-finite Gaussian CDF in OU minimax query matrix")
    return split_steps, terminal_probabilities, q_values


@functools.lru_cache(maxsize=None)
def query_variance_contributions(
    schedule: tuple[float, ...], *, steps: int = 280
) -> np.ndarray:
    """Return one segmentwise variance vector per finite KS query."""
    split_steps, terminal_probabilities, all_q_values = _query_q_values(steps)
    requested_steps = [round((1.0 - split) * steps) for split in schedule]
    column_by_step = {int(step): column for column, step in enumerate(split_steps)}
    try:
        columns = [column_by_step[step] for step in requested_steps]
    except KeyError as exc:
        raise ValueError(
            "OU oracle schedules must be subsets of the 19-split reference schedule"
        ) from exc

    sequence = np.column_stack(
        (
            terminal_probabilities**2,
            all_q_values[:, columns],
            terminal_probabilities,
        )
    )
    lower = terminal_probabilities[:, None] ** 2
    upper = terminal_probabilities[:, None]
    sequence = np.clip(sequence, lower, upper)
    sequence = np.maximum.accumulate(sequence, axis=1)
    sequence[:, -1] = terminal_probabilities
    return np.maximum(np.diff(sequence, axis=1), 0.0)


@functools.lru_cache(maxsize=None)
def _minimax_solution(
    schedule: tuple[float, ...], *, steps: int = 280
) -> tuple[np.ndarray, np.ndarray, float]:
    contributions = query_variance_contributions(schedule, steps=steps)
    split_steps = np.asarray(
        [round((1.0 - split) * steps) for split in schedule], dtype=float
    )
    cost_w = np.diff(np.concatenate(([0.0], split_steps, [float(steps)])))
    cost_w /= float(cost_w.sum())
    simplex, profile, relative_gap = _solve_frank_wolfe_monotone_allocation(
        contributions,
        cost_w=cost_w,
    )
    allocation = simplex / cost_w
    allocation /= float(cost_w @ allocation)
    cumulative = allocation / allocation[0]
    return cumulative, profile, relative_gap


def ou_oracle_allocation(
    schedule: tuple[float, ...], *, steps: int = 280
) -> np.ndarray:
    """Return the finite-query minimax cumulative allocation."""
    allocation, _, _ = _minimax_solution(schedule, steps=steps)
    return allocation.copy()


def ou_oracle_variance_contributions(
    schedule: tuple[float, ...], *, steps: int = 280
) -> np.ndarray:
    """Return the least-favorable convex-mixture variance profile."""
    _, profile, _ = _minimax_solution(schedule, steps=steps)
    return profile.copy()


def ou_oracle_relative_gap(
    schedule: tuple[float, ...], *, steps: int = 280
) -> float:
    """Return the finite-query Frank--Wolfe relative primal-dual gap."""
    _, _, relative_gap = _minimax_solution(schedule, steps=steps)
    return float(relative_gap)
