"""Uniform branching factor: the geometric restriction of the phase-1 problem.

The phase-1 allocation problem solved in ``adaptive.py`` is

    min_a  max_q  sum_l M[q,l] / a_l    s.t.  sum_l w_l a_l = 1,  a nondecreasing

with ``w`` the normalized segment costs and ``M[q,l]`` the per-level variance
increments.  Restricting every split to the same branching factor means
``a_l = a_0 c^l``; monotonicity becomes ``c >= 1`` and the budget constraint
fixes ``a_0 = 1 / sum_j w_j c^j``, leaving one scalar to choose:

    V(c) = ( sum_j w_j c^j ) * max_q ( sum_l M[q,l] c^{-l} )

In ``u = log c`` both factors are log-sum-exps of affine functions of ``u``, so
each is convex, and the maximum over queries preserves convexity.  ``log V`` is
therefore convex in ``u``: a smooth one-dimensional convex program on a box,
which a grid bracket plus golden section solves to machine precision.  Unlike
the free monotone allocation this needs no Frank--Wolfe iteration and no numba,
so it also runs on CPU-only installations.

Stationarity reads as an elasticity match.  With cost weights
``p_j ∝ w_j e^{ju}`` and variance weights ``r_l ∝ M[q*,l] e^{-lu}`` for the
worst query,

    d/du log V = E_p[level] - E_r[level]

so ``c*`` equates the cost-weighted mean level with the variance-weighted mean
level of that query.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

# Largest branching factor ever considered, before the budget-feasibility rail.
DEFAULT_C_CAP = 4.0
# The rail keeps at least this many phase-2 roots: the objective is a continuum
# approximation, and flooring the path count at every split makes it a poor
# guide once only a handful of roots remain.
DEFAULT_N0_MIN = 32

_GRID_POINTS = 193
_GOLDEN_ITERS = 128
_MAX_BLOCK_ENTRIES = 4_000_000
_NEG_INF = -np.inf


def prepare_minimax_inputs(variance2_per_level, tau2, cost_weights):
    """Return ``(M, cost_w)`` in the same form the free solver builds them."""
    variance2_per_level = np.asarray(variance2_per_level, dtype=float)
    tau2 = np.asarray(tau2, dtype=float)
    if variance2_per_level.ndim != 2:
        raise ValueError("variance2_per_level must be (num_levels, grid_pts)")
    num_levels, grid_pts = variance2_per_level.shape
    if tau2.shape != (grid_pts,):
        raise ValueError(f"tau2 shape {tau2.shape} != ({grid_pts},)")
    if not np.isfinite(variance2_per_level).all() or not np.isfinite(tau2).all():
        raise ValueError("variance/tau2 contain non-finite values")

    cost_w = np.asarray(cost_weights, dtype=float)
    if cost_w.shape != (num_levels,):
        raise ValueError(f"cost_weights shape {cost_w.shape} != ({num_levels},)")
    if not np.isfinite(cost_w).all() or np.any(cost_w <= 0.0):
        raise ValueError("cost_weights must be finite and positive")
    cost_w = cost_w / float(cost_w.sum())

    M = np.maximum(variance2_per_level, 0.0).T.copy()
    M[:, 0] += np.maximum(tau2, 0.0)
    return M, cost_w


def _log_or_neg_inf(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, _NEG_INF, dtype=float)
    positive = values > 0.0
    out[positive] = np.log(values[positive])
    return out


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(peak)
    shifted = np.where(finite, values - np.where(finite, peak, 0.0), _NEG_INF)
    total = np.sum(np.exp(np.where(np.isfinite(shifted), shifted, _NEG_INF)), axis=axis)
    result = np.squeeze(peak, axis=axis) + np.log(np.where(total > 0.0, total, 1.0))
    return np.where(np.squeeze(finite, axis=axis), result, _NEG_INF)


def _softmax(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, 1.0 / values.size)
    shifted = np.where(finite, values - np.max(values[finite]), _NEG_INF)
    weights = np.where(finite, np.exp(shifted), 0.0)
    return weights / float(weights.sum())


class UniformCObjective:
    """``log V(u)`` and its derivative for one phase-1 instance."""

    def __init__(self, M: np.ndarray, cost_w: np.ndarray):
        M = np.ascontiguousarray(np.maximum(np.asarray(M, dtype=float), 0.0))
        cost_w = np.asarray(cost_w, dtype=float).reshape(-1)
        if M.ndim != 2 or M.shape[1] != cost_w.shape[0]:
            raise ValueError("M must be (num_queries, num_levels) matching cost_w")
        self.M = M
        self.cost_w = cost_w / float(cost_w.sum())
        self.num_levels = int(M.shape[1])
        self.levels = np.arange(self.num_levels, dtype=float)
        self.log_w = _log_or_neg_inf(self.cost_w)
        self.log_M = _log_or_neg_inf(self.M)
        self.degenerate = bool(np.all(~np.isfinite(self.log_M)))

    def log_cost(self, u):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        terms = self.log_w[None, :] + np.outer(u, self.levels)
        return _logsumexp(terms, axis=1)

    def log_worst_variance(self, u):
        """``max_q log sum_l M[q,l] e^{-l u}`` and the maximizing query."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        out = np.empty(u.shape[0], dtype=float)
        arg = np.zeros(u.shape[0], dtype=np.int64)
        block = max(1, _MAX_BLOCK_ENTRIES // max(1, self.log_M.size))
        for start in range(0, u.shape[0], block):
            chunk = u[start : start + block]
            terms = self.log_M[:, :, None] - np.multiply.outer(
                self.levels, chunk
            )[None, :, :]
            per_query = _logsumexp(terms, axis=1)
            arg[start : start + block] = np.argmax(per_query, axis=0)
            out[start : start + block] = np.max(per_query, axis=0)
        return out, arg

    def log_value(self, u):
        worst, _ = self.log_worst_variance(u)
        return self.log_cost(u) + worst

    def derivative(self, u: float) -> float:
        """``d/du log V`` at the active query."""
        u = float(u)
        cost_weights = _softmax(self.log_w + u * self.levels)
        per_query = _logsumexp(self.log_M - u * self.levels[None, :], axis=1)
        active = int(np.argmax(per_query))
        var_weights = _softmax(self.log_M[active] - u * self.levels)
        return float(cost_weights @ self.levels - var_weights @ self.levels)

    def value(self, c: float) -> float:
        return float(np.exp(self.log_value(math.log(float(c)))[0]))


def solve_uniform_c(
    M: np.ndarray,
    cost_w: np.ndarray,
    *,
    c_min: float = 1.0,
    c_max: float = DEFAULT_C_CAP,
) -> Dict[str, Any]:
    """Globally minimize ``max_q V_q(c)`` over ``c`` in ``[c_min, c_max]``.

    The objective is convex in ``log c``, so the coarse grid brackets the
    minimum and golden section refines it; the returned point is global.
    """
    c_min = float(c_min)
    c_max = float(c_max)
    if not (0.0 < c_min <= c_max) or not math.isfinite(c_max):
        raise ValueError(f"require 0 < c_min <= c_max < inf, got [{c_min}, {c_max}]")

    problem = UniformCObjective(M, cost_w)
    if problem.degenerate or problem.num_levels == 1 or c_max <= c_min:
        return {
            "c": c_min,
            "log_value": float(problem.log_value(math.log(c_min))[0]),
            "at_lower_bound": True,
            "at_upper_bound": bool(c_max <= c_min),
            "derivative": float("nan"),
        }

    lo, hi = math.log(c_min), math.log(c_max)
    grid = np.linspace(lo, hi, _GRID_POINTS)
    values = problem.log_value(grid)
    best = int(np.argmin(values))
    left = grid[max(best - 1, 0)]
    right = grid[min(best + 1, _GRID_POINTS - 1)]

    inv_phi = (math.sqrt(5.0) - 1.0) * 0.5
    mid_left = right - inv_phi * (right - left)
    mid_right = left + inv_phi * (right - left)
    value_left = float(problem.log_value(mid_left)[0])
    value_right = float(problem.log_value(mid_right)[0])
    for _ in range(_GOLDEN_ITERS):
        if right - left <= 1e-13 * max(1.0, abs(right)):
            break
        if value_left <= value_right:
            right, mid_right, value_right = mid_right, mid_left, value_left
            mid_left = right - inv_phi * (right - left)
            value_left = float(problem.log_value(mid_left)[0])
        else:
            left, mid_left, value_left = mid_left, mid_right, value_right
            mid_right = left + inv_phi * (right - left)
            value_right = float(problem.log_value(mid_right)[0])

    candidates = np.array([lo, hi, 0.5 * (left + right)], dtype=float)
    candidate_values = problem.log_value(candidates)
    winner = int(np.argmin(candidate_values))
    u_star = float(candidates[winner])
    return {
        "c": float(math.exp(u_star)),
        "log_value": float(candidate_values[winner]),
        "at_lower_bound": bool(u_star <= lo + 1e-9),
        "at_upper_bound": bool(u_star >= hi - 1e-9),
        "derivative": problem.derivative(u_star),
    }


def allocation_from_factors(factors, num_levels: int) -> np.ndarray:
    factors = [float(value) for value in factors]
    if len(factors) != num_levels - 1:
        raise ValueError("factors must have num_levels - 1 entries")
    allocation = np.ones(num_levels, dtype=float)
    for idx, factor in enumerate(factors):
        allocation[idx + 1] = allocation[idx] * factor
    return allocation


def minimax_objective(M: np.ndarray, cost_w: np.ndarray, factors) -> float:
    """``max_q sum_l M[q,l]/a_l`` for the budget-normalized allocation."""
    M = np.asarray(M, dtype=float)
    cost_w = np.asarray(cost_w, dtype=float)
    cost_w = cost_w / float(cost_w.sum())
    allocation = allocation_from_factors(factors, int(M.shape[1]))
    allocation = allocation / float(cost_w @ allocation)
    return float(np.max(M @ (1.0 / allocation)))


def max_feasible_c(
    runner,
    split_points,
    *,
    budget: int,
    n0_min: int = DEFAULT_N0_MIN,
    c_cap: float = DEFAULT_C_CAP,
) -> float:
    """Largest uniform ``c`` whose exact tree cost still buys ``n0_min`` roots.

    This bounds the *relaxed* profile. A dyadic type can cost up to twice as
    much, and ``trees.design_mixture`` falls back to the cheapest type if the
    ceiling turns out to be optimistic.
    """

    num_splits = len(list(split_points))
    if num_splits == 0:
        return float(c_cap)
    budget = int(budget)
    n0_min = max(1, int(n0_min))

    seg_costs = runner.segment_costs(split_points)

    def fits(c: float) -> bool:
        cost = sum(
            float(seg) * (c ** level) for level, seg in enumerate(seg_costs)
        )
        return n0_min * cost <= budget

    if not fits(1.0):
        return 1.0
    if fits(c_cap):
        return float(c_cap)
    lo, hi = 0.0, math.log(c_cap)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fits(math.exp(mid)):
            lo = mid
        else:
            hi = mid
    return float(math.exp(lo))
