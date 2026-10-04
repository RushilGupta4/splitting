"""Uniform branching factor: the geometric restriction of the phase-1 problem.

The phase-1 allocation problem solved in ``adaptive.py`` is

    min_a  max_q  sum_l M[q,l] / a_l    s.t.  sum_l w_l a_l = 1,  a nondecreasing

with ``w`` the normalized segment costs and ``M[q,l]`` the per-level variance
increments.  Restricting every split to the same factor, ``a_l = a_0 c^l``,
leaves one scalar:

    V(c) = ( sum_j w_j c^j ) * max_q ( sum_l M[q,l] c^{-l} )

``log V`` is convex in ``u = log c`` (a max of log-sum-exps of affine functions),
so a grid bracket plus golden section finds the global minimum on a box.
"""

from __future__ import annotations

import math

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


class UniformCObjective:
    """``log V(u)`` for one phase-1 instance."""

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
        """``max_q log sum_l M[q,l] e^{-l u}``."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        out = np.empty(u.shape[0], dtype=float)
        block = max(1, _MAX_BLOCK_ENTRIES // max(1, self.log_M.size))
        for start in range(0, u.shape[0], block):
            chunk = u[start : start + block]
            terms = self.log_M[:, :, None] - np.multiply.outer(
                self.levels, chunk
            )[None, :, :]
            out[start : start + block] = np.max(_logsumexp(terms, axis=1), axis=0)
        return out

    def log_value(self, u):
        return self.log_cost(u) + self.log_worst_variance(u)


def solve_uniform_c(
    M: np.ndarray,
    cost_w: np.ndarray,
    *,
    c_min: float = 1.0,
    c_max: float = DEFAULT_C_CAP,
) -> float:
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
        return c_min

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
    winner = int(np.argmin(problem.log_value(candidates)))
    return float(math.exp(float(candidates[winner])))


def max_feasible_c(
    runner,
    split_points,
    *,
    budget: int,
) -> float:
    """Largest uniform ``c`` whose exact tree cost still buys ``DEFAULT_N0_MIN`` roots.

    This bounds the *relaxed* profile. A dyadic type can cost up to twice as
    much, and ``trees.design_mixture`` falls back to the cheapest type if the
    ceiling turns out to be optimistic.
    """
    if len(list(split_points)) == 0:
        return DEFAULT_C_CAP
    budget = int(budget)
    seg_costs = runner.segment_costs(split_points)

    def fits(c: float) -> bool:
        cost = sum(
            float(seg) * (c ** level) for level, seg in enumerate(seg_costs)
        )
        return DEFAULT_N0_MIN * cost <= budget

    if not fits(1.0):
        return 1.0
    if fits(DEFAULT_C_CAP):
        return DEFAULT_C_CAP
    lo, hi = 0.0, math.log(DEFAULT_C_CAP)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fits(math.exp(mid)):
            lo = mid
        else:
            hi = mid
    return float(math.exp(lo))
