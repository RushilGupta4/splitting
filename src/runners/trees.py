"""Mixtures of exact regular integer trees.

The optimizer returns a relaxed cumulative profile ``r``; the simulator needs a
tree in which every parent is copied the *same* whole number of times, because
that is what Lemma "Splitting covariance" assumes.  A profile is realizable
exactly when it is integral and satisfies the divisibility chain
``R_{i-1} | R_i``.

When ``r`` is already such a profile we run it unchanged.  Otherwise we round it
with a single shared shift,

    R_i(U) = 2 ** floor(log2 r_i + U),   U ~ Uniform[0, 1),

which keeps the divisibility chain intact (independent per-level rounding would
not) and yields at most ``L`` distinct trees.  The estimator is the convex
combination of those trees.  Weights come from the finite-query minimax LP when
per-query variances are known, and otherwise from the lengths of the ``U``
intervals -- the latter satisfy

    E_U[C(R_U) K_{R_U}] <= (2 / (e ln 2)) C(r) K_r,

so the cost of insisting on exact integer trees is at most 6.15% of the relaxed
optimum, and the LP can only improve on that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

RHO_BOUND = 2.0 / (math.e * math.log(2.0))


class InsufficientTreeBudgetError(ValueError):
    """Raised when the budget cannot afford one root of the cheapest tree."""


@dataclass(frozen=True)
class TreeType:
    """One exact regular integer tree together with its share of the budget."""

    profile: tuple[int, ...]
    split_factors: tuple[int, ...]
    roots: int
    weight: float
    cost: int


def cumulative_profile(
    split_factors: Sequence[float], *, tol: float = 1e-8
) -> np.ndarray:
    """``[1, N_1, N_1 N_2, ...]`` -- the per-root population at each level.

    The monotone optimizer can return a ratio a hair below one through rounding,
    so factors within ``tol`` of one are clamped rather than rejected; anything
    genuinely below one is an error.
    """
    factors = np.asarray(split_factors, dtype=float)
    if factors.size == 0:
        return np.ones(1, dtype=float)
    if not np.isfinite(factors).all() or np.any(factors < 1.0 - tol):
        raise ValueError("split factors must be finite and at least one")
    return np.concatenate(([1.0], np.cumprod(np.maximum(factors, 1.0))))


def is_exact_integer_tree(profile: Sequence[float], *, tol: float = 1e-9) -> bool:
    """Integral, non-decreasing, starts at one, and divisible level to level."""
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1 or values.size == 0:
        return False
    if np.any(np.abs(values - np.round(values)) > tol):
        return False
    values = np.round(values).astype(np.int64)
    if values[0] != 1 or np.any(np.diff(values) < 0):
        return False
    return bool(np.all(values[1:] % values[:-1] == 0))


def tree_cost(profile: Sequence[float], seg_costs: Sequence[float]) -> int:
    """``C(R) = sum_i c_i R_i``, the cost of simulating one root tree."""
    costs = np.asarray(seg_costs, dtype=float)
    values = np.asarray(profile, dtype=float)
    if costs.shape != values.shape:
        raise ValueError("seg_costs and profile must have the same length")
    return int(round(float(costs @ values)))


def split_factors_of(profile: Sequence[float]) -> tuple[int, ...]:
    values = np.round(np.asarray(profile, dtype=float)).astype(np.int64)
    return tuple(int(v) for v in values[1:] // values[:-1])


def dyadic_types(relaxed_profile: Sequence[float]):
    """Coherent dyadic rounding: distinct trees and their ``U``-interval weights."""
    r = np.asarray(relaxed_profile, dtype=float)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("relaxed profile must be 1D and non-empty")
    if not math.isclose(float(r[0]), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("relaxed profile must be normalized to r_0 = 1")
    if np.any(r < 1.0 - 1e-9) or np.any(np.diff(r) < -1e-9):
        raise ValueError("relaxed profile must be non-decreasing and at least one")

    logs = np.log2(np.maximum(r, 1.0))
    fractions = logs - np.floor(logs)
    edges = [0.0]
    edges += sorted(
        {
            float(np.clip(1.0 - frac, 0.0, 1.0))
            for frac in fractions
            if 1e-12 < frac < 1.0 - 1e-12
        }
    )
    edges.append(1.0)

    profiles: list[np.ndarray] = []
    weights: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        width = right - left
        if width <= 1e-12:
            continue
        profile = np.power(2.0, np.floor(logs + 0.5 * (left + right)))
        for index, existing in enumerate(profiles):
            if np.array_equal(existing, profile):
                weights[index] += width
                break
        else:
            profiles.append(profile)
            weights.append(width)

    total = float(sum(weights))
    return profiles, [w / total for w in weights]


def efficiency_matrix(profiles, variances: np.ndarray, seg_costs) -> np.ndarray:
    """``E[s, q] = C_s * K_s(x_q, x_q)``, cost times per-root variance."""
    variances = np.asarray(variances, dtype=float)
    rows = []
    for profile in profiles:
        values = np.asarray(profile, dtype=float)
        rows.append(
            tree_cost(values, seg_costs) * (variances / values[None, :]).sum(axis=1)
        )
    return np.stack(rows, axis=0)


def solve_minimax_lp(E: np.ndarray) -> np.ndarray:
    """``argmin_lambda max_q sum_s lambda_s E[s, q]`` over the simplex."""
    E = np.asarray(E, dtype=float)
    num_types, num_queries = E.shape
    if num_types == 1:
        return np.ones(1)
    objective = np.zeros(num_types + 1)
    objective[-1] = 1.0
    result = linprog(
        objective,
        A_ub=np.concatenate([E.T, -np.ones((num_queries, 1))], axis=1),
        b_ub=np.zeros(num_queries),
        A_eq=np.concatenate([np.ones((1, num_types)), np.zeros((1, 1))], axis=1),
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * num_types + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"minimax LP failed: {result.message}")
    weights = np.maximum(np.asarray(result.x[:num_types], dtype=float), 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("minimax LP returned a zero weight vector")
    return weights / total


def design_mixture(
    split_factors: Sequence[float],
    seg_costs: Sequence[float],
    budget: int,
    *,
    variances: np.ndarray | None = None,
    max_rounds: int = 16,
) -> list[TreeType]:
    """Realize a relaxed allocation as a budget-weighted mixture of exact trees.

    ``variances`` is the ``(num_queries, num_levels)`` matrix of per-level
    variance contributions.  Supply it whenever it is known -- the learned
    allocations and the OU oracle both have it -- and the weights come from the
    minimax LP.  Without it the coherent-shift interval weights are used, which
    need no information beyond the profile itself.
    """
    seg_costs = np.asarray(seg_costs, dtype=float)
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be positive")

    relaxed = cumulative_profile(split_factors)
    if relaxed.shape != seg_costs.shape:
        raise ValueError("split_factors and seg_costs describe different depths")

    if is_exact_integer_tree(relaxed):
        cost = tree_cost(relaxed, seg_costs)
        roots = budget // cost
        if roots < 1:
            raise InsufficientTreeBudgetError(
                f"budget {budget} cannot afford one root of cost {cost}"
            )
        profile = tuple(int(round(v)) for v in relaxed)
        return [
            TreeType(profile, split_factors_of(relaxed), int(roots), 1.0, cost)
        ]

    profiles, interval_weights = dyadic_types(relaxed)
    costs = np.array([tree_cost(p, seg_costs) for p in profiles], dtype=np.int64)

    def weights_for(active: list[int]) -> np.ndarray:
        if variances is None:
            raw = np.array([interval_weights[i] for i in active], dtype=float)
            return raw / raw.sum()
        return solve_minimax_lp(
            efficiency_matrix([profiles[i] for i in active], variances, seg_costs)
        )

    active = list(range(len(profiles)))
    weights = weights_for(active)
    counts = np.floor(weights * budget / costs[active]).astype(np.int64)
    for _ in range(max_rounds):
        survivors = [
            active[p]
            for p in range(len(active))
            if counts[p] >= 1 and weights[p] > 1e-12
        ]
        if len(survivors) == sum(1 for w in weights if w > 1e-12):
            break
        if not survivors:
            active = [int(np.argmin(costs))]
        else:
            active = survivors
        weights = weights_for(active)
        counts = np.floor(weights * budget / costs[active]).astype(np.int64)

    keep = [p for p in range(len(active)) if counts[p] >= 1 and weights[p] > 1e-12]
    if not keep:
        cheapest = int(np.argmin(costs))
        roots = budget // int(costs[cheapest])
        if roots < 1:
            raise InsufficientTreeBudgetError(
                f"budget {budget} cannot afford one root of the cheapest tree "
                f"(cost {int(costs[cheapest])})"
            )
        active, keep, counts = [cheapest], [0], np.array([roots], dtype=np.int64)

    indices = [active[p] for p in keep]
    roots = counts[keep].astype(np.int64)
    chosen = [profiles[i] for i in indices]
    chosen_costs = costs[indices]

    # Spend whatever the floors left over on whichever type is furthest below
    # its design weight. Same rule for both weightings, no proxy re-evaluation.
    design_weights = weights[keep] / weights[keep].sum()
    spent = int(roots @ chosen_costs)
    while True:
        affordable = [
            i for i in range(len(chosen)) if chosen_costs[i] <= budget - spent
        ]
        if not affordable:
            break
        shortfall = design_weights - (roots * chosen_costs) / max(spent, 1)
        best = max(affordable, key=lambda i: shortfall[i])
        roots[best] += 1
        spent += int(chosen_costs[best])

    shares = roots * chosen_costs
    realized = shares / shares.sum()
    return [
        TreeType(
            tuple(int(round(v)) for v in profile),
            split_factors_of(profile),
            int(count),
            float(weight),
            int(cost),
        )
        for profile, count, weight, cost in zip(chosen, roots, realized, chosen_costs)
    ]


def design_cost(design: Sequence[TreeType]) -> int:
    return int(sum(t.roots * t.cost for t in design))


def mean_split_factors(design: Sequence[TreeType]) -> list[float]:
    """Weight-averaged split factors, for reporting and allocation plots."""
    weights = np.array([t.weight for t in design], dtype=float)
    profiles = np.array([t.profile for t in design], dtype=float)
    mean_profile = np.average(profiles, axis=0, weights=weights)
    return (mean_profile[1:] / mean_profile[:-1]).tolist()
