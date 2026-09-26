"""Sibling-pilot Phase 1 for the MMD objective.

For random Fourier features phi with |phi| = 1, the Phase-2 MMD^2 variance is
sum_i w_i / n_i with w_i = Q_{i+1} - Q_i and Q_i = E|E[phi(X_T) | X_{t_i}]|^2.
Two continuations a, b of one state X_{t_i} give E[phi(a).phi(b)] = Q_i, so each
pilot path is continued once more from one anchor level and Q is measured there
directly, with no regression.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import torch

from adaptive import (
    _sample_prior_by_run_batches,
    _sample_segment_by_run_batches,
    _solve_optimal_split_factors,
    _synchronize_runner_device,
    _weighted_pava_non_decreasing,
)
from metrics.mmd import mmd_features
from metrics.utils import coerce_samples_np
from runners.trees import cumulative_profile
from uniform_c import DEFAULT_N0_MIN

MMD_SIBLING = "mmd_sibling"
MMD_SIBLING_DEFAULTS: dict[str, Any] = {
    "estimator": MMD_SIBLING,
    "min_pairs": 5,
    "design_seed": 1,
}


def normalize_mmd_sibling_params(params: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**MMD_SIBLING_DEFAULTS, **dict(params)}
    unknown = sorted(set(merged) - set(MMD_SIBLING_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown mmd_sibling phase1 keys: {unknown}")
    if merged["estimator"] != MMD_SIBLING:
        raise ValueError(f"phase1 estimator must be {MMD_SIBLING!r}")
    for key in ("min_pairs", "design_seed"):
        if isinstance(merged[key], bool) or int(merged[key]) != merged[key]:
            raise ValueError(f"mmd_sibling {key} must be an integer")
        merged[key] = int(merged[key])
    if merged["min_pairs"] < 2:
        raise ValueError("mmd_sibling min_pairs must be at least 2")
    if merged["design_seed"] < 0:
        raise ValueError("mmd_sibling design_seed must be nonnegative")
    return merged


def sibling_geometry(seg_costs, B1: int, min_pairs: int):
    """Most evenly spaced anchor levels (level 0 included) such that every anchor keeps
    ``min_pairs`` sibling pairs within ``B1``.

    Paths are assigned to anchors round-robin; a path anchored at level a costs one full
    path plus the segments from a to T. Returns ``(J, anchors, assignment, used_B1)``.
    """
    costs = np.asarray(seg_costs, dtype=float)
    remaining = np.cumsum(costs[::-1])[::-1]
    full = float(costs.sum())
    for count in range(len(costs), 0, -1):
        anchors = np.unique(np.round(np.linspace(0, len(costs) - 1, count)).astype(int))
        J = 0
        while (J + 1) * full + remaining[
            anchors[np.arange(J + 1) % len(anchors)]
        ].sum() <= B1:
            J += 1
        if J // len(anchors) >= min_pairs:
            assignment = anchors[np.arange(J) % len(anchors)]
            return (
                J,
                anchors,
                assignment,
                int(round(J * full + remaining[assignment].sum())),
            )
    raise ValueError(
        f"B1={B1} is too small for {min_pairs} mmd_sibling pairs at one anchor"
    )


def sibling_trace_profile(
    Y_main: torch.Tensor, Y_branch: torch.Tensor, assignment, anchors, num_levels: int
):
    """``(variance2_per_level[L, 1], tau2[1])`` in the crossfit payload convention.

    Q at each anchor is the mean sibling inner product, anchored below by the U-statistic of
    |P phi|^2 and above by E|phi|^2; values between anchors are interpolated and the
    sequence is made nondecreasing by PAVA with the terminal value pinned.
    """
    Y_main = Y_main.double()
    Y_branch = Y_branch.double()
    J = int(Y_main.shape[0])
    total = Y_main.sum(dim=0)
    lower = max(
        float((total.square().sum() - Y_main.square().sum()) / (J * (J - 1))), 0.0
    )
    top = float(Y_main.square().sum(dim=1).mean())
    pair = (Y_main * Y_branch).sum(dim=1).cpu().numpy()
    q_anchor = np.array([pair[assignment == a].mean() for a in anchors])
    q = np.clip(
        np.append(np.interp(np.arange(num_levels), anchors, q_anchor), top), lower, top
    )
    weights = np.ones_like(q)
    weights[-1] = 1e12
    q = np.clip(_weighted_pava_non_decreasing(q, weights), lower, top)
    q[-1] = top
    return np.diff(q)[:, None], np.array([q[0] - lower])


def floor_for_min_roots(variance2, tau2, seg_costs, B2: int, optimization_mode: str):
    """Keep the design away from a vanishing root term.

    Exact zeros are floored at 1e-12 of the total for every optimizer. For ``monotone``
    the floor is raised, by bisection, to the smallest value at which the relaxed
    allocation still affords ``DEFAULT_N0_MIN`` Phase-2 roots -- the same rail that
    ``max_feasible_c`` puts on ``learned_c``.
    """
    w = np.asarray(variance2, dtype=float)[:, 0].copy()
    w[0] += float(np.asarray(tau2, dtype=float)[0])
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("mmd_sibling trace profile is zero")
    costs = np.asarray(seg_costs, dtype=float)
    cost_w = costs / costs.sum()

    def floored(floor):
        return np.maximum(w, floor)

    def affordable(floor):
        factors = _solve_optimal_split_factors(
            floored(floor)[:, None], np.zeros(1), cost_w, "monotone"
        )
        return DEFAULT_N0_MIN * float(costs @ cumulative_profile(factors)) <= B2

    floor = 1e-12 * total
    if optimization_mode == "monotone" and not affordable(floor):
        lo, hi = np.log(floor), np.log(total)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if affordable(np.exp(mid)):
                hi = mid
            else:
                lo = mid
        floor = float(np.exp(hi))
    return floored(floor)[:, None], np.zeros(1)


def _simulate_sibling_pilot(
    runner,
    *,
    chunk_size,
    J,
    split_points,
    assignment,
    anchors,
    max_sampling_batch_size,
    generator,
):
    """Straight pilot paths plus one branch per path from its anchor level; returns postprocessed endpoints."""
    starts = [runner.start_time] + list(split_points)
    ends = list(split_points) + [runner.end_time]
    x, run_ids = _sample_prior_by_run_batches(
        runner,
        [J] * chunk_size,
        max_sampling_batch_size=max_sampling_batch_size,
        generator=generator,
    )
    anchored = {}
    for level, (start, end) in enumerate(zip(starts, ends)):
        if level in anchors:
            mask = torch.as_tensor(
                np.tile(assignment == level, chunk_size), device=x.device
            )
            anchored[level] = (x[mask].clone(), run_ids[mask], mask)
        x = _sample_segment_by_run_batches(
            runner,
            x,
            run_ids,
            num_runs=chunk_size,
            start_time=start,
            end_time=end,
            max_sampling_batch_size=max_sampling_batch_size,
            generator=generator,
        )
    branch = torch.empty_like(x)
    for level, (xb, ids, mask) in anchored.items():
        for start, end in zip(starts[level:], ends[level:]):
            xb = _sample_segment_by_run_batches(
                runner,
                xb,
                ids,
                num_runs=chunk_size,
                start_time=start,
                end_time=end,
                max_sampling_batch_size=max_sampling_batch_size,
                generator=generator,
            )
        branch[mask] = xb
    return runner.postprocess_samples(x), runner.postprocess_samples(branch)


def run_mmd_sibling_phase1_batch(
    runner,
    *,
    B,
    B1,
    split_percentages,
    phase1: Mapping[str, Any],
    design_state: Mapping[str, Any],
    optimization_mode: str,
    reuse_phase1_samples: bool,
    chunk_size: int,
    max_sampling_batch_size=None,
    generator=None,
):
    """Phase-1 payloads for ``chunk_size`` runs, in the format ``_solve_phase1_allocation`` consumes.

    Both endpoints of every pilot path are exact draws from the discretized law, so the reused
    Phase-1 sample is their concatenation at equal weight.
    """
    params = normalize_mmd_sibling_params(phase1)
    _, split_points = runner.resolve_split_percentages(split_percentages)
    seg_costs = runner.segment_costs(split_points)
    J, anchors, assignment, used_B1 = sibling_geometry(
        seg_costs, int(B1), params["min_pairs"]
    )

    _synchronize_runner_device(runner)
    started = time.perf_counter()
    main, branch = _simulate_sibling_pilot(
        runner,
        chunk_size=int(chunk_size),
        J=J,
        split_points=split_points,
        assignment=assignment,
        anchors=set(int(a) for a in anchors),
        max_sampling_batch_size=max_sampling_batch_size,
        generator=generator,
    )
    _synchronize_runner_device(runner)
    simulation_seconds = (time.perf_counter() - started) / int(chunk_size)

    payloads = []
    for k in range(int(chunk_size)):
        started = time.perf_counter()
        run_main, run_branch = main[k * J : (k + 1) * J], branch[k * J : (k + 1) * J]
        variance2, tau2 = sibling_trace_profile(
            mmd_features(run_main, design_state),
            mmd_features(run_branch, design_state),
            assignment,
            anchors,
            len(seg_costs),
        )
        variance2, tau2 = floor_for_min_roots(
            variance2, tau2, seg_costs, int(B) - used_B1, optimization_mode
        )
        payloads.append(
            {
                "split_points": list(split_points),
                "variance2_per_level": variance2,
                "tau2": tau2,
                "used_B1": used_B1,
                "phase1_x0_samples": (
                    coerce_samples_np(torch.cat([run_main, run_branch]))
                    if reuse_phase1_samples
                    else None
                ),
                "phase1_simulation_seconds": simulation_seconds,
                "query_seconds": 0.0,
                "estimation_seconds": time.perf_counter() - started,
            }
        )
    return payloads
