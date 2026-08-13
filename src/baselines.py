import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Sequence

from tqdm import tqdm

from runners.splitting import max_floor_split_roots_for_budget
from trials import (
    PHASE2_SEED_OFFSET,
    collect_sampling_trial_result_futures,
    iter_run_chunks,
    make_torch_generator,
    submit_sampling_trial_result_futures,
    summarize_sampling_trials,
)

log = logging.getLogger(__name__)


def _run_phase2_loop(
    sample_batch_fn: Callable,
    *,
    runner,
    comparison_mode: str,
    metric_states: Any,
    metrics: Sequence[str],
    n_runs: int,
    n_parallel: int,
    seed: int | None,
    run_offset: int,
):
    """Run n_runs Phase-2 trials. sample_batch_fn(chunk_size, generator) -> samples_by_run."""
    trial_results: list = [None] * n_runs
    workers = max(1, min(int(n_parallel), n_runs))
    futures: list = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start, end in tqdm(
            list(iter_run_chunks(n_runs, n_parallel)), desc="Runs", leave=False
        ):
            chunk_size = end - start
            run_seed = (
                None
                if seed is None
                else seed + PHASE2_SEED_OFFSET + run_offset + start
            )
            samples_by_run = sample_batch_fn(
                chunk_size, make_torch_generator(run_seed, runner.device)
            )
            submit_sampling_trial_result_futures(
                executor=executor,
                futures=futures,
                run_indices=range(start, end),
                samples_by_run=samples_by_run,
                runner=runner,
                comparison_mode=comparison_mode,
                metric_states=metric_states,
                metrics=metrics,
            )
        collect_sampling_trial_result_futures(trial_results, futures)
    return trial_results


def run_solver_baseline_sampling(
    runner,
    comparison_state: Any,
    *,
    comparison_mode: str,
    metrics: Sequence[str] = ("ks",),
    solver: str,
    B: int,
    solver_kwargs: Mapping[str, Any] | None = None,
    n_runs: int,
    seed: int | None,
    debug: bool = False,
    n_parallel: int = 1,
    run_offset: int = 0,
    return_trial_results: bool = False,
    max_sampling_batch_size=None,
):
    solver_kwargs = dict(solver_kwargs or {})
    cost = float(runner.solver_cost(solver, **solver_kwargs))
    if cost <= 0.0:
        raise ValueError(f"runner.solver_cost returned non-positive value {cost}")
    n0 = int(B // cost)
    if n0 < 1:
        raise ValueError(
            f"Budget B={B} is too small for solver baseline with cost {cost}"
        )

    def sample_batch(chunk_size, generator):
        samples, _ = runner.run_solver_baseline_batch(
            solver=solver,
            chunk_size=chunk_size,
            n0=n0,
            generator=generator,
            max_sampling_batch_size=max_sampling_batch_size,
            **solver_kwargs,
        )
        return samples

    trial_results = _run_phase2_loop(
        sample_batch,
        runner=runner,
        comparison_mode=comparison_mode,
        metric_states=comparison_state,
        metrics=metrics,
        n_runs=n_runs,
        n_parallel=n_parallel,
        seed=seed,
        run_offset=run_offset,
    )
    result = {
        "mode": "solver_baseline",
        "B": int(B),
        **summarize_sampling_trials(trial_results, metrics),
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result


def run_fixed_N_sampling(
    runner,
    comparison_state: Any,
    *,
    comparison_mode: str,
    metrics: Sequence[str] = ("ks",),
    B: int,
    split_percentages: Sequence[float],
    N_i_list: Sequence[float],
    n_runs: int,
    seed: int | None,
    debug: bool = False,
    n_parallel: int = 1,
    run_offset: int = 0,
    return_trial_results: bool = False,
    max_sampling_batch_size=None,
    result_mode: str = "fixed_N",
    include_allocation: bool = False,
):
    if split_percentages:
        _, split_points = runner.resolve_split_percentages(split_percentages)
    else:
        split_points = []
    split_factors = [float(x) for x in N_i_list]
    cost_per_root = (
        runner.expected_cost_per_root(split_points, split_factors)
        if split_points
        else runner.segment_cost(runner.start_time, runner.end_time)
    )
    n0 = (
        max_floor_split_roots_for_budget(
            runner,
            split_points,
            split_factors,
            budget=int(B),
            expected_cost_per_root=cost_per_root,
        )
        if split_points
        else int(B // cost_per_root)
    )
    if n0 < 1:
        raise ValueError(
            f"Budget B={B} is too small; expected cost per root is {cost_per_root:.6f}"
        )

    def sample_batch(chunk_size, generator):
        samples, _, _ = runner.run_split_batch(
            n0_by_run=[n0] * chunk_size,
            split_points=split_points,
            split_factors_by_run=[split_factors] * chunk_size,
            generator=generator,
            max_sampling_batch_size=max_sampling_batch_size,
        )
        return samples

    trial_results = _run_phase2_loop(
        sample_batch,
        runner=runner,
        comparison_mode=comparison_mode,
        metric_states=comparison_state,
        metrics=metrics,
        n_runs=n_runs,
        n_parallel=n_parallel,
        seed=seed,
        run_offset=run_offset,
    )
    if include_allocation:
        for trial in trial_results:
            trial["N_i"] = list(split_factors)
            trial["n0"] = int(n0)
    result = {
        "mode": str(result_mode),
        "B": int(B),
        **summarize_sampling_trials(trial_results, metrics),
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result


def run_uniform_c_sampling(
    runner,
    comparison_state: Any,
    *,
    comparison_mode: str,
    metrics: Sequence[str] = ("ks",),
    B: int,
    split_percentages: Sequence[float],
    c: float,
    n_runs: int,
    seed: int | None,
    debug: bool = False,
    n_parallel: int = 1,
    run_offset: int = 0,
    return_trial_results: bool = False,
    max_sampling_batch_size=None,
):
    del debug
    c = float(c)
    if not math.isfinite(c) or c < 1.0:
        raise ValueError("c must be finite and at least 1")
    _, split_points = runner.resolve_split_percentages(split_percentages)
    split_factors = [c] * len(split_points)
    cost_per_root = runner.expected_cost_per_root(split_points, split_factors)
    n0 = max_floor_split_roots_for_budget(
        runner,
        split_points,
        split_factors,
        budget=int(B),
        expected_cost_per_root=cost_per_root,
    )

    def sample_batch(chunk_size, generator):
        samples, _, _ = runner.run_split_batch(
            n0_by_run=[n0] * chunk_size,
            split_points=split_points,
            split_factors_by_run=[split_factors] * chunk_size,
            generator=generator,
            max_sampling_batch_size=max_sampling_batch_size,
        )
        return samples

    trial_results = _run_phase2_loop(
        sample_batch,
        runner=runner,
        comparison_mode=comparison_mode,
        metric_states=comparison_state,
        metrics=metrics,
        n_runs=n_runs,
        n_parallel=n_parallel,
        seed=seed,
        run_offset=run_offset,
    )
    result = {
        "mode": "uniform_c",
        "B": int(B),
        **summarize_sampling_trials(trial_results, metrics),
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result
