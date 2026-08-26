import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Sequence

from tqdm import tqdm

from runners.splitting import run_mixture_batch
from runners.trees import design_cost, design_mixture, mean_split_factors
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
    weights=None,
):
    """Run n_runs Phase-2 trials.

    ``sample_batch_fn(chunk_size, generator)`` returns, per run, the list of
    sample blocks making up that run's estimator; ``weights`` are their mixture
    weights (``None`` means equal weight per observation).
    """
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
                weights_by_run=[weights] * len(samples_by_run),
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
        return [[sample] for sample in samples]

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
    max_paths_in_flight=None,
    result_mode: str = "fixed_N",
    include_allocation: bool = False,
    variances=None,
):
    if split_percentages:
        _, split_points = runner.resolve_split_percentages(split_percentages)
    else:
        split_points = []
    split_factors = [float(x) for x in N_i_list]
    design = None
    if split_points:
        # `variances` decides the weighting: the OU oracle supplies its analytic
        # per-query matrix and gets the minimax LP; uniform_c has no variance
        # information and falls back to the coherent-shift interval weights.
        design = design_mixture(
            split_factors,
            runner.segment_costs(split_points),
            int(B),
            variances=variances,
        )
        n0 = sum(tree.roots for tree in design)
    else:
        full_cost = runner.segment_cost(runner.start_time, runner.end_time)
        n0 = int(B // full_cost)
        if n0 < 1:
            raise ValueError(
                f"Budget B={B} is too small; one path costs {full_cost:.6f}"
            )

    def sample_batch(chunk_size, generator):
        if design is None:
            samples, _, _ = runner.run_split_batch(
                n0_by_run=[n0] * chunk_size,
                split_points=[],
                split_factors_by_run=[[] for _ in range(chunk_size)],
                generator=generator,
                max_sampling_batch_size=max_sampling_batch_size,
            )
            return [[sample] for sample in samples]
        return run_mixture_batch(
            runner,
            designs_by_run=[design] * chunk_size,
            split_points=split_points,
            generator=generator,
            max_sampling_batch_size=max_sampling_batch_size,
            max_paths_in_flight=max_paths_in_flight,
        )

    weights = None if design is None else [tree.weight for tree in design]

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
        weights=weights,
    )
    if include_allocation:
        realized = mean_split_factors(design) if design else list(split_factors)
        for trial in trial_results:
            trial["N_i"] = list(realized)
            trial["n0"] = int(n0)
    result = {
        "mode": str(result_mode),
        "B": int(B),
        **summarize_sampling_trials(trial_results, metrics),
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result
