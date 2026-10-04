from typing import Any, Mapping, Sequence

from runners.splitting import run_mixture_batch
from runners.trees import design_mixture, mean_split_factors
from trials import run_phase2


def _run_without_pilot(sample_batch_fn, weights, **phase2):
    """Phase-2 trials of a method without a pilot: all sampling is "phase 2"."""

    def sample_batch(start, end, generator):
        samples_by_run = sample_batch_fn(end - start, generator)
        return samples_by_run, [weights] * len(samples_by_run)

    trial_results, seconds = run_phase2(sample_batch, **phase2)
    for trial, run_seconds in zip(trial_results, seconds):
        trial["phase2_seconds"] = float(run_seconds)
        trial["total_seconds"] = float(run_seconds)
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
    n_parallel: int = 1,
    run_offset: int = 0,
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

    return _run_without_pilot(
        sample_batch,
        None,
        runner=runner,
        comparison_mode=comparison_mode,
        metric_states=comparison_state,
        metrics=metrics,
        n_runs=n_runs,
        n_parallel=n_parallel,
        seed=seed,
        run_offset=run_offset,
    )


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
    n_parallel: int = 1,
    run_offset: int = 0,
    max_sampling_batch_size=None,
    max_paths_in_flight=None,
    variances=None,
):
    """``fixed_N`` without split points; otherwise a mixture design for the given
    factors (uniform_c, the OU oracle), whose trials carry the realized ``N_i``."""
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

    trial_results = _run_without_pilot(
        sample_batch,
        None if design is None else [tree.weight for tree in design],
        runner=runner,
        comparison_mode=comparison_mode,
        metric_states=comparison_state,
        metrics=metrics,
        n_runs=n_runs,
        n_parallel=n_parallel,
        seed=seed,
        run_offset=run_offset,
    )
    if design is not None:
        realized = mean_split_factors(design)
        for trial in trial_results:
            trial["N_i"] = list(realized)
            trial["n0"] = int(n0)
    return trial_results
