import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Sequence

import torch
from tqdm import tqdm

from diffusion import (
    DDIM,
    expected_cost_per_root,
    resolve_split_percentages,
    run_probabilistic_inference_batch,
    run_solver_baseline_batch,
)
from trials import (
    PHASE2_SEED_OFFSET,
    collect_sampling_trial_result_futures,
    iter_run_chunks,
    make_torch_generator,
    submit_sampling_trial_result_futures,
    summarize_sampling_trials,
    warm_sampling_ks,
)

log = logging.getLogger(__name__)


def _run_phase2_loop(
    sample_batch_fn: Callable,
    *,
    target_spec: Dict[str, Any],
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    n_runs: int,
    n_parallel: int,
    seed: int | None,
    run_start_index: int,
    device: str,
):
    """Run n_runs Phase-2 trials. sample_batch_fn(chunk_size, generator) -> samples_by_run."""
    trial_results: list = [None] * n_runs
    warm_sampling_ks(reference_mode, target_spec)
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
                else seed + PHASE2_SEED_OFFSET + run_start_index + start
            )
            samples_by_run = sample_batch_fn(
                chunk_size, make_torch_generator(run_seed, device)
            )
            submit_sampling_trial_result_futures(
                executor=executor,
                futures=futures,
                run_indices=range(start, end),
                samples_by_run=samples_by_run,
                target_spec=target_spec,
                reference_cdf_state=reference_cdf_state,
                reference_mode=reference_mode,
            )
        collect_sampling_trial_result_futures(trial_results, futures)
    return trial_results


def run_solver_baseline_sampling(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    solver: str,
    B: int,
    T: int,
    sampling_steps: int,
    eta: float = 0.0,
    n_runs: int,
    seed: int | None,
    device: str,
    reference_mode: str = "ddpm_samples",
    debug: bool = False,
    n_parallel: int = 1,
    run_start_index: int = 0,
    return_trial_results: bool = False,
):
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be at least 1")
    n0 = int(B // sampling_steps)
    if n0 < 1:
        raise ValueError(
            f"Budget B={B} is too small for solver baseline with {sampling_steps} steps"
        )

    def sample_batch(chunk_size, generator):
        samples, _ = run_solver_baseline_batch(
            model=model,
            data_mean=data_mean,
            data_std=data_std,
            solver=solver,
            chunk_size=chunk_size,
            n0=n0,
            T=T,
            sampling_steps=sampling_steps,
            eta=eta,
            device=device,
            generator=generator,
        )
        return samples

    trial_results = _run_phase2_loop(
        sample_batch,
        target_spec=target_spec,
        reference_cdf_state=reference_cdf_state,
        reference_mode=reference_mode,
        n_runs=n_runs,
        n_parallel=n_parallel,
        seed=seed,
        run_start_index=run_start_index,
        device=device,
    )
    result = {
        "mode": "solver_baseline",
        "B": int(B),
        **summarize_sampling_trials(trial_results),
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result


def run_fixed_N_sampling(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    B: int,
    T: int,
    sampling_steps: int,
    eta: float,
    split_percentages: Sequence[float],
    N_i_list: Sequence[float],
    n_runs: int,
    seed: int | None,
    device: str,
    reference_mode: str = "ddpm_samples",
    debug: bool = False,
    n_parallel: int = 1,
    run_start_index: int = 0,
    return_trial_results: bool = False,
):
    ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
    _, split_points = resolve_split_percentages(ddim, split_percentages)
    split_factors = [float(x) for x in N_i_list]
    cost_per_root = expected_cost_per_root(ddim, split_points, split_factors)
    n0 = int(B // cost_per_root)
    if n0 < 1:
        raise ValueError(
            f"Budget B={B} is too small; expected cost per root is {cost_per_root:.6f}"
        )

    def sample_batch(chunk_size, generator):
        samples, _, _ = run_probabilistic_inference_batch(
            model=model,
            ddim=ddim,
            n0_by_run=[n0] * chunk_size,
            split_points=split_points,
            split_factors_by_run=[split_factors] * chunk_size,
            data_mean=data_mean,
            data_std=data_std,
            generator=generator,
        )
        return samples

    trial_results = _run_phase2_loop(
        sample_batch,
        target_spec=target_spec,
        reference_cdf_state=reference_cdf_state,
        reference_mode=reference_mode,
        n_runs=n_runs,
        n_parallel=n_parallel,
        seed=seed,
        run_start_index=run_start_index,
        device=device,
    )
    result = {
        "mode": "fixed_N",
        "B": int(B),
        **summarize_sampling_trials(trial_results),
    }
    if return_trial_results:
        result["trial_results"] = trial_results
    return result
