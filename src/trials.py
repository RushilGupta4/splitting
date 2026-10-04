import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from metrics import compute_trial_metrics

PHASE2_SEED_OFFSET = 1_000_000


def make_torch_generator(seed: int | None, device: str):
    if seed is None:
        return None
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def synchronize(device) -> None:
    """Wait for queued GPU work so wall-clock stage timers are accurate."""
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def iter_run_chunks(n_runs: int, n_parallel: int):
    if n_runs < 1 or n_parallel < 1:
        raise ValueError("n_runs and n_parallel must be at least 1")
    for start in range(0, n_runs, n_parallel):
        yield start, min(start + n_parallel, n_runs)


def build_sampling_trial_result(
    parts,
    runner,
    comparison_mode: str,
    metric_states,
    metrics,
    part_weights=None,
):
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    parts = [p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in parts]

    synchronize(runner.device)
    started = time.perf_counter()
    metric_values, metric_payloads = compute_trial_metrics(
        parts,
        runner,
        comparison_mode=comparison_mode,
        metric_states=metric_states,
        metrics=metrics,
        part_weights=part_weights,
    )

    synchronize(runner.device)
    result: Dict[str, Any] = {
        "metrics": metric_values,
        # Evaluation cost (decoding, embedding, MMD), not part of the method's cost.
        "scoring_seconds": time.perf_counter() - started,
    }
    if metric_payloads:
        result["metric_payloads"] = metric_payloads
    return result


def submit_sampling_trial_result_futures(
    executor: ThreadPoolExecutor,
    futures: List[Tuple[int, Any]],
    run_indices: Sequence[int],
    samples_by_run: Sequence,
    runner,
    comparison_mode: str,
    metric_states,
    metrics,
    *,
    weights_by_run: Sequence | None = None,
):
    run_indices = list(run_indices)
    if weights_by_run is None:
        weights_by_run = [None] * len(samples_by_run)
    if len(samples_by_run) != len(run_indices):
        raise ValueError("run_indices must match samples_by_run length")
    for local_idx, run_idx in enumerate(run_indices):
        future = executor.submit(
            build_sampling_trial_result,
            samples_by_run[local_idx],
            runner,
            comparison_mode,
            metric_states,
            metrics,
            weights_by_run[local_idx],
        )
        if metric_states.get("serial_scoring"):
            future.result()
        futures.append((int(run_idx), future))


def run_phase2(
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
    desc: str = "Runs",
):
    """Sample and score ``n_runs`` Phase-2 runs in chunks of ``n_parallel``.

    ``sample_batch_fn(start, end, generator)`` returns ``(parts_by_run,
    weights_by_run)`` for runs ``start:end``: each run's sample blocks and their
    mixture weights (``None`` weights every observation equally). Returns the
    per-run metric results and per-run sampling seconds.
    """
    results: list = [None] * n_runs
    seconds: list = [None] * n_runs
    futures: list = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(n_parallel), n_runs))) as executor:
        for start, end in tqdm(
            list(iter_run_chunks(n_runs, n_parallel)), desc=desc, leave=False
        ):
            run_seed = (
                None if seed is None else seed + PHASE2_SEED_OFFSET + run_offset + start
            )
            synchronize(runner.device)
            started = time.perf_counter()
            parts_by_run, weights_by_run = sample_batch_fn(
                start, end, make_torch_generator(run_seed, runner.device)
            )
            synchronize(runner.device)
            seconds[start:end] = [(time.perf_counter() - started) / (end - start)] * (
                end - start
            )
            submit_sampling_trial_result_futures(
                executor=executor,
                futures=futures,
                run_indices=range(start, end),
                samples_by_run=parts_by_run,
                runner=runner,
                comparison_mode=comparison_mode,
                metric_states=metric_states,
                metrics=metrics,
                weights_by_run=weights_by_run,
            )
        for run_idx, future in tqdm(futures, desc="Metrics", leave=False):
            results[run_idx] = future.result()
    return results, seconds


def mean_vector(values: Sequence[Sequence[float | int]]):
    if not values:
        return []
    return np.mean(np.asarray(values, dtype=float), axis=0).tolist()


def std_vector(values: Sequence[Sequence[float | int]]):
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] < 2:
        return np.zeros(arr.shape[1:], dtype=float).tolist()
    return np.std(arr, axis=0, ddof=1).tolist()
