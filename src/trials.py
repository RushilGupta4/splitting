from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from metrics import compute_trial_metrics, summarize_metric_trials

PHASE2_SEED_OFFSET = 1_000_000


def make_torch_generator(seed: int | None, device: str):
    if seed is None:
        return None
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def iter_run_chunks(n_runs: int, n_parallel: int):
    if n_runs < 1 or n_parallel < 1:
        raise ValueError("n_runs and n_parallel must be at least 1")
    for start in range(0, n_runs, n_parallel):
        yield start, min(start + n_parallel, n_runs)


def build_sampling_trial_result(
    samples,
    runner,
    comparison_mode: str,
    metric_states,
    metrics,
    phase1_x0_samples=None,
):
    samples_tensor = (
        samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
    )
    leaf_count = int(samples_tensor.reshape(-1, samples_tensor.shape[-1]).shape[0])

    metric_values, metric_payloads = compute_trial_metrics(
        samples_tensor,
        runner,
        comparison_mode=comparison_mode,
        metric_states=metric_states,
        metrics=metrics,
        phase1_x0_samples=phase1_x0_samples,
    )

    result: Dict[str, Any] = {
        "metrics": metric_values,
        "leaf_count": int(leaf_count),
    }
    if metric_payloads:
        result["metric_payloads"] = metric_payloads
    if "ks" in metric_values:
        result["ks_distance"] = float(metric_values["ks"])
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
    phase1_x0_samples_by_run: Sequence | None = None,
):
    run_indices = list(run_indices)
    if phase1_x0_samples_by_run is None:
        phase1_x0_samples_by_run = [None] * len(samples_by_run)
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
            phase1_x0_samples_by_run[local_idx],
        )
        futures.append((int(run_idx), future))


def collect_sampling_trial_result_futures(
    trial_results: List[Dict[str, Any] | None],
    futures: Sequence[Tuple[int, Any]],
    *,
    desc: str = "Metrics",
):
    for run_idx, future in tqdm(futures, desc=desc, leave=False):
        trial_results[run_idx] = future.result()


def summarize_sampling_trials(
    trial_results: Sequence[Dict[str, Any]],
    metrics: Sequence[str] = ("ks",),
):
    if not trial_results:
        result = {
            "n_valid_runs": 0,
            "extinction_rate": 0.0,
        }
        for metric in metrics:
            result[f"mean_{metric}"] = float("nan")
            result[f"std_{metric}"] = float("nan")
            result[f"n_valid_{metric}"] = 0
        return result

    leaf_counts = np.array(
        [trial.get("leaf_count", 0) for trial in trial_results], dtype=float
    )
    return {
        "extinction_rate": (
            float((leaf_counts == 0).mean()) if leaf_counts.size else 0.0
        ),
        **summarize_metric_trials(trial_results, metrics),
    }


def mean_scalar(values: Sequence[float | int]):
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


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
