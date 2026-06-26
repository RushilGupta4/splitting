from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

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
    comparison_state,
    phase1_x0_samples=None,
):
    samples_tensor = (
        samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
    )
    leaf_count = int(samples_tensor.reshape(-1, samples_tensor.shape[-1]).shape[0])

    ks_distance = runner.compute_ks_distance(
        samples_tensor,
        comparison_mode=comparison_mode,
        comparison_state=comparison_state,
        extra_samples=phase1_x0_samples,
    )

    result: Dict[str, Any] = {
        "ks_distance": float(ks_distance),
        "leaf_count": int(leaf_count),
    }
    return result


def submit_sampling_trial_result_futures(
    executor: ThreadPoolExecutor,
    futures: List[Tuple[int, Any]],
    run_indices: Sequence[int],
    samples_by_run: Sequence,
    runner,
    comparison_mode: str,
    comparison_state,
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
            comparison_state,
            phase1_x0_samples_by_run[local_idx],
        )
        futures.append((int(run_idx), future))


def collect_sampling_trial_result_futures(
    trial_results: List[Dict[str, Any] | None],
    futures: Sequence[Tuple[int, Any]],
    *,
    desc: str = "KS",
):
    for run_idx, future in tqdm(futures, desc=desc, leave=False):
        trial_results[run_idx] = future.result()


def summarize_sampling_trials(trial_results: Sequence[Dict[str, Any]]):
    if not trial_results:
        return {
            "n_valid_runs": 0,
            "extinction_rate": 0.0,
            "mean_ks": float("nan"),
            "std_ks": float("nan"),
        }

    values = np.array(
        [trial.get("ks_distance", float("nan")) for trial in trial_results],
        dtype=float,
    )
    leaf_counts = np.array(
        [trial.get("leaf_count", 0) for trial in trial_results], dtype=float
    )
    valid = values[~np.isnan(values)]
    mean_ks = float(valid.mean()) if valid.size else float("nan")
    std_ks = float(valid.std()) if valid.size else float("nan")

    return {
        "n_valid_runs": int(valid.size),
        "extinction_rate": (
            float((leaf_counts == 0).mean()) if leaf_counts.size else 0.0
        ),
        "mean_ks": mean_ks,
        "std_ks": std_ks,
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
