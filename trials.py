from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ks import coerce_samples_np, compute_ks_distance, warm_ks_kernel_for_mode

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
    samples_np: np.ndarray,
    target_spec: Dict[str, Any],
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    phase1_x0_samples: torch.Tensor | np.ndarray | None = None,
):
    samples_np = coerce_samples_np(samples_np)
    leaf_count = int(samples_np.shape[0])
    ks_samples = samples_np
    if phase1_x0_samples is not None and int(phase1_x0_samples.shape[0]) > 0:
        ks_samples = np.concatenate(
            [coerce_samples_np(phase1_x0_samples), samples_np], axis=0
        )
    if ks_samples.shape[0] > 0:
        ks_distance, _, _ = compute_ks_distance(
            ks_samples, target_spec, reference_mode, reference_cdf_state
        )
    else:
        ks_distance = float("nan")
    return {"ks_distance": float(ks_distance), "leaf_count": leaf_count}


def submit_sampling_trial_result_futures(
    executor: ThreadPoolExecutor,
    futures: List[Tuple[int, Any]],
    run_indices: Sequence[int],
    samples_by_run: Sequence[torch.Tensor | np.ndarray],
    target_spec: Dict[str, Any],
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    *,
    phase1_x0_samples_by_run: Sequence[torch.Tensor | np.ndarray | None] | None = None,
):
    run_indices = list(run_indices)
    if phase1_x0_samples_by_run is None:
        phase1_x0_samples_by_run = [None] * len(samples_by_run)
    if len(samples_by_run) != len(run_indices):
        raise ValueError("run_indices must match samples_by_run length")
    samples_cpu = [coerce_samples_np(s) for s in samples_by_run]
    phase1_cpu = [
        None if s is None else coerce_samples_np(s) for s in phase1_x0_samples_by_run
    ]
    for local_idx, run_idx in enumerate(run_indices):
        future = executor.submit(
            build_sampling_trial_result,
            samples_cpu[local_idx],
            target_spec,
            reference_cdf_state,
            reference_mode,
            phase1_cpu[local_idx],
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
    ks_distances = np.array(
        [trial["ks_distance"] for trial in trial_results], dtype=float
    )
    leaf_counts = np.array(
        [trial["leaf_count"] for trial in trial_results], dtype=float
    )
    valid = ks_distances[~np.isnan(ks_distances)]
    return {
        "mean_ks": float(valid.mean()) if valid.size else float("nan"),
        "std_ks": float(valid.std()) if valid.size else float("nan"),
        "n_valid_runs": int(valid.size),
        "extinction_rate": (
            float((leaf_counts == 0).mean()) if leaf_counts.size else 0.0
        ),
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
    return np.std(np.asarray(values, dtype=float), axis=0).tolist()


def warm_sampling_ks(reference_mode: str, target_spec: Dict[str, Any]):
    warm_ks_kernel_for_mode(reference_mode, target_spec)
