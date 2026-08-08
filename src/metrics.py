from __future__ import annotations

import math
import time
from threading import Lock
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SUPPORTED_METRICS = ("ks", "mmd")
REFERENCE_SAMPLE_METRICS = frozenset(SUPPORTED_METRICS)
MMD_CACHE_VERSION = 1
MMD_DEFAULT_PARAMS: dict[str, Any] = {
    "num_frequencies": 1024,
    "bandwidth_multipliers": [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "bandwidth_pairs": 8192,
    "seed": 0,
    "batch_size": 2500,
    "device": "runner",
    "quantize_images": True,
}


def normalize_metrics(
    raw_metrics=None, *, supported: Sequence[str] | None = None
) -> list[str]:
    if raw_metrics is None:
        raw_metrics = ["ks"]
    elif isinstance(raw_metrics, str):
        raw_metrics = [raw_metrics]
    supported_set = set(SUPPORTED_METRICS if supported is None else supported)
    metrics: list[str] = []
    seen = set()
    for raw_metric in raw_metrics:
        metric = str(raw_metric).lower()
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}; available: {list(SUPPORTED_METRICS)}"
            )
        if metric not in supported_set:
            raise ValueError(f"Metric {metric!r} is not supported by this runner")
        if metric not in seen:
            metrics.append(metric)
            seen.add(metric)
    if not metrics:
        raise ValueError("metrics must be non-empty")
    return metrics


def normalize_metric_params(raw_params=None) -> dict[str, dict[str, Any]]:
    if raw_params is None:
        return {}
    if not isinstance(raw_params, Mapping):
        raise ValueError("metric_params must be a mapping")
    params: dict[str, dict[str, Any]] = {}
    for raw_metric, raw_value in raw_params.items():
        metric = str(raw_metric).lower()
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Unknown metric_params entry {metric!r}")
        if raw_value is None:
            value = {}
        elif isinstance(raw_value, Mapping):
            value = dict(raw_value)
        else:
            raise ValueError(f"metric_params[{metric!r}] must be a mapping")
        if metric == "ks":
            if value:
                raise ValueError("ks no longer accepts metric parameters")
        elif metric == "mmd":
            value = _normalize_mmd_params(value)
        params[metric] = value
    return params


def uses_reference_samples(metrics: Sequence[str]) -> bool:
    return bool(REFERENCE_SAMPLE_METRICS.intersection(metrics))


def _normalize_mmd_params(
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(MMD_DEFAULT_PARAMS)
    merged["bandwidth_multipliers"] = list(MMD_DEFAULT_PARAMS["bandwidth_multipliers"])
    if params:
        unknown = sorted(set(params) - set(MMD_DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"Unknown mmd metric_params keys: {unknown}")
        merged.update(dict(params))

    for key in ("num_frequencies", "bandwidth_pairs", "batch_size"):
        if isinstance(merged[key], bool):
            raise ValueError(f"mmd {key} must be >= 1")
        merged[key] = int(merged[key])
        if merged[key] < 1:
            raise ValueError(f"mmd {key} must be >= 1")
    if isinstance(merged["seed"], bool):
        raise ValueError("mmd seed must be a nonnegative integer")
    merged["seed"] = int(merged["seed"])
    if merged["seed"] < 0:
        raise ValueError("mmd seed must be a nonnegative integer")

    raw_multipliers = merged["bandwidth_multipliers"]
    if isinstance(raw_multipliers, (str, bytes)) or not isinstance(
        raw_multipliers, Sequence
    ):
        raise ValueError("mmd bandwidth_multipliers must be a non-empty sequence")
    multipliers = [float(value) for value in raw_multipliers]
    if not multipliers or any(
        not math.isfinite(value) or value <= 0.0 for value in multipliers
    ):
        raise ValueError(
            "mmd bandwidth_multipliers must contain finite positive values"
        )
    if merged["num_frequencies"] % len(multipliers) != 0:
        raise ValueError(
            "mmd num_frequencies must be divisible by the number of bandwidths"
        )
    merged["bandwidth_multipliers"] = multipliers
    merged["device"] = str(merged["device"])
    if not merged["device"]:
        raise ValueError("mmd device must be non-empty")
    merged["quantize_images"] = bool(merged["quantize_images"])
    return merged


def validate_metric_dimensions(runner, metrics: Sequence[str]) -> None:
    if "ks" in metrics and int(runner.input_dim) > 2:
        raise ValueError(
            "KS supports only dimensions 1 and 2; "
            f"runner {runner.runner_name!r} has dimension {int(runner.input_dim)}"
        )


def metric_cache_key(runner, metrics: Sequence[str], metric_params: Mapping[str, Any]):
    normalized_params = normalize_metric_params(metric_params)
    validate_metric_dimensions(runner, metrics)
    key: dict[str, Any] = {
        "metrics": list(metrics),
    }
    if "mmd" in metrics:
        params = normalized_params.get("mmd") or {}
        key["mmd"] = _mmd_metric_cache_key(
            params,
            representation=_mmd_representation(runner, params),
        )
    return key


def prepare_metric_states(
    runner,
    *,
    comparison_mode: str,
    reference_samples=None,
    metrics: Sequence[str],
    metric_params: Mapping[str, Any] | None = None,
):
    metric_params = normalize_metric_params(metric_params)
    validate_metric_dimensions(runner, metrics)
    states: dict[str, Any] = {}
    if "ks" in metrics:
        states["ks"] = runner.prepare_comparison_state(
            comparison_mode=comparison_mode,
            reference_samples=reference_samples,
            metric_params=metric_params.get("ks"),
        )
    if "mmd" in metrics:
        states["mmd"] = _prepare_mmd_state(
            runner,
            comparison_mode=comparison_mode,
            reference_samples=reference_samples,
            params=metric_params.get("mmd") or {},
        )
    return states


def compute_trial_metrics(
    samples_tensor: torch.Tensor,
    runner,
    *,
    comparison_mode: str,
    metric_states: Mapping[str, Any],
    metrics: Sequence[str],
    phase1_x0_samples=None,
):
    values: dict[str, float] = {}
    payloads: dict[str, Any] = {}
    if "ks" in metrics:
        values["ks"] = float(
            runner.compute_ks_distance(
                samples_tensor,
                comparison_mode=comparison_mode,
                comparison_state=metric_states.get("ks"),
                extra_samples=phase1_x0_samples,
            )
        )
    if "mmd" in metrics:
        mmd_value, mmd_payload = _compute_mmd_metric(
            samples_tensor,
            metric_states.get("mmd"),
            phase1_x0_samples=phase1_x0_samples,
        )
        values["mmd"] = mmd_value
        payloads["mmd"] = mmd_payload
    return values, payloads


def summarize_metric_trials(
    trial_results: Sequence[Mapping[str, Any]], metrics: Sequence[str]
):
    summary: dict[str, Any] = {}
    for metric in metrics:

        def metric_value(trial):
            values = trial.get("metrics") or {}
            if metric in values:
                return values[metric]
            if metric == "ks":
                return trial.get("ks_distance", np.nan)
            return np.nan

        values = np.array(
            [metric_value(trial) for trial in trial_results],
            dtype=float,
        )
        valid = values[~np.isnan(values)]
        summary[f"mean_{metric}"] = float(valid.mean()) if valid.size else float("nan")
        summary[f"std_{metric}"] = (
            float(valid.std(ddof=1)) if valid.size > 1 else float("nan")
        )
        summary[f"n_valid_{metric}"] = int(valid.size)
    if "ks" in metrics:
        summary["n_valid_runs"] = summary["n_valid_ks"]
    return summary


def aggregate_cached_metric_rows(
    trials: Sequence[Mapping[str, Any]], metrics: Sequence[str]
):
    return summarize_metric_trials(trials, metrics)


def _mmd_metric_cache_key(
    params: Mapping[str, Any],
    *,
    representation: str,
) -> dict[str, Any]:
    params = _normalize_mmd_params(params)
    if representation not in {"quantized_image_target", "float_target"}:
        raise ValueError(f"Unknown mmd representation {representation!r}")
    return {
        "kind": "target_space_random_fourier_mmd",
        "version": MMD_CACHE_VERSION,
        "estimator": "biased_empirical_root",
        "kernel": "equal_weight_multiscale_gaussian",
        "standardization": "reference_coordinate_zscore_v1",
        "representation": representation,
        "num_frequencies": int(params["num_frequencies"]),
        "bandwidth_multipliers": list(params["bandwidth_multipliers"]),
        "bandwidth_pairs": int(params["bandwidth_pairs"]),
        "seed": int(params["seed"]),
        "quantize_images": bool(params["quantize_images"]),
    }


def _mmd_representation(runner, params: Mapping[str, Any]) -> str:
    params = _normalize_mmd_params(params)
    is_image = bool(runner.target_spec.get("image_shape"))
    if is_image and params["quantize_images"]:
        return "quantized_image_target"
    return "float_target"


def _mmd_target_dimension(runner) -> int | None:
    target_spec = runner.target_spec
    image_shape = target_spec.get("image_shape")
    if image_shape:
        return int(math.prod(int(value) for value in image_shape))
    for key in ("sample_dim", "dimension"):
        if key in target_spec:
            return int(target_spec[key])
    input_dim = getattr(runner, "input_dim", None)
    return None if input_dim is None else int(input_dim)


def _samples_to_mmd_coordinates(
    samples,
    *,
    expected_dimension: int | None = None,
) -> torch.Tensor:
    tensor = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
    tensor = tensor.detach().to(dtype=torch.float32)
    if tensor.ndim < 1:
        raise ValueError(
            f"Expected MMD samples with a sample dimension, got {tuple(tensor.shape)}"
        )
    count = int(tensor.shape[0])
    dimension = 1 if tensor.ndim == 1 else int(math.prod(tensor.shape[1:]))
    if dimension < 1:
        raise ValueError(
            "MMD samples must have at least one coordinate, "
            f"got {tuple(tensor.shape)}"
        )
    coordinates = tensor.reshape(count, dimension)
    if expected_dimension is not None and dimension != int(expected_dimension):
        raise ValueError(
            "MMD generated/reference coordinate width mismatch: "
            f"expected {int(expected_dimension)}, got {dimension}"
        )
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("mmd samples must all be finite")
    return coordinates.contiguous()


def _mmd_device(params: Mapping[str, Any], runner_device: str) -> torch.device:
    requested = str(params["device"])
    device = torch.device(runner_device if requested == "runner" else requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested MMD device {device} but CUDA is unavailable")
    return device


def _mmd_quantize(values: torch.Tensor) -> torch.Tensor:
    return values.clamp(0.0, 1.0).mul(255.0).round().div(255.0)


def _mmd_reference_standardization(
    reference: torch.Tensor,
    *,
    batch_size: int,
    quantize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    count, dimension = map(int, reference.shape)
    totals = torch.zeros(dimension, dtype=torch.float64)
    squared_totals = torch.zeros(dimension, dtype=torch.float64)
    for start in range(0, count, int(batch_size)):
        batch = reference[start : start + int(batch_size)].to(
            device="cpu", dtype=torch.float64
        )
        if quantize:
            batch = _mmd_quantize(batch)
        totals.add_(batch.sum(dim=0))
        squared_totals.add_(batch.square().sum(dim=0))
    mean = totals / float(count)
    variance = (squared_totals / float(count) - mean.square()).clamp_min_(0.0)
    std = variance.sqrt_()
    positive = std[std > 0.0]
    relative_floor = (
        float(positive.median().item()) * 1e-6 if int(positive.numel()) else 0.0
    )
    floor = max(1e-12, relative_floor)
    scale = torch.where(std > floor, std, torch.ones_like(std))
    return mean.to(dtype=torch.float32), scale.to(dtype=torch.float32)


def _mmd_standardize_batch(
    batch: torch.Tensor,
    *,
    device: torch.device,
    mean: torch.Tensor,
    scale: torch.Tensor,
    quantize: bool,
) -> torch.Tensor:
    values = batch.to(device=device, dtype=torch.float32)
    if quantize:
        values = _mmd_quantize(values)
    return values.sub(mean).div(scale)


def _mmd_reference_bandwidth(
    reference: torch.Tensor,
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
    params: Mapping[str, Any],
    device: torch.device,
    quantize: bool,
) -> float:
    count = int(reference.shape[0])
    pair_count = int(params["bandwidth_pairs"])
    rng = np.random.default_rng(int(params["seed"]))
    left_indices = rng.integers(0, count, size=pair_count, dtype=np.int64)
    offsets = rng.integers(1, count, size=pair_count, dtype=np.int64)
    right_indices = (left_indices + offsets) % count
    distances = []
    batch_size = int(params["batch_size"])
    for start in range(0, pair_count, batch_size):
        stop = min(start + batch_size, pair_count)
        left_index = torch.as_tensor(
            left_indices[start:stop], device=reference.device, dtype=torch.long
        )
        right_index = torch.as_tensor(
            right_indices[start:stop], device=reference.device, dtype=torch.long
        )
        left = _mmd_standardize_batch(
            reference.index_select(0, left_index),
            device=device,
            mean=mean,
            scale=scale,
            quantize=quantize,
        )
        right = _mmd_standardize_batch(
            reference.index_select(0, right_index),
            device=device,
            mean=mean,
            scale=scale,
            quantize=quantize,
        )
        distances.append(torch.linalg.vector_norm(left - right, dim=1).cpu())
    sampled = torch.cat(distances).to(dtype=torch.float64)
    bandwidth = float(sampled.median().item())
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        positive = sampled[sampled > 0.0]
        bandwidth = float(positive.median().item()) if int(positive.numel()) else 1.0
    return bandwidth


def _mmd_orthogonal_frequencies(
    dimension: int,
    count: int,
    *,
    bandwidth: float,
    generator: torch.Generator,
) -> torch.Tensor:
    blocks = []
    remaining = int(count)
    while remaining:
        block_size = min(int(dimension), remaining)
        directions, triangular = torch.linalg.qr(
            torch.randn(
                (int(dimension), block_size),
                generator=generator,
                dtype=torch.float32,
            ),
            mode="reduced",
        )
        diagonal = torch.diagonal(triangular)
        signs = torch.where(diagonal < 0.0, -1.0, 1.0)
        directions.mul_(signs.unsqueeze(0))
        radii = torch.linalg.vector_norm(
            torch.randn(
                (int(dimension), block_size),
                generator=generator,
                dtype=torch.float32,
            ),
            dim=0,
        )
        blocks.append(directions.mul_(radii.div(float(bandwidth)).unsqueeze(0)))
        remaining -= block_size
    return torch.cat(blocks, dim=1)


def _mmd_feature_means(
    sample_parts: Sequence[torch.Tensor],
    state: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    frequency_count = int(state["frequencies"].shape[1])
    cosine_sum = torch.zeros(frequency_count, dtype=torch.float64)
    sine_sum = torch.zeros(frequency_count, dtype=torch.float64)
    count = 0
    batch_size = int(state["params"]["batch_size"])
    for samples in sample_parts:
        sample_count = int(samples.shape[0])
        count += sample_count
        for start in range(0, sample_count, batch_size):
            batch = _mmd_standardize_batch(
                samples[start : start + batch_size],
                device=state["device"],
                mean=state["mean"],
                scale=state["scale"],
                quantize=state["representation"] == "quantized_image_target",
            )
            projected = batch @ state["frequencies"]
            cosine_sum.add_(torch.cos(projected).sum(dim=0, dtype=torch.float64).cpu())
            sine_sum.add_(torch.sin(projected).sum(dim=0, dtype=torch.float64).cpu())
    if count == 0:
        return cosine_sum, sine_sum, 0
    return cosine_sum.div_(count), sine_sum.div_(count), count


def _prepare_mmd_state(
    runner,
    *,
    comparison_mode: str,
    reference_samples,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    params = _normalize_mmd_params(params)
    if comparison_mode == "true_dist":
        raise ValueError("mmd requires a comparison mode backed by reference samples")
    if reference_samples is None:
        raise ValueError("mmd requires reference samples")

    reference = _samples_to_mmd_coordinates(
        reference_samples,
        expected_dimension=_mmd_target_dimension(runner),
    )
    reference_count, dimension = map(int, reference.shape)
    if reference_count < 2:
        raise ValueError("mmd requires at least two reference observations")
    representation = _mmd_representation(runner, params)
    quantize = representation == "quantized_image_target"
    device = _mmd_device(params, runner.device)
    mean, scale = _mmd_reference_standardization(
        reference,
        batch_size=int(params["batch_size"]),
        quantize=quantize,
    )
    mean = mean.to(device=device)
    scale = scale.to(device=device)
    bandwidth = _mmd_reference_bandwidth(
        reference,
        mean=mean,
        scale=scale,
        params=params,
        device=device,
        quantize=quantize,
    )

    frequencies_per_bandwidth = int(params["num_frequencies"]) // len(
        params["bandwidth_multipliers"]
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(params["seed"]))
    frequency_blocks = []
    bandwidths = []
    for multiplier in params["bandwidth_multipliers"]:
        current_bandwidth = bandwidth * float(multiplier)
        bandwidths.append(current_bandwidth)
        frequency_blocks.append(
            _mmd_orthogonal_frequencies(
                dimension,
                frequencies_per_bandwidth,
                bandwidth=current_bandwidth,
                generator=generator,
            )
        )
    state = {
        "params": params,
        "representation": representation,
        "dimension": dimension,
        "reference_count": int(reference_count),
        "device": device,
        "mean": mean,
        "scale": scale,
        "base_bandwidth": bandwidth,
        "bandwidths": bandwidths,
        "frequencies": torch.cat(frequency_blocks, dim=1).to(device=device),
        "lock": Lock(),
    }
    with torch.inference_mode():
        reference_cosine, reference_sine, measured_count = _mmd_feature_means(
            [reference], state
        )
    if measured_count != reference_count:
        raise RuntimeError("mmd reference feature count mismatch")
    state["reference_cosine_mean"] = reference_cosine
    state["reference_sine_mean"] = reference_sine
    return state


def _compute_mmd_metric(samples, state, *, phase1_x0_samples=None):
    if state is None:
        raise ValueError("mmd metric state is required")
    started = time.perf_counter()
    dimension = int(state["dimension"])
    generated = _samples_to_mmd_coordinates(
        samples,
        expected_dimension=dimension,
    )
    sample_parts = []
    if phase1_x0_samples is not None:
        phase1 = _samples_to_mmd_coordinates(
            phase1_x0_samples,
            expected_dimension=dimension,
        )
        if int(phase1.shape[0]) > 0:
            sample_parts.append(phase1)
    if int(generated.shape[0]) > 0:
        sample_parts.append(generated)
    count = sum(int(part.shape[0]) for part in sample_parts)
    payload = {
        "n_samples": int(count),
        "reference_count": int(state["reference_count"]),
        "dimension": dimension,
        "estimator": "biased_empirical_root",
        "kernel": "equal_weight_multiscale_gaussian_rff",
        "frequency_design": "orthogonal_random_features",
        "representation": state["representation"],
        "base_bandwidth": float(state["base_bandwidth"]),
        "bandwidth_multipliers": [
            float(value) for value in state["params"]["bandwidth_multipliers"]
        ],
        "bandwidths": [float(value) for value in state["bandwidths"]],
        "num_frequencies": int(state["frequencies"].shape[1]),
        "feature_dimension": 2 * int(state["frequencies"].shape[1]),
    }
    if count == 0:
        payload["scoring_seconds"] = float(time.perf_counter() - started)
        return float("nan"), payload

    with state["lock"], torch.inference_mode():
        cosine_mean, sine_mean, measured_count = _mmd_feature_means(sample_parts, state)
    if measured_count != count:
        raise RuntimeError("mmd generated feature count mismatch")
    cosine_delta = cosine_mean - state["reference_cosine_mean"]
    sine_delta = sine_mean - state["reference_sine_mean"]
    mmd_squared = float((cosine_delta.square() + sine_delta.square()).mean().item())
    mmd_squared = max(mmd_squared, 0.0)
    value = math.sqrt(mmd_squared)
    payload["mmd_squared"] = mmd_squared
    payload["scoring_seconds"] = float(time.perf_counter() - started)
    return value, payload
