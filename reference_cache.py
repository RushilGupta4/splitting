import hashlib
import json
import os
import re
from typing import Any, Mapping

import torch


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(value: object) -> str:
    text = str(value) if value is not None else ""
    cleaned = _SAFE_FILENAME_RE.sub("_", text).strip("._-")
    return cleaned or "x"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _digest_cache_key(cache_key: Mapping[str, object]) -> str:
    serialized = json.dumps(
        _json_safe(dict(cache_key)), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def reference_samples_path_for_key(
    checkpoint_path: str, cache_key: Mapping[str, object]
) -> str:
    digest = _digest_cache_key(cache_key)
    runner_name = safe_filename(cache_key.get("runner", "runner"))
    mode = safe_filename(cache_key.get("comparison_mode", "reference"))
    return os.path.join(
        os.path.dirname(checkpoint_path) or ".",
        f"samples_{runner_name}_{mode}_{digest[:16]}.pt",
    )


# Legacy paths kept for backward compatibility with existing caches.

def format_eta_for_filename(eta: float) -> str:
    return format(float(eta), ".12g").replace("-", "m").replace(".", "p")


def reference_samples_path(checkpoint_path: str, sampling_steps: int, eta: float) -> str:
    return os.path.join(
        os.path.dirname(checkpoint_path) or ".",
        f"samples_steps{int(sampling_steps)}_eta{format_eta_for_filename(eta)}.pt",
    )


def true_reference_samples_path(checkpoint_path: str) -> str:
    return os.path.join(os.path.dirname(checkpoint_path) or ".", "true_samples.pt")


def _try_load(
    path: str,
    required_count: int,
    *,
    expected_sampling_steps: int | None = None,
    expected_eta: float | None = None,
):
    """Return truncated samples tensor or None on miss/mismatch/insufficient."""
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu")
    samples = payload["samples"] if isinstance(payload, dict) else payload
    samples = torch.as_tensor(samples)
    if samples.ndim != 2:
        return None
    if int(samples.shape[0]) < int(required_count):
        return None
    if isinstance(payload, dict):
        steps = payload.get("sampling_steps")
        eta = payload.get("eta")
        if expected_sampling_steps is not None and steps is not None and int(steps) != int(expected_sampling_steps):
            return None
        if expected_eta is not None and eta is not None and float(eta) != float(expected_eta):
            return None
    return samples[: int(required_count)]


def load_reference_samples_checked(
    path: str,
    required_count: int,
    *,
    expected_sampling_steps: int | None = None,
    expected_eta: float | None = None,
) -> torch.Tensor:
    samples = _try_load(
        path,
        required_count,
        expected_sampling_steps=expected_sampling_steps,
        expected_eta=expected_eta,
    )
    if samples is None:
        raise FileNotFoundError(
            f"Missing/insufficient reference samples at {path}; "
            f"need >= {required_count}. Run ensure_samples.py first."
        )
    return samples


def reference_is_sufficient(path: str, required_count: int) -> bool:
    return _try_load(path, required_count) is not None


def save_reference_samples_with_key(
    path: str,
    samples: torch.Tensor,
    *,
    cache_key: Mapping[str, object],
):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload: dict[str, Any] = {
        "runner": cache_key.get("runner"),
        "comparison_mode": cache_key.get("comparison_mode"),
        "reference_cache_key": _json_safe(dict(cache_key)),
        "num_samples": int(samples.shape[0]),
        "samples": samples.cpu(),
    }
    torch.save(payload, path)


def save_reference_samples(
    path: str,
    samples: torch.Tensor,
    *,
    checkpoint: str,
    T: int,
    sampling_steps: int,
    eta: float,
    reference_mode: str = "ddpm_samples",
    target_spec=None,
):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload: dict[str, object] = {
        "checkpoint": checkpoint,
        "reference_mode": reference_mode,
        "T": int(T),
        "sampling_steps": int(sampling_steps),
        "eta": float(eta),
        "num_samples": int(samples.shape[0]),
        "samples": samples.cpu(),
    }
    if target_spec is not None:
        payload["target_spec"] = target_spec
    torch.save(payload, path)


def load_reference_samples_for_runner(
    runner,
    comparison_mode: str,
    required_count: int,
    *,
    legacy_paths: tuple[str, ...] = (),
) -> torch.Tensor:
    """Load reference samples by the runner's cache key, with optional legacy fallback."""
    if runner.checkpoint_path is None:
        raise ValueError("runner must have a checkpoint_path to load reference samples")
    cache_key = runner.reference_cache_key(comparison_mode)
    primary = reference_samples_path_for_key(runner.checkpoint_path, cache_key)
    samples = _try_load(primary, required_count)
    if samples is not None:
        return samples
    for legacy in legacy_paths:
        samples = _try_load(legacy, required_count)
        if samples is not None:
            return samples
    raise FileNotFoundError(
        f"Missing/insufficient reference samples at {primary}; "
        f"need >= {required_count}. Run ensure_samples.py first."
    )
