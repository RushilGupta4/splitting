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


def load_reference_samples_for_runner(
    runner,
    comparison_mode: str,
    reference_generation_config: Mapping[str, object],
    required_count: int,
) -> torch.Tensor:
    """Load reference samples by the runner's cache key."""
    if runner.checkpoint_path is None:
        raise ValueError("runner must have a checkpoint_path to load reference samples")
    cache_key = runner.reference_cache_key(comparison_mode, reference_generation_config)
    primary = reference_samples_path_for_key(runner.checkpoint_path, cache_key)
    samples = _try_load(primary, required_count)
    if samples is not None:
        return samples
    raise FileNotFoundError(
        f"Missing/insufficient reference samples at {primary}; "
        f"need >= {required_count}. Run ensure_samples.py first."
    )
