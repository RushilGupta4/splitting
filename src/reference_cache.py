import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
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
    try:
        payload = torch.load(path, map_location="cpu", mmap=True)
    except Exception:
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            return None
    try:
        if isinstance(payload, dict):
            if "samples" not in payload:
                return None
            samples = payload["samples"]
        else:
            samples = payload
        samples = torch.as_tensor(samples)
    except Exception:
        return None
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


def load_reference_samples_if_sufficient(
    path: str,
    required_count: int,
) -> torch.Tensor | None:
    """Load a sufficient cache once, using mmap to avoid an eager full copy."""
    return _try_load(path, required_count)


def reference_preview_path(reference_path: str) -> str:
    stem, _ = os.path.splitext(reference_path)
    return f"{stem}_preview_5x5.png"


@lru_cache(maxsize=32)
def _fingerprint_file(real_path: str, size: int, mtime_ns: int) -> dict[str, object]:
    del mtime_ns
    digest = hashlib.sha256()
    with open(real_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "size": int(size),
        "sha256": digest.hexdigest(),
    }


def checkpoint_fingerprint(path: str | None) -> dict[str, object] | None:
    """Return a process-cached content fingerprint for a local checkpoint."""
    if path is None or not os.path.isfile(path):
        return None
    real_path = os.path.realpath(path)
    stat = os.stat(real_path)
    return _fingerprint_file(real_path, int(stat.st_size), int(stat.st_mtime_ns))


def save_reference_preview(
    reference_path: str,
    samples: torch.Tensor,
    *,
    image_shape,
) -> str:
    """Atomically save the first 25 CIFAR samples as a 5x5 PNG grid."""
    shape = tuple(int(v) for v in image_shape)
    if shape != (3, 32, 32):
        raise ValueError(
            "Reference previews currently require CIFAR image_shape=(3, 32, 32), "
            f"got {shape}"
        )
    values = torch.as_tensor(samples)
    if values.ndim != 2 or int(values.shape[1]) != 3 * 32 * 32:
        raise ValueError(
            "CIFAR reference preview expects flattened samples with shape "
            f"[N, 3072], got {tuple(values.shape)}"
        )
    if int(values.shape[0]) < 25:
        raise ValueError(
            f"CIFAR reference preview requires at least 25 samples, got {values.shape[0]}"
        )
    images = values[:25].to(dtype=torch.float32).reshape(25, 3, 32, 32)
    if not torch.isfinite(images).all():
        raise ValueError("CIFAR reference preview samples contain non-finite values")
    if bool(torch.any(images < 0.0)) or bool(torch.any(images > 1.0)):
        raise ValueError("CIFAR reference preview samples must lie in [0, 1]")

    from torchvision.utils import save_image

    path = reference_preview_path(reference_path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{os.path.basename(path)}.",
        suffix=".png",
        dir=parent,
        delete=False,
    )
    temp_path = handle.name
    handle.close()
    try:
        save_image(
            images,
            temp_path,
            nrow=5,
            padding=2,
            pad_value=1.0,
            normalize=False,
        )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return path


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
    directory = parent or "."
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
        delete=False,
    )
    temp_path = handle.name
    handle.close()
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


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
