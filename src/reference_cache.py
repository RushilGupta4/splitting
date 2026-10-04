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


def load_reference_samples_if_sufficient(path: str, required_count: int):
    """Truncated samples tensor, or None on a miss or too few samples; mmap avoids an eager copy."""
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
    return samples[: int(required_count)]


def load_reference_samples_checked(path: str, required_count: int) -> torch.Tensor:
    samples = load_reference_samples_if_sufficient(path, required_count)
    if samples is None:
        raise FileNotFoundError(
            f"Missing/insufficient reference samples at {path}; "
            f"need >= {required_count}. Run ensure_samples.py first."
        )
    return samples


def reference_path_for_runner(runner, comparison_mode, reference_generation_config):
    """``(path, cache_key)`` of a runner's base reference samples."""
    cache_key = runner.reference_cache_key(comparison_mode, reference_generation_config)
    return reference_samples_path_for_key(runner.checkpoint_path, cache_key), cache_key


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


def _atomic_write(path: str, suffix: str, write) -> None:
    """``write(temp_path)`` next to ``path``, then move it into place."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{os.path.basename(path)}.", suffix=suffix, dir=parent, delete=False
    )
    temp_path = handle.name
    handle.close()
    try:
        write(temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


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
    """Atomically save the first 25 RGB samples as a 5x5 PNG grid."""
    shape = tuple(int(v) for v in image_shape)
    if len(shape) != 3 or shape[0] != 3 or min(shape[1:]) < 1:
        raise ValueError(
            "Reference previews require image_shape=(3, height, width), "
            f"got {shape}"
        )
    values = torch.as_tensor(samples)
    sample_dim = shape[0] * shape[1] * shape[2]
    if values.ndim != 2 or int(values.shape[1]) != sample_dim:
        raise ValueError(
            "Reference preview expects flattened samples with shape "
            f"[N, {sample_dim}], got {tuple(values.shape)}"
        )
    if int(values.shape[0]) < 25:
        raise ValueError(
            f"Reference preview requires at least 25 samples, got {values.shape[0]}"
        )
    images = values[:25].to(dtype=torch.float32).reshape(25, *shape)
    if not torch.isfinite(images).all():
        raise ValueError("Reference preview samples contain non-finite values")
    if bool(torch.any(images < 0.0)) or bool(torch.any(images > 1.0)):
        raise ValueError("Reference preview samples must lie in [0, 1]")

    from torchvision.utils import save_image

    path = reference_preview_path(reference_path)
    _atomic_write(
        path,
        ".png",
        lambda temp_path: save_image(
            images, temp_path, nrow=5, padding=2, pad_value=1.0, normalize=False
        ),
    )
    return path


def save_reference_samples_with_key(
    path: str,
    samples: torch.Tensor,
    *,
    cache_key: Mapping[str, object],
):
    payload: dict[str, Any] = {
        "runner": cache_key.get("runner"),
        "comparison_mode": cache_key.get("comparison_mode"),
        "reference_cache_key": _json_safe(dict(cache_key)),
        "num_samples": int(samples.shape[0]),
        "samples": samples.cpu(),
    }
    _atomic_write(path, ".tmp", lambda temp_path: torch.save(payload, temp_path))


def load_reference_samples_for_runner(
    runner,
    comparison_mode: str,
    reference_generation_config: Mapping[str, object],
    required_count: int,
) -> torch.Tensor:
    """Load reference samples by the runner's cache key."""
    if runner.checkpoint_path is None:
        raise ValueError("runner must have a checkpoint_path to load reference samples")
    path, _ = reference_path_for_runner(
        runner, comparison_mode, reference_generation_config
    )
    return load_reference_samples_checked(path, required_count)


def load_space_references(
    runner,
    comparison_mode: str,
    reference_generation_config: Mapping[str, object],
    base_samples,
    spaces_needed,
    required_count: int,
    *,
    batch_size: int,
) -> dict:
    """``{space: reference}`` for MMD spaces other than the sample space.

    Each is the base reference mapped into ``space`` (decoded, embedded), cached
    next to it under the base key plus the space's extractor key; built on first use.
    """
    from metrics import spaces

    base_key = runner.reference_cache_key(comparison_mode, reference_generation_config)
    references = {}
    for space in spaces_needed:
        key = {**dict(base_key), "space": spaces.space_cache_key(space)}
        path = reference_samples_path_for_key(runner.checkpoint_path, key)
        samples = load_reference_samples_if_sufficient(path, required_count)
        if samples is None:
            samples = spaces.features_in_chunks(
                runner, space, base_samples[: int(required_count)], batch_size=batch_size
            )
            save_reference_samples_with_key(path, samples, cache_key=key)
        references[space] = samples
    return references
