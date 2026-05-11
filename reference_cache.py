import os
from typing import Dict

import torch


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
    if samples.ndim != 2 or samples.shape[1] != 2:
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
    payload: Dict[str, object] = {
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
