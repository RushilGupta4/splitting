from __future__ import annotations

import math

import torch

SDE_SAMPLERS = ("euler", "milstein")


def _coerce_state(
    x: torch.Tensor,
    *,
    dimension: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if values.ndim == 1:
        if int(dimension) != 1:
            raise ValueError(
                f"Expected SDE state with shape [N, {int(dimension)}], got {tuple(values.shape)}"
            )
        values = values.reshape(-1, 1)
    elif values.ndim != 2:
        raise ValueError(
            f"Expected SDE state with shape [N, {int(dimension)}], got {tuple(values.shape)}"
        )
    if int(values.shape[1]) != int(dimension):
        raise ValueError(
            f"Expected SDE state with shape [N, {int(dimension)}], got {tuple(values.shape)}"
        )
    return values.to(device=device, dtype=dtype).contiguous()


def sample_sde_segment(
    *,
    case,
    sampler: str,
    x: torch.Tensor,
    start_idx: int,
    end_idx: int,
    sampling_steps: int,
    terminal_time: float,
    dimension: int,
    coupling_strength: float,
    device: str,
    dtype: torch.dtype,
    generator=None,
    progress_callback=None,
) -> torch.Tensor:
    if end_idx < start_idx:
        raise ValueError(f"SDE segment must move forward, got {start_idx} -> {end_idx}")
    out = _coerce_state(x, dimension=int(dimension), device=device, dtype=dtype)
    if out.shape[0] == 0 or end_idx == start_idx:
        return out

    dt = float(terminal_time) / float(sampling_steps)
    sqdt = math.sqrt(dt)
    coupling = float(coupling_strength)
    for step_idx in range(start_idx, end_idx):
        t = step_idx * dt
        case_t = t / float(terminal_time)
        x_prev = out
        drift = case.drift(case_t, x_prev)
        if coupling != 0.0:
            drift = drift + coupling * (x_prev.mean(dim=1, keepdim=True) - x_prev)
        diffusion = case.diffusion(case_t, x_prev)
        d_w = sqdt * torch.randn(
            out.shape,
            device=out.device,
            dtype=out.dtype,
            generator=generator,
        )
        out = x_prev + drift * dt + diffusion * d_w
        if sampler == "milstein":
            diffusion_derivative = case.diffusion_derivative(case_t, x_prev)
            out = out + 0.5 * diffusion * diffusion_derivative * (d_w * d_w - dt)
        if progress_callback is not None:
            progress_callback(1)
    return out
