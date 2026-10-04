from __future__ import annotations

import math

import torch

SDE_SAMPLERS = ("euler",)


def sample_sde_segment(
    case,
    x: torch.Tensor,
    start_idx: int,
    end_idx: int,
    *,
    sampling_steps: int,
    terminal_time: float,
    generator=None,
    progress_callback=None,
) -> torch.Tensor:
    """Euler--Maruyama steps ``start_idx -> end_idx`` with diagonal diffusion."""
    if x.shape[0] == 0 or end_idx == start_idx:
        return x
    dt = float(terminal_time) / float(sampling_steps)
    sqdt = math.sqrt(dt)
    for step_idx in range(start_idx, end_idx):
        case_t = step_idx * dt / float(terminal_time)
        drift = case.drift(case_t, x)
        diffusion = torch.as_tensor(
            case.diffusion(case_t, x), device=x.device, dtype=x.dtype
        )
        try:
            diffusion = torch.broadcast_to(diffusion, x.shape)
        except RuntimeError as exc:
            raise ValueError(
                "Diagonal SDE diffusion must broadcast to the state shape "
                f"{tuple(x.shape)}, got {tuple(diffusion.shape)}"
            ) from exc
        noise = sqdt * torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        )
        x = x + drift * dt + diffusion * noise
        if progress_callback is not None:
            progress_callback(1)
    return x
