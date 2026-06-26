from __future__ import annotations

import math

import torch

SDE1D_SAMPLERS = ("euler", "milstein")


def sample_sde_segment(
    *,
    case,
    sampler: str,
    x: torch.Tensor,
    start_idx: int,
    end_idx: int,
    sampling_steps: int,
    terminal_time: float,
    device: str,
    dtype: torch.dtype,
    generator=None,
) -> torch.Tensor:
    if end_idx < start_idx:
        raise ValueError(f"SDE segment must move forward, got {start_idx} -> {end_idx}")
    if x.shape[0] == 0 or end_idx == start_idx:
        return x.reshape(-1, 1).to(device=device, dtype=dtype)

    out = x.reshape(-1, 1).to(device=device, dtype=dtype)
    dt = float(terminal_time) / float(sampling_steps)
    sqdt = math.sqrt(dt)
    for step_idx in range(start_idx, end_idx):
        t = step_idx * dt
        case_t = t / float(terminal_time)
        x_prev = out
        drift = case.drift(case_t, x_prev)
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
    return out
