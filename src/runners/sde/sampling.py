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


def _diagonal_noise_increment(
    diffusion: torch.Tensor,
    state: torch.Tensor,
    *,
    sqdt: float,
    generator=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    diffusion = torch.as_tensor(
        diffusion,
        device=state.device,
        dtype=state.dtype,
    )
    try:
        diffusion = torch.broadcast_to(diffusion, state.shape)
    except RuntimeError as exc:
        raise ValueError(
            "Diagonal SDE diffusion must broadcast to the state shape "
            f"{tuple(state.shape)}, got {tuple(diffusion.shape)}"
        ) from exc
    d_w = sqdt * torch.randn(
        state.shape,
        device=state.device,
        dtype=state.dtype,
        generator=generator,
    )
    return diffusion * d_w, d_w


def _matrix_noise_increment(
    diffusion: torch.Tensor,
    state: torch.Tensor,
    *,
    sqdt: float,
    generator=None,
) -> torch.Tensor:
    diffusion = torch.as_tensor(
        diffusion,
        device=state.device,
        dtype=state.dtype,
    )
    num_samples, state_dim = state.shape
    if diffusion.ndim == 2:
        if int(diffusion.shape[0]) != int(state_dim):
            raise ValueError(
                "Constant matrix SDE diffusion must have shape [D, M] with "
                f"D={state_dim}, got {tuple(diffusion.shape)}"
            )
        noise_dim = int(diffusion.shape[1])
        d_w = sqdt * torch.randn(
            (num_samples, noise_dim),
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
        return d_w @ diffusion.transpose(0, 1)
    if diffusion.ndim == 3:
        if tuple(diffusion.shape[:2]) != (num_samples, state_dim):
            raise ValueError(
                "Batched matrix SDE diffusion must have shape [N, D, M] with "
                f"(N, D)=({num_samples}, {state_dim}), got {tuple(diffusion.shape)}"
            )
        noise_dim = int(diffusion.shape[2])
        d_w = sqdt * torch.randn(
            (num_samples, noise_dim),
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
        return torch.bmm(diffusion, d_w.unsqueeze(-1)).squeeze(-1)
    raise ValueError(
        "Matrix SDE diffusion must have shape [D, M] or [N, D, M], "
        f"got {tuple(diffusion.shape)}"
    )


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
    diffusion_structure = str(getattr(case, "diffusion_structure", "diagonal"))
    if diffusion_structure not in {"diagonal", "matrix"}:
        raise ValueError(
            f"Unknown SDE diffusion structure {diffusion_structure!r}"
        )
    if sampler == "milstein" and diffusion_structure != "diagonal":
        raise ValueError("Milstein requires diagonal diffusion")
    for step_idx in range(start_idx, end_idx):
        t = step_idx * dt
        case_t = t / float(terminal_time)
        x_prev = out
        drift = case.drift(case_t, x_prev)
        if coupling != 0.0:
            drift = drift + coupling * (x_prev.mean(dim=1, keepdim=True) - x_prev)
        diffusion = case.diffusion(case_t, x_prev)
        if diffusion_structure == "diagonal":
            noise_increment, d_w = _diagonal_noise_increment(
                diffusion,
                x_prev,
                sqdt=sqdt,
                generator=generator,
            )
        else:
            noise_increment = _matrix_noise_increment(
                diffusion,
                x_prev,
                sqdt=sqdt,
                generator=generator,
            )
            d_w = None
        out = x_prev + drift * dt + noise_increment
        if sampler == "milstein":
            diffusion_derivative = case.diffusion_derivative(case_t, x_prev)
            diffusion = torch.as_tensor(
                diffusion,
                device=x_prev.device,
                dtype=x_prev.dtype,
            )
            diffusion_derivative = torch.as_tensor(
                diffusion_derivative,
                device=x_prev.device,
                dtype=x_prev.dtype,
            )
            out = out + 0.5 * diffusion * diffusion_derivative * (d_w * d_w - dt)
        if progress_callback is not None:
            progress_callback(1)
    return out
