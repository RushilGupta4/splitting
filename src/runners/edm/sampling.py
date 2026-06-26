from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from utils import validate_split_percentages

EPS = 1e-12
MAX_CHURN_RATE = 2 ** 0.5 - 1.0


@dataclass(frozen=True)
class EDMSchedule:
    sampling_steps: int
    sigma_min: float
    sigma_max: float
    rho: float
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if int(self.sampling_steps) < 2:
            raise ValueError("sampling_steps must be at least 2")
        if float(self.sigma_min) <= 0.0:
            raise ValueError("sigma_min must be > 0")
        if float(self.sigma_max) <= float(self.sigma_min):
            raise ValueError("sigma_max must be > sigma_min")
        if float(self.rho) <= 0.0:
            raise ValueError("rho must be > 0")
        i = torch.linspace(
            0.0,
            1.0,
            int(self.sampling_steps),
            device=self.device,
            dtype=self.dtype,
        )
        inv_rho = 1.0 / float(self.rho)
        sigmas = (
            float(self.sigma_max) ** inv_rho
            + i * (float(self.sigma_min) ** inv_rho - float(self.sigma_max) ** inv_rho)
        ) ** float(self.rho)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=self.device, dtype=self.dtype)])
        object.__setattr__(self, "sigmas", sigmas)

    @property
    def num_steps(self) -> int:
        return int(self.sampling_steps)

    def index_for_sigma(self, sigma: float) -> int:
        target = torch.as_tensor(float(sigma), device=self.sigmas.device, dtype=self.sigmas.dtype)
        idx = int(torch.argmin(torch.abs(self.sigmas - target)).item())
        if abs(float(self.sigmas[idx].item()) - float(sigma)) > max(EPS, 1e-6 * max(1.0, abs(float(sigma)))):
            raise ValueError(f"sigma {sigma} is not on the EDM schedule")
        return idx


def _denoise(model, x, sigma):
    sigma_tensor = torch.full((x.shape[0], 1), float(sigma), device=x.device, dtype=x.dtype)
    return model(x, sigma_tensor)


def _schedule_segment_indices(sigma_a, sigma_b, schedule: EDMSchedule):
    start_idx = schedule.index_for_sigma(float(sigma_a))
    end_idx = schedule.index_for_sigma(float(sigma_b))
    if start_idx > end_idx:
        raise ValueError(f"EDM segment must move toward lower sigma, got {sigma_a} -> {sigma_b}")
    return start_idx, end_idx


def _validate_churn_args(
    churn_rate: float,
    churn_min_noise_level: float,
    churn_max_noise_level: float,
    noise_level_inflation_factor: float,
):
    churn_rate = float(churn_rate)
    churn_min_noise_level = float(churn_min_noise_level)
    churn_max_noise_level = float(churn_max_noise_level)
    noise_level_inflation_factor = float(noise_level_inflation_factor)
    if churn_rate < 0.0:
        raise ValueError("churn_rate must be >= 0")
    if noise_level_inflation_factor < 0.0:
        raise ValueError("noise_level_inflation_factor must be >= 0")
    if churn_min_noise_level > churn_max_noise_level:
        raise ValueError("churn_min_noise_level must be <= churn_max_noise_level")
    return (
        churn_rate,
        churn_min_noise_level,
        churn_max_noise_level,
        noise_level_inflation_factor,
    )


def _apply_stochastic_churn(
    x,
    sigma_curr: float,
    schedule: EDMSchedule,
    *,
    churn_rate: float,
    churn_min_noise_level: float,
    churn_max_noise_level: float,
    noise_level_inflation_factor: float,
    generator=None,
):
    gamma = 0.0
    if churn_min_noise_level <= sigma_curr <= churn_max_noise_level:
        gamma = min(churn_rate / schedule.num_steps, MAX_CHURN_RATE)
    sigma_hat = sigma_curr * (1.0 + gamma)
    if gamma <= 0.0:
        return x, sigma_hat

    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    noise_scale = torch.sqrt(
        torch.as_tensor(
            max(sigma_hat ** 2 - sigma_curr ** 2, 0.0),
            device=x.device,
            dtype=x.dtype,
        )
    )
    return x + noise * noise_level_inflation_factor * noise_scale, sigma_hat


def edm_sample_segment(
    model,
    x,
    sigma_a,
    sigma_b,
    schedule: EDMSchedule,
    *,
    S_churn: float,
    S_min: float,
    S_max: float,
    S_noise: float,
    generator=None,
):
    S_churn, S_min, S_max, S_noise = _validate_churn_args(
        S_churn,
        S_min,
        S_max,
        S_noise,
    )
    if x.shape[0] == 0:
        return x
    start_idx, end_idx = _schedule_segment_indices(sigma_a, sigma_b, schedule)
    if start_idx == end_idx:
        return x
    with torch.inference_mode():
        for idx in range(start_idx, end_idx):
            sigma_curr = float(schedule.sigmas[idx].item())
            sigma_next = float(schedule.sigmas[idx + 1].item())
            x_hat, sigma_hat = _apply_stochastic_churn(
                x,
                sigma_curr,
                schedule,
                churn_rate=S_churn,
                churn_min_noise_level=S_min,
                churn_max_noise_level=S_max,
                noise_level_inflation_factor=S_noise,
                generator=generator,
            )

            denoised = _denoise(model, x_hat, sigma_hat)
            d_curr = (x_hat - denoised) / sigma_hat
            x_euler = x_hat + (sigma_next - sigma_hat) * d_curr
            if sigma_next <= 0.0:
                x = x_euler
            else:
                denoised_next = _denoise(model, x_euler, sigma_next)
                d_next = (x_euler - denoised_next) / sigma_next
                x = x_hat + 0.5 * (sigma_next - sigma_hat) * (d_curr + d_next)
    return x


def dpmpp_2s_sample_segment(
    model,
    x,
    sigma_a,
    sigma_b,
    schedule: EDMSchedule,
    *,
    stochastic_churn_rate: float,
    churn_min_noise_level: float,
    churn_max_noise_level: float,
    noise_level_inflation_factor: float,
    generator=None,
):
    (
        stochastic_churn_rate,
        churn_min_noise_level,
        churn_max_noise_level,
        noise_level_inflation_factor,
    ) = _validate_churn_args(
        stochastic_churn_rate,
        churn_min_noise_level,
        churn_max_noise_level,
        noise_level_inflation_factor,
    )
    if x.shape[0] == 0:
        return x
    start_idx, end_idx = _schedule_segment_indices(sigma_a, sigma_b, schedule)
    if start_idx == end_idx:
        return x
    with torch.inference_mode():
        for idx in range(start_idx, end_idx):
            sigma_curr = float(schedule.sigmas[idx].item())
            sigma_next = float(schedule.sigmas[idx + 1].item())
            x_hat, sigma_hat = _apply_stochastic_churn(
                x,
                sigma_curr,
                schedule,
                churn_rate=stochastic_churn_rate,
                churn_min_noise_level=churn_min_noise_level,
                churn_max_noise_level=churn_max_noise_level,
                noise_level_inflation_factor=noise_level_inflation_factor,
                generator=generator,
            )

            denoised = _denoise(model, x_hat, sigma_hat)
            if sigma_next <= 0.0:
                x = denoised
                continue

            sigma_mid = (sigma_hat * sigma_next) ** 0.5
            mid_over_current = sigma_mid / sigma_hat
            x_mid = mid_over_current * x_hat + (1.0 - mid_over_current) * denoised
            denoised_mid = _denoise(model, x_mid, sigma_mid)
            next_over_current = sigma_next / sigma_hat
            x = next_over_current * x_hat + (1.0 - next_over_current) * denoised_mid
    return x


def sde_euler_maruyama_sample_segment(
    model,
    x,
    sigma_a,
    sigma_b,
    schedule: EDMSchedule,
    *,
    generator=None,
):
    if x.shape[0] == 0:
        return x
    start_idx, end_idx = _schedule_segment_indices(sigma_a, sigma_b, schedule)
    if start_idx == end_idx:
        return x
    with torch.inference_mode():
        for idx in range(start_idx, end_idx):
            sigma_curr = float(schedule.sigmas[idx].item())
            sigma_next = float(schedule.sigmas[idx + 1].item())
            delta_var = sigma_curr ** 2 - sigma_next ** 2
            delta_var_tensor = torch.as_tensor(
                max(delta_var, 0.0), device=x.device, dtype=x.dtype
            )
            denoised = _denoise(model, x, sigma_curr)
            score = (denoised - x) / (sigma_curr ** 2)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = x + delta_var * score + torch.sqrt(delta_var_tensor) * noise
    return x


def resolve_split_percentages(schedule: EDMSchedule, percentages: Sequence[float]):
    validate_split_percentages(percentages)
    N = schedule.num_steps
    indices = []
    for p in percentages:
        j = int(round((1.0 - float(p)) * N))
        if j <= 0 or j >= N:
            raise ValueError(f"split percentage {p} maps to invalid index {j}")
        indices.append(j)
    for i in range(len(indices) - 1):
        if indices[i] >= indices[i + 1]:
            raise ValueError("split indices must be strictly increasing along sigma trajectory")
    split_sigmas = [float(schedule.sigmas[j].item()) for j in indices]
    remaining_steps = [N - j for j in indices]
    return remaining_steps, split_sigmas
