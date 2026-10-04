import math
import time
from typing import List, Sequence, Tuple

import numpy as np
import torch

from runners.splitting import collect_run_batches, normalize_max_sampling_batch_size
from utils import validate_split_percentages


def model_input_dim(model) -> int:
    input_dim = getattr(model, "input_dim", None)
    if input_dim is not None:
        return int(input_dim)
    orig_model = getattr(model, "_orig_mod", None)
    if orig_model is not None and getattr(orig_model, "input_dim", None) is not None:
        return int(orig_model.input_dim)
    raise AttributeError("Could not determine model input dimension")


class DDPM:
    r"""Segmentable ancestral DDPM reverse process.

    The timestep schedule and reverse transition are delegated to diffusers'
    ``DDPMScheduler``.  Segment boundaries use the existing half-open time
    convention ``end_t <= t < start_t``; because ``scheduler.step`` always
    consults the complete inference schedule, consecutive segments compose to
    exactly the same trajectory as a single uninterrupted call when they share
    a generator.
    """

    def __init__(
        self,
        T=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        device="cpu",
        sampling_steps=None,
        variance_type="fixed_small",
        prediction_type="epsilon",
        thresholding=False,
        dynamic_thresholding_ratio=0.995,
        clip_sample=True,
        clip_sample_range=1.0,
        sample_max_value=1.0,
        timestep_spacing="leading",
        steps_offset=0,
        rescale_betas_zero_snr=False,
    ):
        from diffusers import DDPMScheduler

        self.device = str(device)
        self.T = int(T)
        self.sampling_steps = (
            self.T if sampling_steps is None else int(sampling_steps)
        )
        if self.T < 1:
            raise ValueError("T must be at least 1")
        if self.sampling_steps < 1:
            raise ValueError("sampling_steps must be at least 1")
        if self.sampling_steps > self.T:
            raise ValueError("sampling_steps cannot exceed T")

        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.T,
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            beta_schedule=str(beta_schedule),
            trained_betas=trained_betas,
            variance_type=str(variance_type),
            prediction_type=str(prediction_type),
            thresholding=bool(thresholding),
            dynamic_thresholding_ratio=float(dynamic_thresholding_ratio),
            clip_sample=bool(clip_sample),
            clip_sample_range=float(clip_sample_range),
            sample_max_value=float(sample_max_value),
            timestep_spacing=str(timestep_spacing),
            steps_offset=int(steps_offset),
            rescale_betas_zero_snr=bool(rescale_betas_zero_snr),
        )
        self.scheduler.set_timesteps(self.sampling_steps, device=self.device)
        self.timesteps = [int(t) for t in self.scheduler.timesteps.detach().cpu()]
        if len(self.timesteps) != self.sampling_steps:
            raise RuntimeError(
                "DDPMScheduler returned an unexpected number of timesteps: "
                f"{len(self.timesteps)} != {self.sampling_steps}"
            )
        if len(set(self.timesteps)) != len(self.timesteps):
            raise ValueError("DDPM inference schedule contains duplicate timesteps")
        self._valid_boundaries = {0, self.T, *(t + 1 for t in self.timesteps)}
        self._segment_timestep_cache: dict[tuple[int, int], list[int]] = {}

    def _segment_timesteps(self, start_t: int, end_t: int) -> list[int]:
        key = (int(start_t), int(end_t))
        invalid = [boundary for boundary in key if boundary not in self._valid_boundaries]
        if invalid:
            raise ValueError(
                "DDPM segment boundaries must align with the inference schedule; "
                f"invalid boundaries: {invalid}"
            )
        if key[0] < key[1]:
            raise ValueError(
                f"DDPM segment must move toward lower time, got {key[0]} -> {key[1]}"
            )
        cached = self._segment_timestep_cache.get(key)
        if cached is None:
            cached = [t for t in self.timesteps if key[1] <= t < key[0]]
            self._segment_timestep_cache[key] = cached
        return cached

    def segment_cost(self, start_t: int, end_t: int) -> int:
        return len(self._segment_timesteps(start_t, end_t))

    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        timestep: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        t = int(timestep)
        t_model = torch.full(
            (x_t.shape[0],),
            t,
            device=x_t.device,
            dtype=torch.long,
        )
        model_output = model(x_t, t_model)
        return self.scheduler.step(
            model_output,
            t,
            x_t,
            generator=generator,
        ).prev_sample

    def sample_loop(
        self,
        model,
        x_T: torch.Tensor,
        start_t: int,
        end_t: int,
        generator: torch.Generator | None = None,
        progress_callback=None,
    ) -> torch.Tensor:
        x = x_T
        with torch.inference_mode():
            for timestep in self._segment_timesteps(start_t, end_t):
                x = self.p_sample(
                    model,
                    x,
                    timestep,
                    generator=generator,
                )
                if progress_callback is not None:
                    progress_callback(1)
        return x


class DDIM:
    r"""DDIM noise schedule and reverse process."""

    def __init__(
        self,
        T=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        device="cpu",
        eta=1.0,
        sampling_steps=None,
        prediction_type="epsilon",
        thresholding=False,
        clip_sample=True,
        clip_sample_range=1.0,
        timestep_spacing="leading",
        steps_offset=0,
        rescale_betas_zero_snr=False,
    ):
        self.T = T
        self.device = device
        self.eta = eta
        self.sampling_steps = sampling_steps if sampling_steps is not None else T
        self.prediction_type = str(prediction_type)
        self.thresholding = bool(thresholding)
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)
        self.timestep_spacing = str(timestep_spacing)
        self.steps_offset = int(steps_offset)

        if self.thresholding:
            raise NotImplementedError("DDIM dynamic thresholding is not implemented")
        if self.prediction_type not in {"epsilon", "sample", "v_prediction"}:
            raise ValueError(f"Unsupported DDIM prediction_type={self.prediction_type!r}")
        if rescale_betas_zero_snr:
            raise NotImplementedError("rescale_betas_zero_snr is not implemented")

        self.betas = self._make_betas(
            trained_betas=trained_betas,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self._create_timestep_schedule()
        self._segment_timestep_cache = {}

    def _make_betas(self, *, trained_betas, beta_start, beta_end, beta_schedule):
        if trained_betas is not None:
            return torch.as_tensor(trained_betas, dtype=torch.float32, device=self.device)
        if beta_schedule == "linear":
            return torch.linspace(beta_start, beta_end, self.T, device=self.device)
        if beta_schedule == "scaled_linear":
            return torch.linspace(
                math.sqrt(beta_start), math.sqrt(beta_end), self.T, device=self.device
            ) ** 2
        raise ValueError(f"Unsupported DDIM beta_schedule={beta_schedule!r}")

    def _create_timestep_schedule(self):
        if self.sampling_steps < 1:
            raise ValueError("sampling_steps must be at least 1")
        if self.sampling_steps > self.T:
            raise ValueError("sampling_steps cannot exceed T")

        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.T - 1, self.sampling_steps).round()[::-1]
        elif self.timestep_spacing == "leading":
            step_ratio = self.T // self.sampling_steps
            timesteps = (np.arange(0, self.sampling_steps) * step_ratio).round()[::-1]
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.T / self.sampling_steps
            timesteps = np.round(np.arange(self.T, 0, -step_ratio)) - 1
        else:
            raise ValueError(f"Unsupported timestep_spacing={self.timestep_spacing!r}")
        timesteps = np.clip(timesteps.astype(np.int64), 0, self.T - 1)
        self.timesteps = timesteps.tolist()

    def segment_cost(self, start_t: int, end_t: int) -> int:
        return len(self._segment_timesteps(start_t, end_t))

    def _segment_timesteps(self, start_t: int, end_t: int):
        key = (int(start_t), int(end_t))
        cached = self._segment_timestep_cache.get(key)
        if cached is None:
            cached = [t for t in self.timesteps if key[1] <= t < key[0]]
            self._segment_timestep_cache[key] = cached
        return cached

    def p_sample(self, model, x_t, t, t_prev, generator: torch.Generator | None = None):
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        model_output = model(x_t, t_tensor)

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = (
            self.alphas_cumprod[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=self.device)
        )

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        if self.prediction_type == "epsilon":
            eps_pred = model_output
            x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
        elif self.prediction_type == "sample":
            x0_pred = model_output
            eps_pred = (x_t - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t
        else:
            x0_pred = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * model_output
            eps_pred = sqrt_alpha_bar_t * model_output + sqrt_one_minus_alpha_bar_t * x_t

        if self.clip_sample:
            x0_pred = x0_pred.clamp(
                -self.clip_sample_range,
                self.clip_sample_range,
            )

        if t_prev >= 0 and self.eta > 0:
            sigma = self.eta * torch.sqrt(
                (1.0 - alpha_bar_t_prev)
                / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_t_prev)
            )
        else:
            sigma = torch.tensor(0.0, device=self.device)

        sqrt_one_minus_alpha_bar_t_prev_minus_sigma_sq = torch.sqrt(
            torch.clamp(1.0 - alpha_bar_t_prev - sigma**2, min=0.0)
        )

        sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
        x_prev = (
            sqrt_alpha_bar_t_prev * x0_pred
            + sqrt_one_minus_alpha_bar_t_prev_minus_sigma_sq * eps_pred
        )

        if t_prev >= 0 and self.eta > 0:
            noise = torch.randn(
                x_t.shape,
                device=x_t.device,
                dtype=x_t.dtype,
                generator=generator,
            )
            x_prev = x_prev + sigma * noise

        return x_prev

    def sample_loop(
        self,
        model,
        x_T,
        start_t,
        end_t,
        generator: torch.Generator | None = None,
        progress_callback=None,
    ):
        x = x_T
        relevant_timesteps = self._segment_timesteps(start_t, end_t)

        with torch.inference_mode():
            for i, t in enumerate(relevant_timesteps):
                if i + 1 < len(relevant_timesteps):
                    t_prev = relevant_timesteps[i + 1]
                else:
                    t_prev = end_t - 1
                x = self.p_sample(model, x, t, t_prev, generator=generator)
                if progress_callback is not None:
                    progress_callback(1)

        return x


def sample_segment(
    sampler: DDPM | DDIM,
    model,
    x: torch.Tensor,
    start_t: int,
    end_t: int,
    generator: torch.Generator | None = None,
    progress_callback=None,
):
    if x.shape[0] == 0:
        return x
    return sampler.sample_loop(
        model,
        x,
        start_t,
        end_t,
        generator=generator,
        progress_callback=progress_callback,
    )


def resolve_split_percentages(
    sampler: DDPM | DDIM, split_percentages: Sequence[float]
) -> Tuple[List[int], List[int]]:
    validate_split_percentages(split_percentages)
    remaining_steps = [
        int(round(sampler.sampling_steps * pct)) for pct in split_percentages
    ]

    for i, steps_left in enumerate(remaining_steps):
        if steps_left <= 0 or steps_left >= sampler.sampling_steps:
            raise ValueError(
                "Each split percentage must map to an interior split. "
                f"Got round({sampler.sampling_steps} * {split_percentages[i]}) = {steps_left}."
            )

    for i in range(len(remaining_steps) - 1):
        if remaining_steps[i] <= remaining_steps[i + 1]:
            raise ValueError(
                "Rounded split points must be strictly decreasing. "
                f"Got {remaining_steps[i]} <= {remaining_steps[i + 1]} from split_percentages "
                f"{split_percentages[i]} and {split_percentages[i + 1]}."
            )

    split_points = [
        int(sampler.timesteps[sampler.sampling_steps - steps_left] + 1)
        for steps_left in remaining_steps
    ]
    return remaining_steps, split_points


def sample_full_solver(
    model,
    solver: str,
    x: torch.Tensor,
    *,
    T: int,
    sampling_steps: int,
    eta: float,
    device: str,
    generator: torch.Generator | None = None,
):
    if solver == "ddim":
        sampler = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
        return sampler.sample_loop(model, x, sampler.T, 0, generator=generator)
    raise ValueError(f"Unknown solver baseline '{solver}'")


def run_solver_baseline_batch(
    model,
    postprocess_fn,
    *,
    solver: str,
    chunk_size: int,
    n0: int,
    T: int,
    sampling_steps: int,
    eta: float,
    device: str,
    generator: torch.Generator | None = None,
    max_sampling_batch_size=None,
):
    input_dim = model_input_dim(model)
    max_sampling_batch_size = normalize_max_sampling_batch_size(
        max_sampling_batch_size
    )
    sampling_start = time.perf_counter()
    if max_sampling_batch_size is not None:
        counts_by_run = [int(n0)] * int(chunk_size)

        def sample_batch(total):
            x = torch.randn(total, input_dim, device=device, generator=generator)
            samples = sample_full_solver(
                model,
                solver,
                x,
                T=T,
                sampling_steps=sampling_steps,
                eta=eta,
                device=device,
                generator=generator,
            )
            return postprocess_fn(samples).to(dtype=torch.float32)

        if int(chunk_size) < 1 or int(n0) < 1:
            sampling_time = time.perf_counter() - sampling_start
            empty_template = postprocess_fn(
                torch.empty((0, input_dim), device=device)
            ).to(dtype=torch.float32)
            return [empty_template[:0] for _ in counts_by_run], sampling_time

        samples_by_run = collect_run_batches(
            counts_by_run,
            max_sampling_batch_size,
            sample_batch=sample_batch,
            empty_template=torch.empty(0, device=device),
        )
        sampling_time = time.perf_counter() - sampling_start
        return samples_by_run, sampling_time

    x = torch.randn(chunk_size * n0, input_dim, device=device, generator=generator)
    samples = sample_full_solver(
        model,
        solver,
        x,
        T=T,
        sampling_steps=sampling_steps,
        eta=eta,
        device=device,
        generator=generator,
    )
    samples = postprocess_fn(samples)
    sampling_time = time.perf_counter() - sampling_start
    samples = samples.to(dtype=torch.float32).reshape(chunk_size, n0, -1).contiguous()
    return [samples[i] for i in range(chunk_size)], sampling_time
