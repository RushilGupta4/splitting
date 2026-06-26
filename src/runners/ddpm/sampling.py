import time
from typing import List, Sequence, Tuple

import numpy as np
import torch

from runners.splitting import balanced_split_with_run_ids
from utils import validate_split_percentages


def model_input_dim(model) -> int:
    input_dim = getattr(model, "input_dim", None)
    if input_dim is not None:
        return int(input_dim)
    orig_model = getattr(model, "_orig_mod", None)
    if orig_model is not None and getattr(orig_model, "input_dim", None) is not None:
        return int(orig_model.input_dim)
    raise AttributeError("Could not determine model input dimension")


class DDIM:
    r"""DDIM noise schedule and reverse process."""

    def __init__(
        self,
        T=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
        eta=1.0,
        sampling_steps=None,
    ):
        self.T = T
        self.device = device
        self.eta = eta
        self.sampling_steps = sampling_steps if sampling_steps is not None else T

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self._create_timestep_schedule()

    def _create_timestep_schedule(self):
        if self.sampling_steps == self.T:
            self.timesteps = list(range(self.T - 1, -1, -1))
        else:
            step_size = max(1, self.T // self.sampling_steps)
            self.timesteps = list(range(self.T - 1, -1, -step_size))[
                : self.sampling_steps
            ]
            if self.timesteps[-1] != 0:
                self.timesteps[-1] = 0

    def segment_cost(self, start_t: int, end_t: int) -> int:
        return sum(1 for t in self.timesteps if end_t <= t < start_t)

    def p_sample(self, model, x_t, t, t_prev, generator: torch.Generator | None = None):
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        eps_pred = model(x_t, t_tensor)

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = (
            self.alphas_cumprod[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=self.device)
        )

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

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
        self, model, x_T, start_t, end_t, generator: torch.Generator | None = None
    ):
        x = x_T
        relevant_timesteps = [t for t in self.timesteps if end_t <= t < start_t]

        with torch.inference_mode():
            for i, t in enumerate(relevant_timesteps):
                if i + 1 < len(relevant_timesteps):
                    t_prev = relevant_timesteps[i + 1]
                else:
                    t_prev = end_t - 1
                x = self.p_sample(model, x, t, t_prev, generator=generator)

        return x


def sample_segment(
    ddim: DDIM,
    model,
    x: torch.Tensor,
    start_t: int,
    end_t: int,
    generator: torch.Generator | None = None,
):
    if x.shape[0] == 0:
        return x
    return ddim.sample_loop(model, x, start_t, end_t, generator=generator)


def resolve_split_percentages(
    ddim: DDIM, split_percentages: Sequence[float]
) -> Tuple[List[int], List[int]]:
    validate_split_percentages(split_percentages)
    remaining_steps = [
        int(round(ddim.sampling_steps * pct)) for pct in split_percentages
    ]

    for i, steps_left in enumerate(remaining_steps):
        if steps_left <= 0 or steps_left >= ddim.sampling_steps:
            raise ValueError(
                "Each split percentage must map to an interior split. "
                f"Got round({ddim.sampling_steps} * {split_percentages[i]}) = {steps_left}."
            )

    for i in range(len(remaining_steps) - 1):
        if remaining_steps[i] <= remaining_steps[i + 1]:
            raise ValueError(
                "Rounded split points must be strictly decreasing. "
                f"Got {remaining_steps[i]} <= {remaining_steps[i + 1]} from split_percentages "
                f"{split_percentages[i]} and {split_percentages[i + 1]}."
            )

    split_points = [
        int(ddim.timesteps[ddim.sampling_steps - steps_left] + 1)
        for steps_left in remaining_steps
    ]
    return remaining_steps, split_points


def segment_costs(ddim: DDIM, split_points: Sequence[int]) -> List[float]:
    start_points = [ddim.T] + list(split_points)
    end_points = list(split_points) + [0]
    return [
        float(ddim.segment_cost(start_t, end_t))
        for start_t, end_t in zip(start_points, end_points)
    ]


def expected_cost_per_root(
    ddim: DDIM, split_points: Sequence[int], split_factors: Sequence[float]
):
    cost = float(ddim.segment_cost(ddim.T, split_points[0]))
    cumulative_split = 1.0

    for idx, split_factor in enumerate(split_factors):
        cumulative_split *= split_factor
        if idx + 1 < len(split_points):
            segment_cost = ddim.segment_cost(split_points[idx], split_points[idx + 1])
        else:
            segment_cost = ddim.segment_cost(split_points[idx], 0)
        cost += cumulative_split * segment_cost

    return cost


def run_probabilistic_inference_batch(
    model,
    ddim: DDIM,
    n0_by_run: Sequence[int],
    split_points: Sequence[int],
    split_factors_by_run: Sequence[Sequence[float]],
    postprocess_fn,
    generator: torch.Generator | None = None,
):
    if len(n0_by_run) == 0:
        return [], [], 0.0
    if len(n0_by_run) != len(split_factors_by_run):
        raise ValueError("n0_by_run and split_factors_by_run must have the same length")
    if any(int(n0) < 1 for n0 in n0_by_run):
        raise ValueError("all n0 values must be at least 1")

    input_dim = int(model_input_dim(model))
    n0_tensor = torch.as_tensor(n0_by_run, device=ddim.device, dtype=torch.long)

    num_runs = int(n0_tensor.numel())
    run_ids = torch.repeat_interleave(
        torch.arange(num_runs, device=ddim.device, dtype=torch.long), n0_tensor
    )
    x = torch.randn(
        int(n0_tensor.sum().item()), input_dim, device=ddim.device, generator=generator
    )

    sampling_start = time.perf_counter()

    if not split_points:
        realized_costs = n0_tensor * int(ddim.segment_cost(ddim.T, 0))
        x = sample_segment(ddim, model, x, ddim.T, 0, generator=generator)
        x = postprocess_fn(x)
        sampling_time = time.perf_counter() - sampling_start
        samples_by_run = []
        for run_idx in range(num_runs):
            samples_by_run.append(x[run_ids == run_idx])
        return samples_by_run, realized_costs.detach().cpu().tolist(), sampling_time

    realized_costs = n0_tensor * int(ddim.segment_cost(ddim.T, split_points[0]))
    x = sample_segment(ddim, model, x, ddim.T, split_points[0], generator=generator)

    split_factors_array = np.asarray(split_factors_by_run, dtype=float)
    if not np.isfinite(split_factors_array).all() or np.any(split_factors_array < 0.0):
        raise ValueError("split_factors_by_run must contain finite nonnegative values")
    split_factors_tensor = torch.as_tensor(
        split_factors_array, device=ddim.device, dtype=x.dtype
    )
    if split_factors_tensor.shape != (num_runs, len(split_points)):
        raise ValueError(
            "split_factors_by_run must have shape "
            f"({num_runs}, {len(split_points)}), got {tuple(split_factors_tensor.shape)}"
        )

    for idx, split_point in enumerate(split_points):
        x, run_ids = balanced_split_with_run_ids(
            x,
            run_ids,
            split_factors_tensor[:, idx],
            generator=generator,
        )

        if idx + 1 < len(split_points):
            end_t = split_points[idx + 1]
        else:
            end_t = 0

        current_counts = torch.bincount(run_ids, minlength=num_runs)
        realized_costs = realized_costs + current_counts * int(
            ddim.segment_cost(split_point, end_t)
        )
        x = sample_segment(ddim, model, x, split_point, end_t, generator=generator)

    x = postprocess_fn(x)
    sampling_time = time.perf_counter() - sampling_start

    samples_by_run = []
    for run_idx in range(num_runs):
        samples_by_run.append(x[run_ids == run_idx])

    return samples_by_run, realized_costs.detach().cpu().tolist(), sampling_time


def sample_dpmpp_2m(model, x: torch.Tensor, T: int, sampling_steps: int):
    from diffusers import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=T,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    scheduler.set_timesteps(sampling_steps, device=x.device)
    scheduler._step_index = 0

    with torch.inference_mode():
        for t in scheduler.timesteps:
            if isinstance(t, torch.Tensor):
                t_tensor = t.to(device=x.device, dtype=torch.long).expand(x.shape[0])
            else:
                t_tensor = torch.full(
                    (x.shape[0],), int(t), device=x.device, dtype=torch.long
                )
            eps_pred = model(x, t_tensor)
            x = scheduler.step(eps_pred, t, x).prev_sample
    return x


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
        ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
        return ddim.sample_loop(model, x, ddim.T, 0, generator=generator)
    if solver == "dpmpp_2m":
        return sample_dpmpp_2m(model, x, T=T, sampling_steps=sampling_steps)
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
):
    input_dim = model_input_dim(model)
    sampling_start = time.perf_counter()
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
