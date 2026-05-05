import math
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from numba import njit
from tqdm import tqdm

from model import Denoiser
from utils import (
    denormalize,
    get_checkpoint_normalization_stats,
    get_checkpoint_target_spec,
    infer_input_dim_from_checkpoint,
    mixture_lower_orthant_cdf,
    validate_split_percentages,
)


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

    def p_sample(self, model, x_t, t, t_prev):
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

        if t_prev >= 0 and sigma > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise

        return x_prev

    def sample_loop(self, model, x_T, start_t, end_t):
        x = x_T
        relevant_timesteps = [t for t in self.timesteps if end_t <= t < start_t]

        with torch.inference_mode():
            for i, t in enumerate(relevant_timesteps):
                if i + 1 < len(relevant_timesteps):
                    t_prev = relevant_timesteps[i + 1]
                else:
                    t_prev = end_t - 1
                x = self.p_sample(model, x, t, t_prev)

        return x


def _coerce_samples_np(samples) -> np.ndarray:
    samples_np = (
        samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    )
    samples_np = np.asarray(samples_np, dtype=float).reshape(-1, 2)
    if samples_np.ndim != 2 or samples_np.shape[1] != 2:
        raise ValueError(f"Expected samples of shape [N, 2], got {samples_np.shape}")
    return samples_np


def _prepare_empirical_cdf_state(samples) -> Dict[str, Any]:
    samples_np = _coerce_samples_np(samples)
    if samples_np.shape[0] == 0:
        raise ValueError("Cannot prepare an empirical CDF state from zero samples")

    y_values, y_ranks = np.unique(samples_np[:, 1], return_inverse=True)
    x_order = np.argsort(samples_np[:, 0], kind="mergesort")
    return {
        "count": int(samples_np.shape[0]),
        "x_sorted": np.ascontiguousarray(samples_np[x_order, 0], dtype=np.float64),
        "y_ranks_sorted": np.ascontiguousarray(
            y_ranks[x_order].astype(np.int64) + 1, dtype=np.int64
        ),
        "y_values": np.ascontiguousarray(y_values, dtype=np.float64),
    }


def prepare_reference_cdf_state(samples) -> Dict[str, Any]:
    return _prepare_empirical_cdf_state(samples)


@njit(cache=True)
def _fenwick_add(tree: np.ndarray, index: int):
    while index < tree.shape[0]:
        tree[index] += 1
        index += index & -index


@njit(cache=True)
def _fenwick_prefix_sum(tree: np.ndarray, index: int) -> int:
    total = 0
    while index > 0:
        total += int(tree[index])
        index -= index & -index
    return total


@njit(cache=True)
def _upper_bound(sorted_values: np.ndarray, target: float) -> int:
    left = 0
    right = sorted_values.shape[0]
    while left < right:
        mid = (left + right) // 2
        if sorted_values[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left


@njit(cache=True)
def _empirical_lower_orthant_cdf_numba(
    x_sorted: np.ndarray,
    y_ranks_sorted: np.ndarray,
    y_values: np.ndarray,
    points_np: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    query_order = np.argsort(points_np[:, 0])
    counts = np.empty(points_np.shape[0], dtype=np.float64)
    tree = np.zeros(y_values.shape[0] + 1, dtype=np.int64)

    sample_idx = 0
    for ordered_idx in range(query_order.shape[0]):
        query_idx = query_order[ordered_idx]
        query_x = points_np[query_idx, 0]
        while sample_idx < sample_count and x_sorted[sample_idx] <= query_x:
            _fenwick_add(tree, int(y_ranks_sorted[sample_idx]))
            sample_idx += 1
        y_limit = _upper_bound(y_values, points_np[query_idx, 1])
        counts[query_idx] = _fenwick_prefix_sum(tree, y_limit) / float(sample_count)

    return counts


def _empirical_lower_orthant_cdf_from_state(
    state: Dict[str, Any], points: np.ndarray
) -> np.ndarray:
    points_np = np.ascontiguousarray(
        np.asarray(points, dtype=np.float64).reshape(-1, 2)
    )
    if points_np.shape[0] == 0:
        return np.empty(0, dtype=float)
    return _empirical_lower_orthant_cdf_numba(
        state["x_sorted"],
        state["y_ranks_sorted"],
        state["y_values"],
        points_np,
        int(state["count"]),
    )


def warm_reference_ks_kernel(reference_cdf_state: Dict[str, Any]):
    sample_x = reference_cdf_state["x_sorted"][0]
    sample_y = reference_cdf_state["y_values"][0]
    warm_points = np.ascontiguousarray([[sample_x, sample_y]], dtype=np.float64)
    _empirical_lower_orthant_cdf_numba(
        reference_cdf_state["x_sorted"],
        reference_cdf_state["y_ranks_sorted"],
        reference_cdf_state["y_values"],
        warm_points,
        int(reference_cdf_state["count"]),
    )


def compute_reference_ks_distance(samples, reference_cdf_state: Dict[str, Any]):
    """Compute 2D lower-orthant KS distance against a cached empirical reference."""
    samples_np = _coerce_samples_np(samples)
    candidate_points = np.unique(samples_np, axis=0)
    sample_cdf_state = _prepare_empirical_cdf_state(samples_np)
    empirical_cdf = _empirical_lower_orthant_cdf_from_state(
        sample_cdf_state, candidate_points
    )
    reference_cdf = _empirical_lower_orthant_cdf_from_state(
        reference_cdf_state, candidate_points
    )
    ks_distance = float(np.max(np.abs(empirical_cdf - reference_cdf)))
    return ks_distance, empirical_cdf, reference_cdf


def compute_target_ks_distance(samples, target_spec: Dict[str, Any]):
    """Compute 2D lower-orthant KS distance against the exact target CDF."""
    samples_np = _coerce_samples_np(samples)
    candidate_points = np.unique(samples_np, axis=0)
    sample_cdf_state = _prepare_empirical_cdf_state(samples_np)
    empirical_cdf = _empirical_lower_orthant_cdf_from_state(
        sample_cdf_state, candidate_points
    )
    target_cdf = mixture_lower_orthant_cdf(candidate_points, target_spec)
    ks_distance = float(np.max(np.abs(empirical_cdf - target_cdf)))
    return ks_distance, empirical_cdf, target_cdf


def _compute_ks_distance(
    samples,
    target_spec: Dict[str, Any],
    reference_mode: str,
    reference_cdf_state: Dict[str, Any] | None,
):
    if reference_mode == "samples":
        if reference_cdf_state is None:
            raise ValueError(
                "reference_cdf_state is required when reference_mode='samples'"
            )
        return compute_reference_ks_distance(samples, reference_cdf_state)
    if reference_mode == "true_dist":
        return compute_target_ks_distance(samples, target_spec)
    raise ValueError(f"Unknown reference_mode '{reference_mode}'")


def _sample_loop(ddim: DDIM, model, x: torch.Tensor, start_t: int, end_t: int):
    if x.shape[0] == 0:
        return x
    return ddim.sample_loop(model, x, start_t, end_t)


def _parse_split_percentages(split_percentages_str: str) -> List[float]:
    split_percentages = [
        float(x.strip()) for x in split_percentages_str.split(",") if x.strip()
    ]
    validate_split_percentages(split_percentages)
    return split_percentages


def _parse_x_grid(x_grid_str: str) -> List[float]:
    x_grid = [float(x.strip()) for x in x_grid_str.split(",") if x.strip()]
    if not x_grid:
        raise ValueError("x_grid must have at least one threshold")
    return x_grid


def _debug_log(enabled: bool, message: str):
    if enabled:
        print(f"[debug] {message}")


def _resolve_split_percentages(
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


def _load_model_and_stats(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_args = checkpoint["args"]
    input_dim = infer_input_dim_from_checkpoint(checkpoint)

    model = Denoiser(
        input_dim=input_dim,
        hidden_dim=model_args.get("hidden_dim", 128),
        num_blocks=model_args.get("num_blocks", 4),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if hasattr(torch, "compile"):
        model = torch.compile(model, dynamic=True)
        _debug_log(
            getattr(args, "debug", False), "Compiled denoiser with torch.compile"
        )

    target_spec = get_checkpoint_target_spec(checkpoint)
    data_mean, data_std = get_checkpoint_normalization_stats(
        checkpoint, device=args.device
    )
    return model, target_spec, data_mean, data_std


def _model_input_dim(model) -> int:
    input_dim = getattr(model, "input_dim", None)
    if input_dim is not None:
        return int(input_dim)
    orig_model = getattr(model, "_orig_mod", None)
    if orig_model is not None and getattr(orig_model, "input_dim", None) is not None:
        return int(orig_model.input_dim)
    raise AttributeError("Could not determine model input dimension")


def _pilot_cost_per_root(ddim: DDIM, split_points: Sequence[int], m_pilot: float):
    start_points = [ddim.T] + list(split_points)
    end_points = list(split_points) + [0]
    cost = 0.0
    for idx, (start_t, end_t) in enumerate(zip(start_points, end_points)):
        cost += (m_pilot ** (idx + 1)) * ddim.segment_cost(start_t, end_t)
    return cost


def _pilot_tree_shape_from_scale(pilot_scale: float):
    if pilot_scale <= 1.0:
        raise ValueError("pilot_scale must be > 1")
    pilot_roots = max(1, int(pilot_scale * pilot_scale))
    return pilot_roots, float(pilot_scale)


def _pilot_tree_expected_cost(
    ddim: DDIM, split_points: Sequence[int], pilot_scale: float
):
    pilot_roots, m_pilot = _pilot_tree_shape_from_scale(pilot_scale)
    return pilot_roots * _pilot_cost_per_root(ddim, split_points, m_pilot)


def _derive_pilot_tree_shape(ddim: DDIM, split_points: Sequence[int], B1: int):
    # Use pilot_scale = y so pilot_roots ~ y^2 and per-level branching ~ y.
    min_pilot_scale = 1.0 + 1e-6
    min_cost = _pilot_tree_expected_cost(ddim, split_points, min_pilot_scale)
    if B1 < min_cost:
        raise ValueError(
            f"B1={B1} is too small for pilot_tree; need at least {min_cost:.6f}"
        )

    lower = min_pilot_scale
    upper = 2.0
    while _pilot_tree_expected_cost(ddim, split_points, upper) <= B1:
        upper *= 2.0

    for _ in range(80):
        mid = 0.5 * (lower + upper)
        if _pilot_tree_expected_cost(ddim, split_points, mid) <= B1:
            lower = mid
        else:
            upper = mid

    pilot_roots, m_pilot = _pilot_tree_shape_from_scale(lower)
    return float(lower), int(pilot_roots), float(m_pilot)


def _independent_sigma_cost(
    ddim: DDIM,
    t_curr: int,
    t_next: int,
    outer_count: int,
    middle_count: int,
    inner_count: int,
):
    return (
        outer_count * ddim.segment_cost(ddim.T, t_curr)
        + outer_count * middle_count * ddim.segment_cost(t_curr, t_next)
        + outer_count * middle_count * inner_count * ddim.segment_cost(t_next, 0)
    )


def _derive_independent_counts(
    ddim: DDIM, sigma_times: Sequence[int], B1: int, independent_n2: int
) -> Tuple[float, List[Dict[str, int]], int]:
    if not sigma_times:
        raise ValueError("sigma_times must be non-empty")
    if independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")

    budget_per_sigma = B1 / len(sigma_times)
    if budget_per_sigma <= 0:
        raise ValueError("B1 must be positive for independent sigma estimation")

    counts_by_sigma = []
    total_used_B1 = 0
    next_times = list(sigma_times[1:]) + [0]

    for t_curr, t_next in zip(sigma_times, next_times):
        segment1 = ddim.segment_cost(ddim.T, t_curr)
        segment2 = ddim.segment_cost(t_curr, t_next)
        segment3 = ddim.segment_cost(t_next, 0)
        min_cost = segment1 + independent_n2 * segment2 + independent_n2 * segment3
        if budget_per_sigma < min_cost:
            raise ValueError(
                "B1 is too small for equal per-sigma independent estimation with "
                f"independent_n2={independent_n2}; budget_per_sigma={budget_per_sigma:.6f}, "
                f"minimum required={min_cost:.6f} at sigma time {t_curr}"
            )

        def cost_for_y(y: float):
            return (y**2) * (segment1 + independent_n2 * segment2) + (y**3) * (
                independent_n2 * segment3
            )

        lower = 1.0
        upper = 2.0
        while cost_for_y(upper) <= budget_per_sigma:
            upper *= 2.0

        for _ in range(80):
            mid = 0.5 * (lower + upper)
            if cost_for_y(mid) <= budget_per_sigma:
                lower = mid
            else:
                upper = mid

        outer_count = max(1, int(lower * lower))
        # For the final sigma block, t_next == 0, so inner repeats are exact
        # zero-step duplicates. They do not change the sigma estimate but would
        # overweight those x0 samples when reuse_phase1_samples=True.
        inner_count = 1 if segment3 == 0 else max(1, int(lower))
        used_budget = _independent_sigma_cost(
            ddim, t_curr, t_next, outer_count, independent_n2, inner_count
        )

        counts_by_sigma.append(
            {
                "time": int(t_curr),
                "next_time": int(t_next),
                "outer_count": int(outer_count),
                "middle_count": int(independent_n2),
                "inner_count": int(inner_count),
                "used_budget": int(used_budget),
            }
        )
        total_used_B1 += used_budget

    return budget_per_sigma, counts_by_sigma, int(total_used_B1)


def _sample_branch_counts(num_parents: int, branching_factor: float, device: str):
    if branching_factor <= 1.0:
        raise ValueError("Branching factor must be > 1 so parents can have children")

    base_copies = math.floor(branching_factor)
    extra_prob = branching_factor - base_copies
    counts = torch.full((num_parents,), base_copies, device=device, dtype=torch.long)
    if extra_prob > 0.0:
        counts = counts + (torch.rand(num_parents, device=device) < extra_prob).to(
            torch.long
        )
    return counts


def _repeat_by_counts(x: torch.Tensor, counts: torch.Tensor):
    return x.repeat_interleave(counts, dim=0)


def _actual_pilot_cost(
    ddim: DDIM,
    split_points: Sequence[int],
    level_counts: Sequence[int],
    leaf_count: int,
):
    cost = level_counts[0] * ddim.segment_cost(ddim.T, split_points[0])
    for idx in range(1, len(split_points)):
        cost += level_counts[idx] * ddim.segment_cost(
            split_points[idx - 1], split_points[idx]
        )
    cost += leaf_count * ddim.segment_cost(split_points[-1], 0)
    return int(cost)


def _build_pilot_tree(
    model,
    ddim: DDIM,
    pilot_roots: int,
    split_points: Sequence[int],
    m_pilot: float,
):
    input_dim = _model_input_dim(model)
    root_x_T = torch.randn(pilot_roots, input_dim, device=ddim.device)
    x = root_x_T
    level_counts = []
    child_counts_by_level = []

    for idx, split_point in enumerate(split_points):
        child_counts = _sample_branch_counts(x.shape[0], m_pilot, ddim.device)
        child_counts_by_level.append(child_counts.cpu())
        x = _repeat_by_counts(x, child_counts)
        start_t = ddim.T if idx == 0 else split_points[idx - 1]
        x = _sample_loop(ddim, model, x, start_t, split_point)
        level_counts.append(int(x.shape[0]))

    child_counts = _sample_branch_counts(x.shape[0], m_pilot, ddim.device)
    child_counts_by_level.append(child_counts.cpu())
    x = _repeat_by_counts(x, child_counts)
    x = _sample_loop(ddim, model, x, split_points[-1], 0)

    leaf_x0 = x
    leaf_count = int(leaf_x0.shape[0])
    actual_cost = _actual_pilot_cost(ddim, split_points, level_counts, leaf_count)
    return (
        root_x_T,
        leaf_x0,
        level_counts,
        leaf_count,
        child_counts_by_level,
        actual_cost,
    )


def _rectangle_indicator_grid(values: torch.Tensor, x_grid: Sequence[float]):
    thresholds = torch.as_tensor(x_grid, device=values.device, dtype=values.dtype)
    x1_below = values[:, 0:1] <= thresholds.unsqueeze(0)
    x2_below = values[:, 1:2] <= thresholds.unsqueeze(0)
    return (
        (x1_below.unsqueeze(2) & x2_below.unsqueeze(1))
        .to(values.dtype)
        .reshape(values.shape[0], -1)
    )


def _segment_mean_and_unbiased_var(values: torch.Tensor, counts: torch.Tensor):
    counts = counts.to(device=values.device, dtype=torch.long)
    means = torch.segment_reduce(values, reduce="mean", lengths=counts)
    valid_mask = counts >= 2
    if not valid_mask.any():
        return means, None

    sums = torch.segment_reduce(values, reduce="sum", lengths=counts)
    sums_sq = torch.segment_reduce(values * values, reduce="sum", lengths=counts)
    count_values = counts.to(dtype=values.dtype).unsqueeze(1)
    denom = torch.clamp(count_values - 1.0, min=1.0)
    variances = (sums_sq - (sums * sums) / count_values) / denom
    variances = torch.clamp(variances, min=0.0)
    return means, variances[valid_mask]


def _grid_tensor_to_nested_list(values: torch.Tensor, x_grid: Sequence[float]):
    grid_size = len(x_grid)
    return values.reshape(grid_size, grid_size).detach().cpu().tolist()


def _estimate_sigmas_from_tree(
    leaf_x0: torch.Tensor,
    sigma_times: Sequence[int],
    child_counts_by_level: Sequence[torch.Tensor],
    x_grid: Sequence[float],
):
    num_levels = len(sigma_times)
    current = _rectangle_indicator_grid(leaf_x0, x_grid)
    grid_size = len(x_grid)
    sigma2_by_level = [None] * num_levels
    sigma_by_level = [None] * num_levels
    mean_by_level = [None] * num_levels

    for rev_idx in range(num_levels - 1, -1, -1):
        counts = child_counts_by_level[rev_idx].to(
            device=current.device, dtype=torch.long
        )
        if int(counts.sum().item()) != current.shape[0]:
            raise ValueError("Pilot tree child counts do not match leaf layout")

        child_means, child_vars = _segment_mean_and_unbiased_var(current, counts)
        if child_vars is None:
            raise ValueError(
                "Pilot tree produced no parents with at least two children; increase B1"
            )

        sigma2_flat = child_vars.mean(dim=0)
        mean_flat = child_means.mean(dim=0)
        sigma2_by_level[rev_idx] = sigma2_flat
        sigma_by_level[rev_idx] = torch.sqrt(torch.clamp(sigma2_flat, min=0.0))
        mean_by_level[rev_idx] = mean_flat
        current = child_means

    if current.shape[0] >= 2:
        tau2_flat = torch.clamp(current.var(dim=0, unbiased=True), min=0.0)
    else:
        tau2_flat = torch.zeros(
            grid_size * grid_size, device=current.device, dtype=current.dtype
        )
    tau_flat = torch.sqrt(torch.clamp(tau2_flat, min=0.0))
    tau_mean_flat = current.mean(dim=0)

    sigma2_grids = [
        _grid_tensor_to_nested_list(values, x_grid) for values in sigma2_by_level
    ]
    sigma_grids = [
        _grid_tensor_to_nested_list(values, x_grid) for values in sigma_by_level
    ]
    p_hat_mean_grids = [
        _grid_tensor_to_nested_list(values, x_grid) for values in mean_by_level
    ]
    tau2_grid = _grid_tensor_to_nested_list(tau2_flat, x_grid)
    tau_grid = _grid_tensor_to_nested_list(tau_flat, x_grid)
    tau_mean_grid = _grid_tensor_to_nested_list(tau_mean_flat, x_grid)

    estimates = []
    for level_idx, sigma_time in enumerate(sigma_times):
        sigma2_grid = sigma2_grids[level_idx]
        sigma_grid = sigma_grids[level_idx]
        p_hat_mean_grid = p_hat_mean_grids[level_idx]
        sigma2_array = np.array(sigma2_grid, dtype=float)
        max_index = np.unravel_index(np.argmax(sigma2_array), sigma2_array.shape)
        estimates.append(
            {
                "time": sigma_time,
                "x_grid": list(x_grid),
                "sigma2_grid": sigma2_grid,
                "sigma_grid": sigma_grid,
                "p_hat_mean_grid": p_hat_mean_grid,
                "sigma2_max": float(sigma2_array[max_index]),
                "sigma": float(math.sqrt(max(float(sigma2_array[max_index]), 0.0))),
                "max_location": [
                    float(x_grid[max_index[0]]),
                    float(x_grid[max_index[1]]),
                ],
            }
        )

    tau2_array = np.array(tau2_grid, dtype=float)
    tau_max_index = np.unravel_index(np.argmax(tau2_array), tau2_array.shape)
    tau_estimate = {
        "x_grid": list(x_grid),
        "tau2_grid": tau2_grid,
        "tau_grid": tau_grid,
        "p_hat_mean_grid": tau_mean_grid,
        "tau2_max": float(tau2_array[tau_max_index]),
        "tau": float(math.sqrt(max(float(tau2_array[tau_max_index]), 0.0))),
        "max_location": [
            float(x_grid[tau_max_index[0]]),
            float(x_grid[tau_max_index[1]]),
        ],
    }

    return estimates, tau_estimate


def _sync_device(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _estimate_sigmas_independently(
    model,
    ddim: DDIM,
    sigma_times: Sequence[int],
    counts_by_sigma: Sequence[Dict[str, int]],
    x_grid: Sequence[float],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    collect_phase1_samples: bool = False,
):
    estimates = []
    next_times = list(sigma_times[1:]) + [0]
    if len(counts_by_sigma) != len(sigma_times):
        raise ValueError(
            f"counts_by_sigma must have length {len(sigma_times)}, got {len(counts_by_sigma)}"
        )
    used_B1 = 0
    input_dim = _model_input_dim(model)
    tau2_grid = []
    tau_grid = []
    tau_mean_grid = []
    phase1_samples = []
    sampling_time = 0.0
    aggregation_time = 0.0

    for idx, ((t_curr, t_next), count_spec) in enumerate(
        zip(zip(sigma_times, next_times), counts_by_sigma)
    ):
        outer_count = int(count_spec["outer_count"])
        middle_count = int(count_spec["middle_count"])
        inner_count = int(count_spec["inner_count"])
        sampling_start = time.perf_counter()
        root_x_T = torch.randn(outer_count, input_dim, device=ddim.device)

        x_curr = _sample_loop(ddim, model, root_x_T, ddim.T, t_curr)

        middle_counts = torch.full(
            (x_curr.shape[0],), middle_count, device=ddim.device, dtype=torch.long
        )
        x_curr_rep = _repeat_by_counts(x_curr, middle_counts)
        x_next = _sample_loop(ddim, model, x_curr_rep, t_curr, t_next)

        inner_counts = torch.full(
            (x_next.shape[0],), inner_count, device=ddim.device, dtype=torch.long
        )
        x_next_rep = _repeat_by_counts(x_next, inner_counts)
        x_0 = _sample_loop(ddim, model, x_next_rep, t_next, 0)
        x_0 = denormalize(x_0, data_mean, data_std)
        _sync_device(ddim.device)
        sampling_time += time.perf_counter() - sampling_start

        if collect_phase1_samples:
            phase1_samples.append(x_0)

        used_B1 += _independent_sigma_cost(
            ddim, t_curr, t_next, outer_count, middle_count, inner_count
        )

        aggregation_start = time.perf_counter()
        indicators = _rectangle_indicator_grid(x_0, x_grid)
        num_grid_points = indicators.shape[1]
        indicators = indicators.reshape(
            outer_count, middle_count, inner_count, num_grid_points
        )
        middle_means = indicators.mean(dim=2)
        if middle_count < 2:
            raise ValueError("Independent sigma estimation needs middle_count >= 2")

        p_hat_mean_flat = middle_means.mean(dim=(0, 1))
        outer_vars = torch.clamp(middle_means.var(dim=1, unbiased=True), min=0.0)
        sigma2_flat = outer_vars.mean(dim=0)
        sigma_flat = torch.sqrt(torch.clamp(sigma2_flat, min=0.0))
        outer_means = middle_means.mean(dim=1)
        if outer_means.shape[0] >= 2:
            tau2_flat = torch.clamp(outer_means.var(dim=0, unbiased=True), min=0.0)
        else:
            tau2_flat = torch.zeros(num_grid_points, device=x_0.device, dtype=x_0.dtype)
        tau_flat = torch.sqrt(torch.clamp(tau2_flat, min=0.0))
        tau_mean_flat = outer_means.mean(dim=0)

        sigma2_grid = _grid_tensor_to_nested_list(sigma2_flat, x_grid)
        sigma_grid = _grid_tensor_to_nested_list(sigma_flat, x_grid)
        p_hat_mean_grid = _grid_tensor_to_nested_list(p_hat_mean_flat, x_grid)
        tau2_rows_local = _grid_tensor_to_nested_list(tau2_flat, x_grid)
        tau_rows_local = _grid_tensor_to_nested_list(tau_flat, x_grid)
        tau_mean_rows_local = _grid_tensor_to_nested_list(tau_mean_flat, x_grid)

        if idx == 0:
            tau2_grid = tau2_rows_local
            tau_grid = tau_rows_local
            tau_mean_grid = tau_mean_rows_local

        sigma2_array = np.array(sigma2_grid, dtype=float)
        max_index = np.unravel_index(np.argmax(sigma2_array), sigma2_array.shape)
        estimates.append(
            {
                "time": t_curr,
                "x_grid": list(x_grid),
                "sigma2_grid": sigma2_grid,
                "sigma_grid": sigma_grid,
                "p_hat_mean_grid": p_hat_mean_grid,
                "sigma2_max": float(sigma2_array[max_index]),
                "sigma": float(math.sqrt(max(float(sigma2_array[max_index]), 0.0))),
                "max_location": [
                    float(x_grid[max_index[0]]),
                    float(x_grid[max_index[1]]),
                ],
            }
        )
        _sync_device(ddim.device)
        aggregation_time += time.perf_counter() - aggregation_start

    tau2_array = np.array(tau2_grid, dtype=float)
    tau_max_index = np.unravel_index(np.argmax(tau2_array), tau2_array.shape)
    tau_estimate = {
        "x_grid": list(x_grid),
        "tau2_grid": tau2_grid,
        "tau_grid": tau_grid,
        "p_hat_mean_grid": tau_mean_grid,
        "tau2_max": float(tau2_array[tau_max_index]),
        "tau": float(math.sqrt(max(float(tau2_array[tau_max_index]), 0.0))),
        "max_location": [
            float(x_grid[tau_max_index[0]]),
            float(x_grid[tau_max_index[1]]),
        ],
    }

    if phase1_samples:
        phase1_x0 = torch.cat(phase1_samples, dim=0)
    else:
        phase1_x0 = torch.empty((0, input_dim), device=ddim.device)

    return (
        phase1_x0,
        estimates,
        tau_estimate,
        int(used_B1),
        sampling_time,
        aggregation_time,
    )


def _build_sigma2_matrix(
    sigma_estimates: Sequence[Dict[str, Any]], tau_estimate: Dict[str, Any]
):
    if not sigma_estimates:
        raise ValueError("sigma_estimates must be non-empty")

    tau2_array = np.asarray(tau_estimate["tau2_grid"], dtype=float)
    if tau2_array.ndim != 2:
        raise ValueError(
            f"tau_estimate tau2_grid must be 2D, got shape {tau2_array.shape}"
        )
    if not np.isfinite(tau2_array).all():
        raise ValueError("tau_estimate tau2_grid contains non-finite values")

    sigma2_columns = []
    grid_shape = None
    for level_idx, estimate in enumerate(sigma_estimates):
        sigma2_array = np.asarray(estimate["sigma2_grid"], dtype=float)
        if sigma2_array.ndim != 2:
            raise ValueError(
                f"sigma_estimates[{level_idx}] sigma2_grid must be 2D, got shape {sigma2_array.shape}"
            )
        if not np.isfinite(sigma2_array).all():
            raise ValueError(
                f"sigma_estimates[{level_idx}] sigma2_grid contains non-finite values"
            )

        if grid_shape is None:
            grid_shape = sigma2_array.shape
        elif sigma2_array.shape != grid_shape:
            raise ValueError(
                "All sigma2 grids must share the same shape. "
                f"Expected {grid_shape}, got {sigma2_array.shape} at level {level_idx}."
            )

        sigma2_columns.append(np.clip(sigma2_array.reshape(-1), a_min=0.0, a_max=None))

    if tau2_array.shape != grid_shape:
        raise ValueError(
            "tau_estimate tau2_grid must match sigma grid shape. "
            f"Expected {grid_shape}, got {tau2_array.shape}."
        )

    sigma2_columns[0] = sigma2_columns[0] + np.clip(
        tau2_array.reshape(-1), a_min=0.0, a_max=None
    )

    return np.stack(sigma2_columns, axis=1), grid_shape


def _grid_index_to_point(
    flat_index: int, x_grid: Sequence[float], grid_shape: Sequence[int]
):
    _, n_cols = grid_shape
    x1_idx, x2_idx = divmod(flat_index, n_cols)
    return [float(x_grid[x1_idx]), float(x_grid[x2_idx])]


def _solve_optimal_split_factors(
    sigma_estimates: Sequence[Dict[str, Any]],
    tau_estimate: Dict[str, Any],
    x_grid: Sequence[float],
):
    t_matrix_start = time.perf_counter()
    sigma2_matrix, grid_shape = _build_sigma2_matrix(sigma_estimates, tau_estimate)
    matrix_build_time = time.perf_counter() - t_matrix_start
    t_opt_start = time.perf_counter()
    num_points, num_levels = sigma2_matrix.shape
    weight_floor = 1e-12

    if num_levels < 2:
        sigma2_max = float(np.max(sigma2_matrix))
        j_star = int(np.argmax(sigma2_matrix.sum(axis=1)))
        dual_weights = np.zeros(num_points, dtype=float)
        dual_weights[j_star] = 1.0
        return {
            "split_factors": [],
            "allocation_weights": [1.0],
            "dual_weights": dual_weights.tolist(),
            "dual_status": 0,
            "dual_success": True,
            "dual_message": "Trivial single-level allocation",
            "dual_objective": sigma2_max,
            "worst_case_simplex_objective": sigma2_max,
            "active_grid_points": [
                {
                    "point": _grid_index_to_point(j_star, x_grid, grid_shape),
                    "weight": 1.0,
                }
            ],
            "matrix_build_time": matrix_build_time,
            "optimize_time": time.perf_counter() - t_opt_start,
        }

    # Fast path: single worst-case grid point t* = argmax_j sum_k sigma_k(j)
    # gives M_k ∝ sigma_k(t*) in closed form. KKT check confirms whether this
    # is globally optimal; if not, fall back to SLSQP on the dual simplex.
    sigma_matrix = np.sqrt(np.maximum(sigma2_matrix, 0.0))
    row_sums = sigma_matrix.sum(axis=1)
    j_star = int(np.argmax(row_sums))
    sigmas_at_j = sigma_matrix[j_star]
    total = float(sigmas_at_j.sum())

    if total <= 0.0:
        uniform_M = np.full(num_levels, 1.0 / num_levels, dtype=float)
        dual_weights = np.full(num_points, 1.0 / num_points, dtype=float)
        return {
            "split_factors": (uniform_M[1:] / uniform_M[:-1]).tolist(),
            "allocation_weights": uniform_M.tolist(),
            "dual_weights": dual_weights.tolist(),
            "dual_status": 0,
            "dual_success": True,
            "dual_message": "All sigma estimates are zero; using uniform allocation",
            "dual_objective": 0.0,
            "worst_case_simplex_objective": 0.0,
            "active_grid_points": [
                {
                    "point": _grid_index_to_point(idx, x_grid, grid_shape),
                    "weight": float(weight),
                }
                for idx, weight in enumerate(dual_weights)
                if weight > 1e-9
            ],
            "matrix_build_time": matrix_build_time,
            "optimize_time": time.perf_counter() - t_opt_start,
        }

    M_candidate = sigmas_at_j / total
    kkt_tol = 1e-8
    zero_levels = M_candidate <= 0.0
    kkt_feasible = not (
        np.any(zero_levels) and np.any(sigma2_matrix[:, zero_levels] > 0.0)
    )
    if kkt_feasible:
        safe_M = np.where(M_candidate > 0.0, M_candidate, 1.0)
        per_level = np.where(
            M_candidate[None, :] > 0.0, sigma2_matrix / safe_M[None, :], 0.0
        )
        worst_case_values = per_level.sum(axis=1)
        best = float(worst_case_values[j_star])
        kkt_passes = best > 0.0 and float(worst_case_values.max()) <= best * (
            1.0 + kkt_tol
        )
    else:
        kkt_passes = False

    if kkt_passes:
        M_out = np.maximum(M_candidate, weight_floor)
        M_out /= M_out.sum()
        split_factors = M_out[1:] / M_out[:-1]
        dual_weights = np.zeros(num_points, dtype=float)
        dual_weights[j_star] = 1.0
        worst_case = float(np.sum(sigma2_matrix[j_star] / M_out))
        return {
            "split_factors": split_factors.tolist(),
            "allocation_weights": M_out.tolist(),
            "dual_weights": dual_weights.tolist(),
            "dual_status": 0,
            "dual_success": True,
            "dual_message": "Single-atom closed form (KKT verified)",
            "dual_objective": worst_case,
            "worst_case_simplex_objective": worst_case,
            "active_grid_points": [
                {
                    "point": _grid_index_to_point(j_star, x_grid, grid_shape),
                    "weight": 1.0,
                }
            ],
            "matrix_build_time": matrix_build_time,
            "optimize_time": time.perf_counter() - t_opt_start,
        }

    # KKT failed: the worst-case is a mixture. Fall back to SLSQP on the dual
    # problem max_mu (sum_k sqrt(mu^T sigma2[:, k]))^2 over the simplex,
    # warm-started from a single-atom point at j_star.
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "scipy is required to solve the simplex allocation problem"
        ) from exc

    c_floor = 1e-15
    simplex_tol = 1e-9

    def _phi_and_grad(mu: np.ndarray):
        c = sigma2_matrix.T @ mu
        c_safe = np.maximum(c, c_floor)
        sqrt_c = np.sqrt(c_safe)
        sum_sqrt_c = float(np.sum(sqrt_c))
        phi = sum_sqrt_c**2
        grad = sum_sqrt_c * np.sum(sigma2_matrix / sqrt_c[None, :], axis=1)
        return phi, grad, c

    def objective(mu: np.ndarray):
        phi, _, _ = _phi_and_grad(mu)
        return -phi

    def objective_grad(mu: np.ndarray):
        _, grad, _ = _phi_and_grad(mu)
        return -grad

    initial_mu = np.full(num_points, 0.1 / num_points, dtype=float)
    initial_mu[j_star] += 0.9
    constraints = [
        {
            "type": "eq",
            "fun": lambda mu: float(np.sum(mu) - 1.0),
            "jac": lambda mu: np.ones_like(mu),
        }
    ]
    bounds = [(0.0, 1.0) for _ in range(num_points)]

    solver_result = minimize(
        objective,
        initial_mu,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-7, "maxiter": 1000},
    )

    dual_weights = np.clip(
        np.asarray(solver_result.x, dtype=float), a_min=0.0, a_max=None
    )
    dual_weight_sum = float(dual_weights.sum())
    if dual_weight_sum <= 0.0:
        raise ValueError("Simplex allocation solver returned zero dual weights")
    dual_weights /= dual_weight_sum

    _, _, c_opt = _phi_and_grad(dual_weights)
    primal_weights = np.sqrt(np.maximum(c_opt, 0.0))
    if primal_weights.sum() <= 0.0:
        primal_weights = np.full(num_levels, 1.0 / num_levels, dtype=float)
    else:
        primal_weights /= primal_weights.sum()
    primal_weights = np.maximum(primal_weights, weight_floor)
    primal_weights /= primal_weights.sum()

    split_factors = primal_weights[1:] / primal_weights[:-1]
    worst_case_simplex_objective = float(
        np.max(np.sum(sigma2_matrix / primal_weights[None, :], axis=1))
    )
    if not np.isfinite(worst_case_simplex_objective):
        raise ValueError(
            "Simplex allocation solver produced a non-finite objective value"
        )

    active_grid_points = [
        {
            "point": _grid_index_to_point(idx, x_grid, grid_shape),
            "weight": float(weight),
        }
        for idx, weight in enumerate(dual_weights)
        if weight > simplex_tol
    ]

    return {
        "split_factors": split_factors.tolist(),
        "allocation_weights": primal_weights.tolist(),
        "dual_weights": dual_weights.tolist(),
        "dual_status": int(getattr(solver_result, "status", 0)),
        "dual_success": bool(getattr(solver_result, "success", False)),
        "dual_message": str(getattr(solver_result, "message", "")),
        "dual_objective": float(-solver_result.fun),
        "worst_case_simplex_objective": worst_case_simplex_objective,
        "active_grid_points": active_grid_points,
        "matrix_build_time": matrix_build_time,
        "optimize_time": time.perf_counter() - t_opt_start,
    }


def _expected_cost_per_root(
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


def _probabilistic_split(x: torch.Tensor, split_factor: float):
    if x.shape[0] == 0:
        return x
    if split_factor < 0:
        raise ValueError(f"split_factor must be nonnegative, got {split_factor}")
    if abs(split_factor - 1.0) < 1e-12:
        return x

    if split_factor > 1.0:
        base_copies = math.floor(split_factor)
        extra_prob = split_factor - base_copies
        pieces = []

        if base_copies > 0:
            pieces.append(x.repeat_interleave(base_copies, dim=0))

        if extra_prob > 0:
            extra_mask = torch.rand(x.shape[0], device=x.device) < extra_prob
            if extra_mask.any():
                pieces.append(x[extra_mask])

        if pieces:
            return torch.cat(pieces, dim=0)

        return x.new_empty((0, x.shape[1]))

    keep_mask = torch.rand(x.shape[0], device=x.device) < split_factor
    return x[keep_mask]


def run_probabilistic_inference(
    model,
    ddim: DDIM,
    n0: int,
    split_points: Sequence[int],
    split_factors: Sequence[float],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
):
    input_dim = int(torch.as_tensor(data_mean).numel())
    x = torch.randn(n0, input_dim, device=ddim.device)

    realized_cost = x.shape[0] * ddim.segment_cost(ddim.T, split_points[0])
    x = _sample_loop(ddim, model, x, ddim.T, split_points[0])

    for idx, split_point in enumerate(split_points):
        x = _probabilistic_split(x, split_factors[idx])

        if idx + 1 < len(split_points):
            end_t = split_points[idx + 1]
        else:
            end_t = 0

        realized_cost += x.shape[0] * ddim.segment_cost(split_point, end_t)
        x = _sample_loop(ddim, model, x, split_point, end_t)

    x = denormalize(x, data_mean, data_std)
    return x, int(realized_cost)


def _summarize_results(ks_distances: np.ndarray):
    valid_mask = ~np.isnan(ks_distances)
    valid_ks = ks_distances[valid_mask]
    if valid_ks.size == 0:
        return {
            "mean_ks": float("nan"),
            "std_ks": float("nan"),
            "var_ks": float("nan"),
            "n_valid_runs": 0,
        }

    return {
        "mean_ks": float(np.mean(valid_ks)),
        "std_ks": float(np.std(valid_ks)),
        "var_ks": float(np.var(valid_ks)),
        "n_valid_runs": int(valid_ks.size),
    }


def _run_single_sampling_trial(
    model,
    target_spec: Dict[str, Any],
    ddim: DDIM,
    n0: int,
    split_points: Sequence[int],
    split_factors: Sequence[float],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    phase1_x0_samples: torch.Tensor | None = None,
):
    sampling_start = time.perf_counter()
    samples, realized_cost = run_probabilistic_inference(
        model=model,
        ddim=ddim,
        n0=n0,
        split_points=split_points,
        split_factors=split_factors,
        data_mean=data_mean,
        data_std=data_std,
    )
    phase2_sampling_time = time.perf_counter() - sampling_start
    leaf_count = int(samples.shape[0])
    ks_samples = samples
    if phase1_x0_samples is not None and phase1_x0_samples.shape[0] > 0:
        ks_samples = torch.cat([phase1_x0_samples, samples], dim=0)

    ks_start = time.perf_counter()
    ks_distance = float("nan")
    if ks_samples.shape[0] > 0:
        ks_distance, _, _ = _compute_ks_distance(
            ks_samples, target_spec, reference_mode, reference_cdf_state
        )
    phase2_ks_time = time.perf_counter() - ks_start

    return {
        "ks_distance": float(ks_distance),
        "realized_cost": int(realized_cost),
        "leaf_count": leaf_count,
        "phase2_sampling_time": float(phase2_sampling_time),
        "phase2_ks_time": float(phase2_ks_time),
        "samples": samples.cpu().numpy(),
    }


def _summarize_sampling_trials(trial_results: Sequence[Dict[str, Any]]):
    ks_distances = np.array(
        [trial["ks_distance"] for trial in trial_results], dtype=float
    )
    realized_costs = np.array(
        [trial["realized_cost"] for trial in trial_results], dtype=float
    )
    leaf_counts = np.array(
        [trial["leaf_count"] for trial in trial_results], dtype=float
    )
    phase2_sampling_times = np.array(
        [trial["phase2_sampling_time"] for trial in trial_results], dtype=float
    )
    phase2_ks_times = np.array(
        [trial["phase2_ks_time"] for trial in trial_results], dtype=float
    )
    stats = _summarize_results(ks_distances)
    extinction_rate = float(np.mean(leaf_counts == 0.0)) if leaf_counts.size else 0.0

    return {
        "mean_ks": stats["mean_ks"],
        "std_ks": stats["std_ks"],
        "var_ks": stats["var_ks"],
        "n_valid_runs": stats["n_valid_runs"],
        "ks_distances": ks_distances.tolist(),
        "realized_costs": realized_costs.tolist(),
        "realized_cost_mean": (
            float(realized_costs.mean()) if realized_costs.size else 0.0
        ),
        "realized_cost_std": (
            float(realized_costs.std()) if realized_costs.size else 0.0
        ),
        "leaf_counts": leaf_counts.astype(int).tolist(),
        "leaf_count_mean": float(leaf_counts.mean()) if leaf_counts.size else 0.0,
        "leaf_count_std": float(leaf_counts.std()) if leaf_counts.size else 0.0,
        "extinction_rate": float(extinction_rate),
        "phase2_sampling_times": phase2_sampling_times.tolist(),
        "phase2_sampling_time_mean": (
            float(phase2_sampling_times.mean()) if phase2_sampling_times.size else 0.0
        ),
        "phase2_sampling_time_std": (
            float(phase2_sampling_times.std()) if phase2_sampling_times.size else 0.0
        ),
        "phase2_ks_times": phase2_ks_times.tolist(),
        "phase2_ks_time_mean": (
            float(phase2_ks_times.mean()) if phase2_ks_times.size else 0.0
        ),
        "phase2_ks_time_std": (
            float(phase2_ks_times.std()) if phase2_ks_times.size else 0.0
        ),
        "samples": [trial["samples"] for trial in trial_results],
    }


def _sample_dpmpp_2m(model, x: torch.Tensor, T: int, sampling_steps: int):
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

    with torch.inference_mode():
        for t in scheduler.timesteps:
            t_value = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            t_tensor = torch.full(
                (x.shape[0],), t_value, device=x.device, dtype=torch.long
            )
            eps_pred = model(x, t_tensor)
            x = scheduler.step(eps_pred, t, x).prev_sample
    return x


def _sample_full_solver(
    model,
    solver: str,
    x: torch.Tensor,
    *,
    T: int,
    sampling_steps: int,
    eta: float,
    device: str,
):
    if solver == "ddim":
        ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
        return ddim.sample_loop(model, x, ddim.T, 0)
    if solver == "dpmpp_2m":
        return _sample_dpmpp_2m(model, x, T=T, sampling_steps=sampling_steps)
    raise ValueError(f"Unknown solver baseline '{solver}'")


def _run_single_solver_baseline_trial(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    solver: str,
    B: int,
    T: int,
    sampling_steps: int,
    eta: float,
    reference_mode: str,
    device: str,
):
    input_dim = _model_input_dim(model)
    n0 = int(B // sampling_steps)
    if n0 < 1:
        raise ValueError(
            f"Budget B={B} is too small for solver baseline with {sampling_steps} steps"
        )

    sampling_start = time.perf_counter()
    x = torch.randn(n0, input_dim, device=device)
    samples = _sample_full_solver(
        model,
        solver,
        x,
        T=T,
        sampling_steps=sampling_steps,
        eta=eta,
        device=device,
    )
    samples = denormalize(samples, data_mean, data_std)
    _sync_device(device)
    phase2_sampling_time = time.perf_counter() - sampling_start

    ks_start = time.perf_counter()
    ks_distance, _, _ = _compute_ks_distance(
        samples, target_spec, reference_mode, reference_cdf_state
    )
    phase2_ks_time = time.perf_counter() - ks_start

    return {
        "ks_distance": float(ks_distance),
        "realized_cost": int(n0 * sampling_steps),
        "leaf_count": int(samples.shape[0]),
        "phase2_sampling_time": float(phase2_sampling_time),
        "phase2_ks_time": float(phase2_ks_time),
        "samples": samples.cpu().numpy(),
    }


def run_solver_baseline_sampling(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    solver: str,
    B: int,
    T: int,
    sampling_steps: int,
    eta: float = 0.0,
    n_runs: int,
    seed: int | None,
    device: str,
    reference_mode: str = "samples",
    debug: bool = False,
):
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be at least 1")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n0 = int(B // sampling_steps)
    if n0 < 1:
        raise ValueError(
            f"Budget B={B} is too small for solver baseline with {sampling_steps} steps"
        )

    _debug_log(
        debug,
        "Starting solver baseline "
        f"with solver={solver}, B={B}, sampling_steps={sampling_steps}, eta={eta}",
    )
    _debug_log(
        debug,
        "Solver baseline sampling setup: "
        f"n0={n0}, expected_cost_per_root={sampling_steps:.6f}, "
        f"expected_total_samples={float(n0):.6f}",
    )

    trial_results = []
    for run_idx in tqdm(range(n_runs), desc="Runs", leave=False):
        if seed is not None:
            run_seed = seed + run_idx
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
        trial_results.append(
            _run_single_solver_baseline_trial(
                model=model,
                target_spec=target_spec,
                data_mean=data_mean,
                data_std=data_std,
                reference_cdf_state=reference_cdf_state,
                solver=solver,
                B=B,
                T=T,
                sampling_steps=sampling_steps,
                eta=eta,
                reference_mode=reference_mode,
                device=device,
            )
        )

    if not trial_results:
        raise ValueError("n_runs must be at least 1")

    stats = _summarize_sampling_trials(trial_results)
    _debug_log(
        debug,
        f"Solver baseline sampling complete: mean_ks={stats['mean_ks']:.6f}, "
        f"extinction_rate={stats['extinction_rate']:.6f}",
    )
    expected_total_samples = float(n0)
    return {
        "mode": "solver_baseline",
        "solver": solver,
        "B": int(B),
        "B1": 0,
        "used_B1": 0,
        "B2": int(B),
        "sampling_steps": int(sampling_steps),
        "eta": float(eta),
        "reference_mode": reference_mode,
        "solver_nfe": int(sampling_steps),
        "N_i": [],
        "n0": int(n0),
        "final_root_count": int(n0),
        "expected_cost_per_root": float(sampling_steps),
        "expected_total_samples": expected_total_samples,
        "target_spec": target_spec,
        **stats,
    }


def _mean_scalar(values: Sequence[float | int]):
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _mean_vector(values: Sequence[Sequence[float | int]]):
    if not values:
        return []
    return np.mean(np.asarray(values, dtype=float), axis=0).tolist()


def _std_vector(values: Sequence[Sequence[float | int]]):
    if not values:
        return []
    return np.std(np.asarray(values, dtype=float), axis=0).tolist()


def _mean_grid(values: Sequence[Sequence[Sequence[float]]]):
    if not values:
        return []
    return np.mean(np.asarray(values, dtype=float), axis=0).tolist()


def _average_sigma_estimates(
    sigma_estimates_by_run: Sequence[Sequence[Dict[str, Any]]], x_grid: Sequence[float]
):
    if not sigma_estimates_by_run:
        return []

    averaged_estimates = []
    for level_idx in range(len(sigma_estimates_by_run[0])):
        sigma2_grid = _mean_grid(
            [run[level_idx]["sigma2_grid"] for run in sigma_estimates_by_run]
        )
        sigma_grid = _mean_grid(
            [run[level_idx]["sigma_grid"] for run in sigma_estimates_by_run]
        )
        p_hat_mean_grid = _mean_grid(
            [run[level_idx]["p_hat_mean_grid"] for run in sigma_estimates_by_run]
        )
        sigma2_array = np.asarray(sigma2_grid, dtype=float)
        max_index = np.unravel_index(np.argmax(sigma2_array), sigma2_array.shape)
        averaged_estimates.append(
            {
                "time": int(sigma_estimates_by_run[0][level_idx]["time"]),
                "x_grid": list(x_grid),
                "sigma2_grid": sigma2_grid,
                "sigma_grid": sigma_grid,
                "p_hat_mean_grid": p_hat_mean_grid,
                "sigma2_max": float(sigma2_array[max_index]),
                "sigma": float(math.sqrt(max(float(sigma2_array[max_index]), 0.0))),
                "max_location": [
                    float(x_grid[max_index[0]]),
                    float(x_grid[max_index[1]]),
                ],
            }
        )

    return averaged_estimates


def _average_tau_estimate(
    tau_estimates: Sequence[Dict[str, Any]], x_grid: Sequence[float]
):
    if not tau_estimates:
        return {}

    tau2_grid = _mean_grid(
        [tau_estimate["tau2_grid"] for tau_estimate in tau_estimates]
    )
    tau_grid = _mean_grid([tau_estimate["tau_grid"] for tau_estimate in tau_estimates])
    p_hat_mean_grid = _mean_grid(
        [tau_estimate["p_hat_mean_grid"] for tau_estimate in tau_estimates]
    )
    tau2_array = np.asarray(tau2_grid, dtype=float)
    max_index = np.unravel_index(np.argmax(tau2_array), tau2_array.shape)
    return {
        "x_grid": list(x_grid),
        "tau2_grid": tau2_grid,
        "tau_grid": tau_grid,
        "p_hat_mean_grid": p_hat_mean_grid,
        "tau2_max": float(tau2_array[max_index]),
        "tau": float(math.sqrt(max(float(tau2_array[max_index]), 0.0))),
        "max_location": [
            float(x_grid[max_index[0]]),
            float(x_grid[max_index[1]]),
        ],
    }


def _summarize_allocation_messages(messages: Sequence[str]):
    unique_messages = list(dict.fromkeys(messages))
    if not unique_messages:
        return ""
    if len(unique_messages) == 1:
        return unique_messages[0]
    return "Multiple allocation solver messages across runs"


def _active_grid_points_from_dual_weights(
    dual_weights: Sequence[float], x_grid: Sequence[float], grid_shape: Sequence[int]
):
    simplex_tol = 1e-9
    return [
        {
            "point": _grid_index_to_point(point_idx, x_grid, grid_shape),
            "weight": float(weight),
        }
        for point_idx, weight in enumerate(dual_weights)
        if weight > simplex_tol
    ]


def _run_sampling_trials(
    model,
    target_spec: Dict[str, Any],
    ddim: DDIM,
    n_runs: int,
    n0: int,
    split_points: Sequence[int],
    split_factors: Sequence[float],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    reference_mode: str,
    debug: bool = False,
):
    ks_distances = []
    realized_costs = []
    leaf_counts = []
    extinction_count = 0
    samples_by_run = []

    _debug_log(
        debug,
        f"Sampling phase: running {n_runs} trials with n0={n0} and split factors {list(split_factors)}",
    )

    for _ in tqdm(range(n_runs), desc="Runs", leave=False):
        trial_result = _run_single_sampling_trial(
            model=model,
            target_spec=target_spec,
            ddim=ddim,
            n0=n0,
            split_points=split_points,
            split_factors=split_factors,
            data_mean=data_mean,
            data_std=data_std,
            reference_cdf_state=reference_cdf_state,
            reference_mode=reference_mode,
        )
        realized_costs.append(trial_result["realized_cost"])
        leaf_counts.append(trial_result["leaf_count"])
        samples_by_run.append(trial_result["samples"])

        if trial_result["leaf_count"] == 0:
            extinction_count += 1
            ks_distances.append(float("nan"))
            continue

        ks_distances.append(float(trial_result["ks_distance"]))

    ks_distances = np.array(ks_distances, dtype=float)
    realized_costs = np.array(realized_costs, dtype=float)
    leaf_counts = np.array(leaf_counts, dtype=float)
    stats = _summarize_results(ks_distances)
    extinction_rate = extinction_count / max(n_runs, 1)

    return {
        "mean_ks": stats["mean_ks"],
        "std_ks": stats["std_ks"],
        "var_ks": stats["var_ks"],
        "n_valid_runs": stats["n_valid_runs"],
        "ks_distances": ks_distances.tolist(),
        "realized_costs": realized_costs.tolist(),
        "realized_cost_mean": (
            float(realized_costs.mean()) if realized_costs.size else 0.0
        ),
        "realized_cost_std": (
            float(realized_costs.std()) if realized_costs.size else 0.0
        ),
        "leaf_counts": leaf_counts.astype(int).tolist(),
        "leaf_count_mean": float(leaf_counts.mean()) if leaf_counts.size else 0.0,
        "leaf_count_std": float(leaf_counts.std()) if leaf_counts.size else 0.0,
        "extinction_rate": float(extinction_rate),
        "samples": samples_by_run,
    }


def _run_single_estimate_and_sample_trial(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    B: int,
    B1: int,
    T: int,
    sampling_steps: int,
    eta: float,
    split_percentages: Sequence[float],
    x_grid: Sequence[float],
    independent_n2: int,
    sigma_estimation_mode: str,
    reuse_phase1_samples: bool,
    seed: int | None,
    device: str,
    reference_mode: str,
    debug: bool = False,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
    resolved_step_points, split_points = _resolve_split_percentages(
        ddim, split_percentages
    )
    sigma_times = [T] + list(split_points)

    derived_m_pilot = None
    phase1_metadata: Dict[str, Any]

    if sigma_estimation_mode == "pilot_tree":
        pilot_scale, pilot_roots, derived_m_pilot = _derive_pilot_tree_shape(
            ddim=ddim,
            split_points=split_points,
            B1=B1,
        )

        _debug_log(
            debug,
            "Phase 1 (pilot tree): "
            f"pilot_scale={pilot_scale:.6f}, pilot_levels={len(sigma_times)}, "
            f"pilot_roots={pilot_roots}, derived_m_pilot={derived_m_pilot:.6f}",
        )
        diffusion_start = time.perf_counter()
        (
            _,
            phase1_x0_samples,
            _,
            _,
            child_counts_by_level,
            used_B1,
        ) = _build_pilot_tree(
            model=model,
            ddim=ddim,
            pilot_roots=pilot_roots,
            split_points=split_points,
            m_pilot=derived_m_pilot,
        )
        phase1_x0_samples = denormalize(phase1_x0_samples, data_mean, data_std)
        _sync_device(ddim.device)
        diffusion_time = time.perf_counter() - diffusion_start

        _debug_log(debug, "Estimating sigma_k grids from pilot tree leaves")
        estimation_start = time.perf_counter()
        sigma_estimates, tau_estimate = _estimate_sigmas_from_tree(
            leaf_x0=phase1_x0_samples,
            sigma_times=sigma_times,
            child_counts_by_level=child_counts_by_level,
            x_grid=x_grid,
        )
        _sync_device(ddim.device)
        estimation_time = time.perf_counter() - estimation_start
        phase1_metadata = {
            "pilot_roots": int(pilot_roots),
            "m_pilot": float(derived_m_pilot),
        }
    else:
        _, independent_counts_by_sigma, _ = _derive_independent_counts(
            ddim=ddim,
            sigma_times=sigma_times,
            B1=B1,
            independent_n2=independent_n2,
        )

        _debug_log(
            debug,
            "Phase 1 (independent): " f"counts_by_sigma={independent_counts_by_sigma}",
        )
        _debug_log(debug, "Estimating sigma_k grids independently")
        (
            phase1_x0_samples,
            sigma_estimates,
            tau_estimate,
            used_B1,
            diffusion_time,
            estimation_time,
        ) = _estimate_sigmas_independently(
            model=model,
            ddim=ddim,
            sigma_times=sigma_times,
            counts_by_sigma=independent_counts_by_sigma,
            x_grid=x_grid,
            data_mean=data_mean,
            data_std=data_std,
            collect_phase1_samples=reuse_phase1_samples,
        )
        phase1_metadata = {
            "pilot_roots": int(independent_counts_by_sigma[0]["outer_count"]),
            "m_pilot": None,
        }
    B2 = B - used_B1
    _debug_log(debug, f"Phase 1 complete: used_B1={used_B1}, remaining B2={B2}")
    _debug_log(debug, "Estimating optimal normalized M_k on the simplex")
    allocation_result = _solve_optimal_split_factors(
        sigma_estimates=sigma_estimates,
        tau_estimate=tau_estimate,
        x_grid=x_grid,
    )
    split_factors = allocation_result["split_factors"]
    _debug_log(
        debug,
        "Found optimal normalized M_k="
        f"{allocation_result['allocation_weights']} and derived N_i={split_factors}",
    )
    expected_cost_per_root = _expected_cost_per_root(ddim, split_points, split_factors)
    n0 = int(B2 // expected_cost_per_root)
    if n0 < 1:
        raise ValueError(
            f"Remaining budget B2={B2} is too small; expected cost per root is {expected_cost_per_root:.6f}"
        )

    phase1_samples_for_ks = phase1_x0_samples if reuse_phase1_samples else None

    expected_total_samples = float(n0)
    for split_factor in split_factors:
        expected_total_samples *= split_factor

    _debug_log(
        debug,
        "Phase 2 sampling setup: "
        f"expected_cost_per_root={expected_cost_per_root:.6f}, n0={n0}, expected_total_samples={expected_total_samples:.6f}",
    )
    sampling_result = _run_single_sampling_trial(
        model=model,
        target_spec=target_spec,
        ddim=ddim,
        n0=n0,
        split_points=split_points,
        split_factors=split_factors,
        data_mean=data_mean,
        data_std=data_std,
        reference_cdf_state=reference_cdf_state,
        reference_mode=reference_mode,
        phase1_x0_samples=phase1_samples_for_ks,
    )

    print(
        f"[timings] diffusion={diffusion_time:.3f}s "
        f"estimate_sigmas={estimation_time:.3f}s "
        f"sigma_matrix={allocation_result['matrix_build_time']:.4f}s "
        f"optimize_N_i={allocation_result['optimize_time']:.4f}s "
        f"phase2_sampling={sampling_result['phase2_sampling_time']:.3f}s "
        f"phase2_ks={sampling_result['phase2_ks_time']:.3f}s"
    )

    return {
        "resolved_step_points": [int(x) for x in resolved_step_points],
        "split_points": [int(x) for x in split_points],
        "sigma_times": [int(x) for x in sigma_times],
        "sigma_estimates": sigma_estimates,
        "tau_estimate": tau_estimate,
        "optimal_M_k": [float(x) for x in allocation_result["allocation_weights"]],
        "allocation_dual_weights": [
            float(x) for x in allocation_result["dual_weights"]
        ],
        "allocation_solver_success": bool(allocation_result["dual_success"]),
        "allocation_solver_status": int(allocation_result["dual_status"]),
        "allocation_solver_message": allocation_result["dual_message"],
        "allocation_dual_objective": float(allocation_result["dual_objective"]),
        "allocation_worst_case_simplex_objective": float(
            allocation_result["worst_case_simplex_objective"]
        ),
        "N_i": [float(x) for x in split_factors],
        "n0": int(n0),
        "final_root_count": int(n0),
        "expected_cost_per_root": float(expected_cost_per_root),
        "expected_total_samples": float(expected_total_samples),
        "used_B1": int(used_B1),
        "B2": int(B2),
        **phase1_metadata,
        **sampling_result,
    }


def run_estimate_and_sample(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    B: int,
    B1: int,
    T: int,
    sampling_steps: int,
    eta: float,
    split_percentages: Sequence[float],
    x_grid: Sequence[float],
    independent_n2: int,
    sigma_estimation_mode: str,
    reuse_phase1_samples: bool,
    n_runs: int,
    seed: int | None,
    device: str,
    reference_mode: str = "samples",
    debug: bool = False,
):
    if B1 < 0:
        raise ValueError("B1 must be nonnegative")
    if B1 >= B:
        raise ValueError("B1 must be strictly smaller than B")
    if sigma_estimation_mode == "independent" and independent_n2 < 2:
        raise ValueError("independent_n2 must be at least 2")

    _debug_log(
        debug,
        "Starting estimate_and_sample run "
        f"with B={B}, B1={B1}, sampling_steps={sampling_steps}, eta={eta}",
    )
    trial_results = []
    for run_idx in tqdm(range(n_runs), desc="Runs", leave=False):
        run_seed = None if seed is None else seed + run_idx
        trial_results.append(
            _run_single_estimate_and_sample_trial(
                model=model,
                target_spec=target_spec,
                data_mean=data_mean,
                data_std=data_std,
                reference_cdf_state=reference_cdf_state,
                B=B,
                B1=B1,
                T=T,
                sampling_steps=sampling_steps,
                eta=eta,
                split_percentages=split_percentages,
                x_grid=x_grid,
                independent_n2=independent_n2,
                sigma_estimation_mode=sigma_estimation_mode,
                reuse_phase1_samples=reuse_phase1_samples,
                seed=run_seed,
                device=device,
                reference_mode=reference_mode,
                debug=debug,
            )
        )

    if not trial_results:
        raise ValueError("n_runs must be at least 1")

    stats = _summarize_sampling_trials(trial_results)
    sigma_estimates = _average_sigma_estimates(
        [trial["sigma_estimates"] for trial in trial_results], x_grid
    )
    tau_estimate = _average_tau_estimate(
        [trial["tau_estimate"] for trial in trial_results], x_grid
    )
    allocation_dual_weights = _mean_vector(
        [trial["allocation_dual_weights"] for trial in trial_results]
    )
    grid_shape = np.asarray(
        trial_results[0]["tau_estimate"]["tau2_grid"], dtype=float
    ).shape
    _debug_log(
        debug,
        f"Sampling complete: mean_ks={stats['mean_ks']:.6f}, extinction_rate={stats['extinction_rate']:.6f}",
    )

    return {
        "mode": "estimate_and_sample",
        "B": int(B),
        "B1": int(B1),
        "sampling_steps": int(sampling_steps),
        "eta": float(eta),
        "reference_mode": reference_mode,
        "split_percentages": [float(x) for x in split_percentages],
        "split_step_points": trial_results[0]["resolved_step_points"],
        "split_points": trial_results[0]["split_points"],
        "x_grid": [float(x) for x in x_grid],
        "sigma_estimation_mode": sigma_estimation_mode,
        "independent_n2": int(independent_n2),
        "reuse_phase1_samples": bool(reuse_phase1_samples),
        "sigma_times": trial_results[0]["sigma_times"],
        "sigma_estimates": sigma_estimates,
        "tau_estimate": tau_estimate,
        "optimal_M_k": _mean_vector([trial["optimal_M_k"] for trial in trial_results]),
        "allocation_dual_weights": allocation_dual_weights,
        "allocation_active_grid_points": _active_grid_points_from_dual_weights(
            allocation_dual_weights,
            x_grid,
            grid_shape,
        ),
        "allocation_solver_success": all(
            trial["allocation_solver_success"] for trial in trial_results
        ),
        "allocation_solver_success_rate": _mean_scalar(
            [trial["allocation_solver_success"] for trial in trial_results]
        ),
        "allocation_solver_status": int(
            round(
                _mean_scalar(
                    [trial["allocation_solver_status"] for trial in trial_results]
                )
            )
        ),
        "allocation_solver_message": _summarize_allocation_messages(
            [trial["allocation_solver_message"] for trial in trial_results]
        ),
        "allocation_dual_objective": _mean_scalar(
            [trial["allocation_dual_objective"] for trial in trial_results]
        ),
        "allocation_worst_case_simplex_objective": _mean_scalar(
            [
                trial["allocation_worst_case_simplex_objective"]
                for trial in trial_results
            ]
        ),
        "N_i": _mean_vector([trial["N_i"] for trial in trial_results]),
        "N_i_std": _std_vector([trial["N_i"] for trial in trial_results]),
        "N_i_count": int(len(trial_results)),
        "n0": _mean_scalar([trial["n0"] for trial in trial_results]),
        "final_root_count": _mean_scalar(
            [trial["final_root_count"] for trial in trial_results]
        ),
        "expected_cost_per_root": _mean_scalar(
            [trial["expected_cost_per_root"] for trial in trial_results]
        ),
        "expected_total_samples": _mean_scalar(
            [trial["expected_total_samples"] for trial in trial_results]
        ),
        "target_spec": target_spec,
        "pilot_roots": _mean_scalar([trial["pilot_roots"] for trial in trial_results]),
        "m_pilot": (
            _mean_scalar(
                [
                    trial["m_pilot"]
                    for trial in trial_results
                    if trial["m_pilot"] is not None
                ]
            )
            if any(trial["m_pilot"] is not None for trial in trial_results)
            else None
        ),
        "used_B1": _mean_scalar([trial["used_B1"] for trial in trial_results]),
        "B2": _mean_scalar([trial["B2"] for trial in trial_results]),
        **stats,
    }


def run_fixed_N_sampling(
    model,
    target_spec: Dict[str, Any],
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    reference_cdf_state: Dict[str, Any] | None,
    *,
    B: int,
    T: int,
    sampling_steps: int,
    eta: float,
    split_percentages: Sequence[float],
    N_i_list: Sequence[float],
    n_runs: int,
    seed: int | None,
    device: str,
    reference_mode: str = "samples",
    debug: bool = False,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    _debug_log(
        debug,
        "Starting fixed_N run "
        f"with B={B}, sampling_steps={sampling_steps}, eta={eta}, N_i={list(N_i_list)}",
    )
    ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
    resolved_step_points, split_points = _resolve_split_percentages(
        ddim, split_percentages
    )

    split_factors = [float(x) for x in N_i_list]
    expected_cost_per_root = _expected_cost_per_root(ddim, split_points, split_factors)
    n0 = int(B // expected_cost_per_root)
    if n0 < 1:
        raise ValueError(
            f"Budget B={B} is too small; expected cost per root is {expected_cost_per_root:.6f}"
        )

    expected_total_samples = float(n0)
    for split_factor in split_factors:
        expected_total_samples *= split_factor

    _debug_log(
        debug,
        "Fixed_N sampling setup: "
        f"split_points={split_points}, expected_cost_per_root={expected_cost_per_root:.6f}, n0={n0}",
    )
    stats = _run_sampling_trials(
        model=model,
        target_spec=target_spec,
        ddim=ddim,
        n_runs=n_runs,
        n0=n0,
        split_points=split_points,
        split_factors=split_factors,
        data_mean=data_mean,
        data_std=data_std,
        reference_cdf_state=reference_cdf_state,
        reference_mode=reference_mode,
        debug=debug,
    )
    _debug_log(
        debug,
        f"Fixed_N sampling complete: mean_ks={stats['mean_ks']:.6f}, extinction_rate={stats['extinction_rate']:.6f}",
    )

    return {
        "mode": "fixed_N",
        "B": int(B),
        "B1": 0,
        "used_B1": 0,
        "B2": int(B),
        "sampling_steps": int(sampling_steps),
        "eta": float(eta),
        "reference_mode": reference_mode,
        "split_percentages": [float(x) for x in split_percentages],
        "split_step_points": [int(x) for x in resolved_step_points],
        "split_points": [int(x) for x in split_points],
        "input_N_i": [float(x) for x in split_factors],
        "N_i": [float(x) for x in split_factors],
        "n0": int(n0),
        "final_root_count": int(n0),
        "expected_cost_per_root": float(expected_cost_per_root),
        "expected_total_samples": float(expected_total_samples),
        "target_spec": target_spec,
        **stats,
    }
