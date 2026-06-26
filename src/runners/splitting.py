from __future__ import annotations

import math

import torch


def sample_branch_counts(
    num_parents: int,
    branching_factor: float,
    device: str | torch.device,
    generator: torch.Generator | None = None,
):
    return balanced_branch_counts(
        num_parents,
        branching_factor,
        device,
        generator=generator,
    )


def floor_split_total_count(num_parents: int, branching_factor: float) -> int:
    num_parents = int(num_parents)
    branching_factor = float(branching_factor)
    if num_parents < 0:
        raise ValueError("num_parents must be nonnegative")
    if not math.isfinite(branching_factor) or branching_factor < 0.0:
        raise ValueError("branching_factor must be finite and nonnegative")
    if num_parents == 0 or branching_factor == 0.0:
        return 0

    target = branching_factor * num_parents
    nearest = round(target)
    tol = 1e-12 * max(1.0, abs(target))
    if abs(target - nearest) <= tol:
        return int(nearest)
    return int(math.floor(target))


def balanced_branch_counts(
    num_parents: int,
    branching_factor: float,
    device: str | torch.device,
    generator: torch.Generator | None = None,
):
    num_parents = int(num_parents)
    branching_factor = float(branching_factor)
    if num_parents < 0:
        raise ValueError("num_parents must be nonnegative")
    if not math.isfinite(branching_factor) or branching_factor < 0.0:
        raise ValueError("branching_factor must be finite and nonnegative")
    if num_parents == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    base_copies = math.floor(branching_factor)
    target_total = floor_split_total_count(num_parents, branching_factor)
    extra_count = target_total - base_copies * num_parents
    extra_count = min(max(int(extra_count), 0), num_parents)

    counts = torch.full((num_parents,), base_copies, device=device, dtype=torch.long)
    if extra_count > 0:
        extra_indices = torch.randperm(
            num_parents,
            device=device,
            generator=generator,
        )[:extra_count]
        counts[extra_indices] += 1
    return counts


def balanced_branch_counts_by_group(
    group_ids: torch.Tensor,
    branching_factors_by_group: torch.Tensor,
    *,
    num_groups: int | None = None,
    generator: torch.Generator | None = None,
):
    group_ids = group_ids.to(dtype=torch.long)
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be 1D")
    if group_ids.numel() == 0:
        return torch.empty((0,), device=group_ids.device, dtype=torch.long)

    if num_groups is None:
        num_groups = int(group_ids.max().item()) + 1
    else:
        num_groups = int(num_groups)
    if num_groups < 1:
        raise ValueError("num_groups must be positive")
    if torch.any(group_ids < 0) or torch.any(group_ids >= num_groups):
        raise ValueError("group_ids outside [0, num_groups)")

    factors = torch.as_tensor(
        branching_factors_by_group,
        device=group_ids.device,
        dtype=torch.float64,
    ).reshape(-1)
    if factors.numel() != num_groups:
        raise ValueError("branching_factors_by_group must have length num_groups")
    if not torch.isfinite(factors).all() or torch.any(factors < 0.0):
        raise ValueError("branching factors must be finite and nonnegative")

    counts = torch.empty_like(group_ids, dtype=torch.long)
    for group_idx in range(num_groups):
        positions = torch.nonzero(group_ids == group_idx, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        group_counts = balanced_branch_counts(
            int(positions.numel()),
            float(factors[group_idx].item()),
            group_ids.device,
            generator=generator,
        )
        counts[positions] = group_counts
    return counts


def repeat_by_counts(x: torch.Tensor, counts: torch.Tensor):
    return x.repeat_interleave(counts, dim=0)


def balanced_split_with_run_ids(
    x: torch.Tensor,
    run_ids: torch.Tensor,
    split_factors_by_run: torch.Tensor,
    generator: torch.Generator | None = None,
):
    if x.shape[0] == 0:
        return x, run_ids

    counts = balanced_branch_counts_by_group(
        run_ids,
        split_factors_by_run,
        num_groups=int(split_factors_by_run.numel()),
        generator=generator,
    )

    return x.repeat_interleave(counts, dim=0), run_ids.repeat_interleave(counts, dim=0)


probabilistic_split_with_run_ids = balanced_split_with_run_ids
