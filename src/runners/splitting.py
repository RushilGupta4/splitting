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


def normalize_max_sampling_batch_size(value):
    if value in (None, "", "none", "None", "null", "Null"):
        return None
    value = int(value)
    if value < 1:
        raise ValueError("max_sampling_batch_size must be positive or null")
    return value


def _split_count_for_batch(total: int, batch_idx: int, num_batches: int) -> int:
    total = int(total)
    num_batches = int(num_batches)
    base = total // num_batches
    remainder = total - base * num_batches
    return int(base + (int(batch_idx) < remainder))


def split_counts_by_run_batches(counts_by_run, max_count_per_run):
    max_count_per_run = normalize_max_sampling_batch_size(max_count_per_run)
    counts = [int(count) for count in counts_by_run]
    if any(count < 0 for count in counts):
        raise ValueError("counts_by_run must be nonnegative")
    if not counts:
        return []
    if max_count_per_run is None:
        return [counts]
    max_count = max(counts)
    if max_count == 0:
        return [counts]
    num_batches = int(math.ceil(max_count / float(max_count_per_run)))
    return [
        [
            _split_count_for_batch(count, batch_idx, num_batches)
            for count in counts
        ]
        for batch_idx in range(num_batches)
    ]


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


def iter_run_id_batches(
    run_ids: torch.Tensor,
    num_runs: int,
    max_count_per_run,
):
    max_count_per_run = normalize_max_sampling_batch_size(max_count_per_run)
    run_ids = run_ids.to(dtype=torch.long)
    num_runs = int(num_runs)
    if run_ids.ndim != 1:
        raise ValueError("run_ids must be 1D")
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    if run_ids.numel() == 0:
        yield torch.empty((0,), device=run_ids.device, dtype=torch.long)
        return
    if torch.any(run_ids < 0) or torch.any(run_ids >= num_runs):
        raise ValueError("run_ids outside [0, num_runs)")

    all_indices = torch.arange(run_ids.numel(), device=run_ids.device, dtype=torch.long)
    if max_count_per_run is None:
        yield all_indices
        return

    counts = torch.bincount(run_ids, minlength=num_runs).detach().cpu().tolist()
    max_count = max(int(count) for count in counts)
    if max_count <= max_count_per_run:
        yield all_indices
        return

    num_batches = int(math.ceil(max_count / float(max_count_per_run)))
    positions_by_run = [
        torch.nonzero(run_ids == run_idx, as_tuple=False).flatten()
        for run_idx in range(num_runs)
    ]
    for batch_idx in range(num_batches):
        pieces = []
        for positions in positions_by_run:
            count = int(positions.numel())
            start = batch_idx * (count // num_batches) + min(
                batch_idx, count % num_batches
            )
            size = _split_count_for_batch(count, batch_idx, num_batches)
            if size > 0:
                pieces.append(positions[start : start + size])
        if pieces:
            yield torch.cat(pieces, dim=0)


def child_counts_by_run(
    run_ids: torch.Tensor,
    child_counts: torch.Tensor,
    *,
    num_runs: int,
):
    run_ids = run_ids.to(dtype=torch.long)
    child_counts = child_counts.to(device=run_ids.device, dtype=torch.long)
    if run_ids.ndim != 1 or child_counts.ndim != 1:
        raise ValueError("run_ids and child_counts must be 1D")
    if run_ids.numel() != child_counts.numel():
        raise ValueError("run_ids and child_counts must have the same length")
    num_runs = int(num_runs)
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    totals = torch.zeros(num_runs, device=run_ids.device, dtype=torch.long)
    if run_ids.numel() > 0:
        totals.index_add_(0, run_ids, child_counts)
    return totals


def iter_child_parent_batches_by_run(
    run_ids: torch.Tensor,
    child_counts: torch.Tensor,
    *,
    num_runs: int,
    max_count_per_run,
):
    max_count_per_run = normalize_max_sampling_batch_size(max_count_per_run)
    run_ids = run_ids.to(dtype=torch.long)
    child_counts = child_counts.to(device=run_ids.device, dtype=torch.long)
    num_runs = int(num_runs)
    terminal_counts = child_counts_by_run(
        run_ids,
        child_counts,
        num_runs=num_runs,
    )
    batch_counts_by_run = split_counts_by_run_batches(
        terminal_counts.detach().cpu().tolist(),
        max_count_per_run,
    )
    if not batch_counts_by_run:
        return

    parent_positions_by_run = []
    child_counts_by_parent_run = []
    for run_idx in range(num_runs):
        positions = torch.nonzero(run_ids == run_idx, as_tuple=False).flatten()
        parent_positions_by_run.append(positions.detach().cpu().tolist())
        child_counts_by_parent_run.append(
            child_counts.index_select(0, positions).detach().cpu().tolist()
        )

    offsets_by_run = [0] * num_runs
    for batch_counts in batch_counts_by_run:
        parent_indices = []
        for run_idx, take in enumerate(batch_counts):
            take = int(take)
            if take <= 0:
                continue
            start = offsets_by_run[run_idx]
            end = start + take
            offsets_by_run[run_idx] = end
            cursor = 0
            for parent_idx, count in zip(
                parent_positions_by_run[run_idx],
                child_counts_by_parent_run[run_idx],
            ):
                count = int(count)
                next_cursor = cursor + count
                if next_cursor <= start:
                    cursor = next_cursor
                    continue
                if cursor >= end:
                    break
                overlap_start = max(start, cursor)
                overlap_end = min(end, next_cursor)
                repeats = overlap_end - overlap_start
                if repeats > 0:
                    parent_indices.extend([int(parent_idx)] * int(repeats))
                cursor = next_cursor
        yield (
            torch.as_tensor(parent_indices, device=run_ids.device, dtype=torch.long),
            batch_counts,
        )


def apply_to_run_batches(
    x: torch.Tensor,
    run_ids: torch.Tensor,
    *,
    num_runs: int,
    max_count_per_run,
    fn,
):
    max_count_per_run = normalize_max_sampling_batch_size(max_count_per_run)
    if x.shape[0] == 0 or max_count_per_run is None:
        return fn(x)
    out = torch.empty_like(x)
    for batch_indices in iter_run_id_batches(
        run_ids,
        int(num_runs),
        max_count_per_run,
    ):
        if batch_indices.numel() == 0:
            continue
        out.index_copy_(0, batch_indices, fn(x.index_select(0, batch_indices)))
    return out


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
