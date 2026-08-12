from __future__ import annotations

import math
import time

import numpy as np
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


def append_by_counts(parts_by_run, values: torch.Tensor, counts_by_run):
    """Append contiguous slices of ``values`` to their corresponding runs."""
    offset = 0
    for run_idx, count in enumerate(counts_by_run):
        count = int(count)
        if count > 0:
            parts_by_run[run_idx].append(values[offset : offset + count])
        offset += count


def cat_parts_by_run(parts_by_run, empty_template: torch.Tensor):
    """Concatenate accumulated run parts while preserving empty tensor metadata."""
    return [
        torch.cat(parts, dim=0) if parts else empty_template[:0]
        for parts in parts_by_run
    ]


def collect_run_batches(
    counts_by_run,
    max_count_per_run,
    *,
    sample_batch,
    empty_template: torch.Tensor,
):
    """Sample bounded batches and reconstruct per-run tensors in stable order."""
    counts_by_run = [int(count) for count in counts_by_run]
    parts_by_run = [[] for _ in counts_by_run]
    latest_template = empty_template
    for counts in split_counts_by_run_batches(counts_by_run, max_count_per_run):
        total = int(sum(counts))
        if total <= 0:
            continue
        values = sample_batch(total)
        latest_template = values
        append_by_counts(parts_by_run, values, counts)
    return cat_parts_by_run(parts_by_run, latest_template)


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


def trajectory_segment_costs(runner, split_points):
    """Return segment costs for canonical runner-native split points."""
    points = list(split_points)
    starts = [runner.start_time, *points]
    ends = [*points, runner.end_time]
    return [runner.segment_cost(start, end) for start, end in zip(starts, ends)]


def trajectory_expected_cost_per_root(runner, split_points, split_factors) -> float:
    """Compute expected trajectory cost from cumulative split factors."""
    costs = trajectory_segment_costs(runner, split_points)
    if len(costs) == 1:
        return float(costs[0])
    cumulative_split = 1.0
    cost = float(costs[0])
    for idx, split_factor in enumerate(split_factors):
        cumulative_split *= float(split_factor)
        cost += cumulative_split * float(costs[idx + 1])
    return float(cost)


def floor_split_sampling_cost(runner, split_points, split_factors, n0: int) -> float:
    """Return realized cost after flooring the path count at every split."""
    n0 = int(n0)
    if n0 < 1:
        raise ValueError("n0 must be at least 1")

    points = list(split_points)
    factors = [float(value) for value in split_factors]
    if len(factors) != len(points):
        raise ValueError("split_factors must match split_points length")

    costs = trajectory_segment_costs(runner, points)
    current_count = n0
    total_cost = current_count * int(costs[0])
    for factor, segment_cost in zip(factors, costs[1:]):
        current_count = floor_split_total_count(current_count, factor)
        total_cost += current_count * int(segment_cost)
    return float(total_cost)


def max_floor_split_roots_for_budget(
    runner,
    split_points,
    split_factors,
    *,
    budget: int,
    expected_cost_per_root: float,
) -> int:
    """Find the largest integer root count whose realized cost fits ``budget``."""
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if expected_cost_per_root <= 0.0 or not math.isfinite(expected_cost_per_root):
        raise ValueError("expected_cost_per_root must be finite and positive")

    cost_for_one = floor_split_sampling_cost(runner, split_points, split_factors, 1)
    if cost_for_one > budget:
        raise ValueError(
            f"Budget {budget} is too small; floor split cost for one root is "
            f"{cost_for_one:.6f}"
        )

    upper = max(1, int(budget // expected_cost_per_root))
    while floor_split_sampling_cost(runner, split_points, split_factors, upper) > budget:
        upper //= 2
        if upper < 1:
            raise ValueError(
                f"Budget {budget} is too small; floor split cost for one root is "
                f"{cost_for_one:.6f}"
            )

    lower = upper
    probe = max(upper * 2, 2)
    while floor_split_sampling_cost(runner, split_points, split_factors, probe) <= budget:
        lower = probe
        probe *= 2

    high = probe - 1
    while lower < high:
        mid = (lower + high + 1) // 2
        if floor_split_sampling_cost(runner, split_points, split_factors, mid) <= budget:
            lower = mid
        else:
            high = mid - 1
    return int(lower)


def run_full_trajectory_batch(
    runner,
    *,
    chunk_size: int,
    n0: int,
    generator=None,
    max_sampling_batch_size=None,
):
    """Sample complete trajectories and return stable per-run float32 tensors."""
    chunk_size = int(chunk_size)
    n0 = int(n0)
    max_sampling_batch_size = normalize_max_sampling_batch_size(
        max_sampling_batch_size
    )
    sampling_start = time.perf_counter()

    def sample_batch(total):
        x = runner.sample_prior(total, generator=generator)
        x = runner.sample_segment(
            x,
            runner.start_time,
            runner.end_time,
            generator=generator,
        )
        return runner.postprocess_samples(x)

    if max_sampling_batch_size is not None:
        samples_by_run = collect_run_batches(
            [n0] * chunk_size,
            max_sampling_batch_size,
            sample_batch=lambda total: sample_batch(total).to(dtype=torch.float32),
            empty_template=runner.postprocess_samples(
                runner.sample_prior(0)
            ).to(dtype=torch.float32),
        )
        return samples_by_run, time.perf_counter() - sampling_start

    samples = sample_batch(chunk_size * n0)
    sampling_time = time.perf_counter() - sampling_start
    samples = samples.to(dtype=torch.float32).reshape(chunk_size, n0, -1).contiguous()
    return [samples[idx] for idx in range(chunk_size)], sampling_time


def run_split_trajectory_batch(
    runner,
    *,
    n0_by_run,
    split_points,
    split_factors_by_run,
    generator=None,
    max_sampling_batch_size=None,
):
    """Run generic trajectory splitting using runner-owned sampling primitives.

    ``split_points`` must already be canonicalized by the runner family. The
    ordering of random draws and sampling calls intentionally matches the
    former EDM/SDE implementations.
    """
    if len(n0_by_run) == 0:
        return [], [], 0.0
    if len(n0_by_run) != len(split_factors_by_run):
        raise ValueError(
            "n0_by_run and split_factors_by_run must have the same length"
        )

    device = runner.device
    n0_tensor = torch.as_tensor(n0_by_run, device=device, dtype=torch.long)
    if torch.any(n0_tensor < 1):
        raise ValueError("all n0 values must be at least 1")
    max_sampling_batch_size = normalize_max_sampling_batch_size(
        max_sampling_batch_size
    )
    split_points = list(split_points)
    num_runs = int(n0_tensor.numel())
    realized_costs = torch.zeros(num_runs, device=device, dtype=torch.long)
    sampling_start = time.perf_counter()

    if max_sampling_batch_size is not None and not split_points:
        segment_cost = int(runner.segment_cost(runner.start_time, runner.end_time))

        def sample_full_batch(total):
            x = runner.sample_prior(total, generator=generator)
            x = runner.sample_segment(
                x, runner.start_time, runner.end_time, generator=generator
            )
            return runner.postprocess_samples(x)

        realized_costs += n0_tensor * segment_cost
        samples_by_run = collect_run_batches(
            n0_by_run,
            max_sampling_batch_size,
            sample_batch=sample_full_batch,
            empty_template=runner.postprocess_samples(runner.sample_prior(0)),
        )
        sampling_time = time.perf_counter() - sampling_start
        return samples_by_run, realized_costs.detach().cpu().tolist(), sampling_time

    run_ids = torch.repeat_interleave(
        torch.arange(num_runs, device=device, dtype=torch.long), n0_tensor
    )
    if max_sampling_batch_size is None:
        x = runner.sample_prior(int(n0_tensor.sum().item()), generator=generator)
    else:

        def sample_initial_batch(total):
            batch = runner.sample_prior(total, generator=generator)
            return runner.sample_segment(
                batch,
                runner.start_time,
                split_points[0],
                generator=generator,
            )

        initial_by_run = collect_run_batches(
            n0_by_run,
            max_sampling_batch_size,
            sample_batch=sample_initial_batch,
            empty_template=torch.empty(0, device=device),
        )
        x = torch.cat(initial_by_run, dim=0)

    if not split_points:
        realized_costs += n0_tensor * int(
            runner.segment_cost(runner.start_time, runner.end_time)
        )
        x = runner.sample_segment(
            x, runner.start_time, runner.end_time, generator=generator
        )
    else:
        realized_costs += n0_tensor * int(
            runner.segment_cost(runner.start_time, split_points[0])
        )
        if max_sampling_batch_size is None:
            x = runner.sample_segment(
                x, runner.start_time, split_points[0], generator=generator
            )

        split_factors_array = np.asarray(split_factors_by_run, dtype=float)
        if not np.isfinite(split_factors_array).all() or np.any(
            split_factors_array < 0.0
        ):
            raise ValueError(
                "split_factors_by_run must contain finite nonnegative values"
            )
        if split_factors_array.shape != (num_runs, len(split_points)):
            raise ValueError("split_factors_by_run has incompatible shape")
        split_factors_tensor = torch.as_tensor(
            split_factors_array,
            device=device,
            dtype=torch.float64,
        )

        for idx, split_point in enumerate(split_points):
            end_t = (
                split_points[idx + 1]
                if idx + 1 < len(split_points)
                else runner.end_time
            )
            if max_sampling_batch_size is not None and idx + 1 == len(split_points):
                child_counts = balanced_branch_counts_by_group(
                    run_ids,
                    split_factors_tensor[:, idx],
                    num_groups=num_runs,
                    generator=generator,
                )
                counts = child_counts_by_run(
                    run_ids,
                    child_counts,
                    num_runs=num_runs,
                )
                realized_costs += counts * int(
                    runner.segment_cost(split_point, end_t)
                )
                parts_by_run = [[] for _ in range(num_runs)]
                empty_template = runner.postprocess_samples(x[:0])
                for batch_parent_indices, batch_counts in iter_child_parent_batches_by_run(
                    run_ids,
                    child_counts,
                    num_runs=num_runs,
                    max_count_per_run=max_sampling_batch_size,
                ):
                    if batch_parent_indices.numel() == 0:
                        continue
                    batch = x.index_select(0, batch_parent_indices)
                    batch = runner.sample_segment(
                        batch, split_point, end_t, generator=generator
                    )
                    batch = runner.postprocess_samples(batch)
                    empty_template = batch
                    append_by_counts(parts_by_run, batch, batch_counts)
                sampling_time = time.perf_counter() - sampling_start
                return (
                    cat_parts_by_run(parts_by_run, empty_template),
                    realized_costs.detach().cpu().tolist(),
                    sampling_time,
                )

            x, run_ids = balanced_split_with_run_ids(
                x,
                run_ids,
                split_factors_tensor[:, idx],
                generator=generator,
            )
            counts = torch.bincount(run_ids, minlength=num_runs)
            realized_costs += counts * int(runner.segment_cost(split_point, end_t))
            x = apply_to_run_batches(
                x,
                run_ids,
                num_runs=num_runs,
                max_count_per_run=max_sampling_batch_size,
                fn=lambda batch, split_point=split_point, end_t=end_t: runner.sample_segment(
                    batch,
                    split_point,
                    end_t,
                    generator=generator,
                ),
            )

    x = runner.postprocess_samples(x)
    sampling_time = time.perf_counter() - sampling_start
    samples_by_run = [x[run_ids == run_idx] for run_idx in range(num_runs)]
    return samples_by_run, realized_costs.detach().cpu().tolist(), sampling_time


probabilistic_split_with_run_ids = balanced_split_with_run_ids
