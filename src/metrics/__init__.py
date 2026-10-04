"""Metric selection, cache keys and per-trial evaluation.

The two implementations live alongside this module: :mod:`metrics.ks` and
:mod:`metrics.mmd`. Sample coercion shared by both (and by the runners) is in
:mod:`metrics.utils`; MMD measurement spaces are in :mod:`metrics.spaces`.

``mmd`` is the MMD of the runner's postprocessed samples; ``mmd_<space>`` measures the
same samples in another space (``latent``, ``pixel``, ``inception``, ``dino``), each
against that space's reference. All share the ``mmd`` metric_params.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from metrics import spaces
from metrics.mmd import (
    _compute_mmd_metric,
    _mmd_metric_cache_key,
    _mmd_representation,
    _normalize_mmd_params,
    _prepare_mmd_state,
)

MMD_SPACE_METRICS = tuple(f"mmd_{space}" for space in spaces.SPACES)
SUPPORTED_METRICS = ("ks", "mmd", *MMD_SPACE_METRICS)


def is_mmd(metric: str) -> bool:
    return metric == "mmd" or metric in MMD_SPACE_METRICS


def mmd_space(metric: str, runner) -> str:
    """The space an MMD metric measures in; plain ``mmd`` is the runner's sample space."""
    return runner.sample_space if metric == "mmd" else metric[len("mmd_"):]


def metric_spaces(runner, metrics: Sequence[str]) -> list[str]:
    """Spaces other than the sample space that the metrics need references in."""
    return sorted(
        {mmd_space(m, runner) for m in metrics if is_mmd(m)} - {runner.sample_space}
    )


def normalize_metrics(
    raw_metrics=None, *, supported: Sequence[str] | None = None
) -> list[str]:
    if raw_metrics is None:
        raw_metrics = ["ks"]
    elif isinstance(raw_metrics, str):
        raw_metrics = [raw_metrics]
    supported_set = set(SUPPORTED_METRICS if supported is None else supported)
    metrics: list[str] = []
    for raw_metric in raw_metrics:
        metric = str(raw_metric).lower()
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}; available: {list(SUPPORTED_METRICS)}"
            )
        if metric not in supported_set and not (is_mmd(metric) and "mmd" in supported_set):
            raise ValueError(f"Metric {metric!r} is not supported by this runner")
        if metric not in metrics:
            metrics.append(metric)
    if not metrics:
        raise ValueError("metrics must be non-empty")
    return metrics


def normalize_metric_params(raw_params=None) -> dict[str, dict[str, Any]]:
    if raw_params is None:
        return {}
    if not isinstance(raw_params, Mapping):
        raise ValueError("metric_params must be a mapping")
    params: dict[str, dict[str, Any]] = {}
    for raw_metric, raw_value in raw_params.items():
        metric = str(raw_metric).lower()
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Unknown metric_params entry {metric!r}")
        if raw_value is None:
            value = {}
        elif isinstance(raw_value, Mapping):
            value = dict(raw_value)
        else:
            raise ValueError(f"metric_params[{metric!r}] must be a mapping")
        if metric == "ks":
            if value:
                raise ValueError("ks no longer accepts metric parameters")
        elif metric == "mmd":
            value = _normalize_mmd_params(value)
        params[metric] = value
    return params


def validate_metric_dimensions(runner, metrics: Sequence[str]) -> None:
    if "ks" in metrics and int(runner.input_dim) > 2:
        raise ValueError(
            "KS supports only dimensions 1 and 2; "
            f"runner {runner.runner_name!r} has dimension {int(runner.input_dim)}"
        )


def metric_cache_key(runner, metrics: Sequence[str], metric_params: Mapping[str, Any]):
    params = _normalize_mmd_params(metric_params.get("mmd"))
    key: dict[str, Any] = {"metrics": list(metrics)}
    for metric in metrics:
        if not is_mmd(metric):
            continue
        space = mmd_space(metric, runner)
        key[metric] = _mmd_metric_cache_key(
            params,
            representation=_mmd_representation(spaces.space_runner(runner, space), params),
        )
        if metric != "mmd":
            key[metric].update(spaces.space_cache_key(space))
    return key


def prepare_metric_states(
    runner,
    *,
    comparison_mode: str,
    reference_samples=None,
    metrics: Sequence[str],
    metric_params: Mapping[str, Any] | None = None,
    space_references: Mapping[str, Any] | None = None,
    scoring_batch_size: int = spaces.DEFAULT_BATCH_SIZE,
):
    """``space_references`` maps each space in ``metric_spaces`` to its reference samples."""
    metric_params = metric_params or {}
    mmd_params = _normalize_mmd_params(metric_params.get("mmd"))
    validate_metric_dimensions(runner, metrics)
    states: dict[str, Any] = {"scoring_batch_size": int(scoring_batch_size)}
    if "ks" in metrics:
        states["ks"] = runner.prepare_comparison_state(
            comparison_mode=comparison_mode,
            reference_samples=reference_samples,
            metric_params=metric_params.get("ks"),
        )
    for metric in metrics:
        if not is_mmd(metric):
            continue
        space = mmd_space(metric, runner)
        states[metric] = _prepare_mmd_state(
            spaces.space_runner(runner, space),
            comparison_mode=comparison_mode,
            reference_samples=(
                reference_samples if space == runner.sample_space
                else (space_references or {}).get(space)
            ),
            params=mmd_params,
        )
        states[metric]["space"] = space
    return states


def compute_trial_metrics(
    parts,
    runner,
    *,
    comparison_mode: str,
    metric_states: Mapping[str, Any],
    metrics: Sequence[str],
    part_weights=None,
):
    """Metrics of a weighted sample.

    ``parts`` is a sequence of sample blocks (one per mixture component, plus
    the reused phase-1 pilots when there are any) and ``part_weights`` their
    mixture weights.  ``part_weights=None`` weights every observation equally.
    """
    values: dict[str, float] = {}
    payloads: dict[str, Any] = {}
    if "ks" in metrics:
        values["ks"] = float(
            runner.compute_ks_distance(
                parts,
                comparison_mode=comparison_mode,
                comparison_state=metric_states.get("ks"),
                part_weights=part_weights,
            )
        )
    mmd_metrics = [metric for metric in metrics if is_mmd(metric)]
    if mmd_metrics:
        mapped = spaces.map_parts(
            runner,
            parts,
            sorted({metric_states[metric]["space"] for metric in mmd_metrics}),
            batch_size=metric_states.get("scoring_batch_size", spaces.DEFAULT_BATCH_SIZE),
        )
        for metric in mmd_metrics:
            state = metric_states.get(metric)
            values[metric], payloads[metric] = _compute_mmd_metric(
                mapped[state["space"]], state, part_weights=part_weights
            )
    return values, payloads


def summarize_metric_trials(
    trial_results: Sequence[Mapping[str, Any]], metrics: Sequence[str]
):
    summary: dict[str, Any] = {}
    for metric in metrics:
        values = np.array(
            [
                (trial.get("metrics") or {}).get(
                    metric, trial.get("ks_distance", np.nan) if metric == "ks" else np.nan
                )
                for trial in trial_results
            ],
            dtype=float,
        )
        valid = values[~np.isnan(values)]
        summary[f"mean_{metric}"] = float(valid.mean()) if valid.size else float("nan")
        summary[f"std_{metric}"] = (
            float(valid.std(ddof=1)) if valid.size > 1 else float("nan")
        )
        summary[f"n_valid_{metric}"] = int(valid.size)
    return summary
