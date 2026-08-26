"""Metric selection, cache keys and per-trial evaluation.

The two implementations live alongside this module: :mod:`metrics.ks` and
:mod:`metrics.mmd`. Sample coercion shared by both (and by the runners) is in
:mod:`metrics.utils`.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from metrics.mmd import (
    _compute_mmd_metric,
    _mmd_metric_cache_key,
    _mmd_representation,
    _normalize_mmd_params,
    _prepare_mmd_state,
)

SUPPORTED_METRICS = ("ks", "mmd")
REFERENCE_SAMPLE_METRICS = frozenset(SUPPORTED_METRICS)


def normalize_metrics(
    raw_metrics=None, *, supported: Sequence[str] | None = None
) -> list[str]:
    if raw_metrics is None:
        raw_metrics = ["ks"]
    elif isinstance(raw_metrics, str):
        raw_metrics = [raw_metrics]
    supported_set = set(SUPPORTED_METRICS if supported is None else supported)
    metrics: list[str] = []
    seen = set()
    for raw_metric in raw_metrics:
        metric = str(raw_metric).lower()
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}; available: {list(SUPPORTED_METRICS)}"
            )
        if metric not in supported_set:
            raise ValueError(f"Metric {metric!r} is not supported by this runner")
        if metric not in seen:
            metrics.append(metric)
            seen.add(metric)
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


def uses_reference_samples(metrics: Sequence[str]) -> bool:
    return bool(REFERENCE_SAMPLE_METRICS.intersection(metrics))


def validate_metric_dimensions(runner, metrics: Sequence[str]) -> None:
    if "ks" in metrics and int(runner.input_dim) > 2:
        raise ValueError(
            "KS supports only dimensions 1 and 2; "
            f"runner {runner.runner_name!r} has dimension {int(runner.input_dim)}"
        )


def metric_cache_key(runner, metrics: Sequence[str], metric_params: Mapping[str, Any]):
    normalized_params = normalize_metric_params(metric_params)
    validate_metric_dimensions(runner, metrics)
    key: dict[str, Any] = {
        "metrics": list(metrics),
    }
    if "mmd" in metrics:
        params = normalized_params.get("mmd") or {}
        key["mmd"] = _mmd_metric_cache_key(
            params,
            representation=_mmd_representation(runner, params),
        )
    return key


def prepare_metric_states(
    runner,
    *,
    comparison_mode: str,
    reference_samples=None,
    metrics: Sequence[str],
    metric_params: Mapping[str, Any] | None = None,
):
    metric_params = normalize_metric_params(metric_params)
    validate_metric_dimensions(runner, metrics)
    states: dict[str, Any] = {}
    if "ks" in metrics:
        states["ks"] = runner.prepare_comparison_state(
            comparison_mode=comparison_mode,
            reference_samples=reference_samples,
            metric_params=metric_params.get("ks"),
        )
    if "mmd" in metrics:
        states["mmd"] = _prepare_mmd_state(
            runner,
            comparison_mode=comparison_mode,
            reference_samples=reference_samples,
            params=metric_params.get("mmd") or {},
        )
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
    if "mmd" in metrics:
        mmd_value, mmd_payload = _compute_mmd_metric(
            parts,
            metric_states.get("mmd"),
            part_weights=part_weights,
        )
        values["mmd"] = mmd_value
        payloads["mmd"] = mmd_payload
    return values, payloads


def summarize_metric_trials(
    trial_results: Sequence[Mapping[str, Any]], metrics: Sequence[str]
):
    summary: dict[str, Any] = {}
    for metric in metrics:

        def metric_value(trial):
            values = trial.get("metrics") or {}
            if metric in values:
                return values[metric]
            if metric == "ks":
                return trial.get("ks_distance", np.nan)
            return np.nan

        values = np.array(
            [metric_value(trial) for trial in trial_results],
            dtype=float,
        )
        valid = values[~np.isnan(values)]
        summary[f"mean_{metric}"] = float(valid.mean()) if valid.size else float("nan")
        summary[f"std_{metric}"] = (
            float(valid.std(ddof=1)) if valid.size > 1 else float("nan")
        )
        summary[f"n_valid_{metric}"] = int(valid.size)
    return summary


def aggregate_cached_metric_rows(
    trials: Sequence[Mapping[str, Any]], metrics: Sequence[str]
):
    return summarize_metric_trials(trials, metrics)


