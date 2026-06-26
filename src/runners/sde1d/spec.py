from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


TensorFn = Callable[[float, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SDECase:
    name: str
    label: str
    initial_mean: float
    initial_variance: float
    description: str
    drift: TensorFn
    diffusion: TensorFn
    diffusion_derivative: TensorFn
    target_spec_factory: Callable[[float], dict[str, Any]]


def normal_initial_spec(mean: float, variance: float) -> dict[str, Any]:
    return {
        "kind": "normal",
        "mean": float(mean),
        "variance": float(variance),
    }


def normal_cdf_spec(mean: float, std: float) -> dict[str, Any]:
    return {
        "kind": "normal",
        "mean": float(mean),
        "std": float(std),
    }
