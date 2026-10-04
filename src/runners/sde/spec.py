from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SDECase:
    initial_mean: float
    initial_variance: float
    drift: Callable[[float, torch.Tensor], torch.Tensor]
    diffusion: Callable[[float, torch.Tensor], torch.Tensor]
    target_spec_factory: Callable[[float, int], dict[str, Any]]


def normal_initial_spec(mean: float, variance: float) -> dict[str, Any]:
    return {
        "kind": "normal",
        "mean": float(mean),
        "variance": float(variance),
    }


def diagonal_normal_initial_spec(
    mean: float,
    variance: float,
    dimension: int,
) -> dict[str, Any]:
    dimension = int(dimension)
    return {
        "kind": "diagonal_normal",
        "dimension": dimension,
        "mean": [float(mean)] * dimension,
        "variance": [float(variance)] * dimension,
    }
