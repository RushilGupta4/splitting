from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


TensorFn = Callable[[float, torch.Tensor], torch.Tensor]
TargetSpecFactory = Callable[[float, int, float], dict[str, Any]]
InitialSampler = Callable[..., torch.Tensor]
TensorTransform = Callable[[torch.Tensor], torch.Tensor]


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
    target_spec_factory: TargetSpecFactory
    diffusion_structure: str = "diagonal"
    initial_sampler: InitialSampler | None = None
    terminal_transform: TensorTransform | None = None
    mean_field_coupling_default: float | None = 0.25


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


def normal_cdf_spec(mean: float, std: float) -> dict[str, Any]:
    return {
        "kind": "normal",
        "mean": float(mean),
        "std": float(std),
    }
