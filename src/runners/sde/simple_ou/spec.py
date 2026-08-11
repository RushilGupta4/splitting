from __future__ import annotations

import torch

from runners.sde.spec import (
    SDECase,
    diagonal_normal_initial_spec,
    normal_initial_spec,
)

THETA = 1.35
MU = -0.2
SIGMA = 0.65
INITIAL_MEAN = 0.0
INITIAL_VARIANCE = 1.0
MEAN_FIELD_COUPLING = 0.25


def drift(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    out = THETA * (MU - x)
    if MEAN_FIELD_COUPLING != 0.0:
        out = out + MEAN_FIELD_COUPLING * (x.mean(dim=-1, keepdim=True) - x)
    return out


def diffusion(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    return torch.full_like(x, SIGMA)


def diffusion_derivative(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    return torch.zeros_like(x)


def target_spec(terminal_time: float, dimension: int) -> dict:
    return {
        "name": "simple_ou",
        "label": "Simple OU",
        "dimension": int(dimension),
        "coupling_strength": float(MEAN_FIELD_COUPLING),
        "terminal_time": float(terminal_time),
        "initial_distribution": diagonal_normal_initial_spec(
            INITIAL_MEAN,
            INITIAL_VARIANCE,
            int(dimension),
        ),
        "params": {
            "theta": THETA,
            "mu": MU,
            "sigma": SIGMA,
            "initial_distribution": normal_initial_spec(INITIAL_MEAN, INITIAL_VARIANCE),
        },
        "cdf": None,
    }


SPEC = SDECase(
    name="simple_ou",
    label="Simple OU",
    initial_mean=INITIAL_MEAN,
    initial_variance=INITIAL_VARIANCE,
    drift=drift,
    diffusion=diffusion,
    diffusion_derivative=diffusion_derivative,
    target_spec_factory=target_spec,
    description="dX_t = 1.35(-0.2-X_t)dt + 0.65dW_t, X_0 ~ N(0, 1).",
)
