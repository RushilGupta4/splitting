from __future__ import annotations

import torch

from runners.sde1d.spec import SDECase, normal_initial_spec

R_RATE = 0.1
SIGMA = 0.2
GAMMA = 1.0
INITIAL_MEAN = 40.0
INITIAL_VARIANCE = 1.0
EPS = 1e-12


def drift(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    return -R_RATE * x


def diffusion(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    x_pos = torch.clamp(x, min=0.0)
    return SIGMA * torch.pow(x_pos, GAMMA)


def diffusion_derivative(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    positive = x > EPS
    safe_x = torch.clamp(x, min=EPS)
    derivative = SIGMA * GAMMA * torch.pow(safe_x, GAMMA - 1.0)
    return torch.where(positive, derivative, torch.zeros_like(x))


def target_spec(terminal_time: float) -> dict:
    return {
        "name": "cev_security_price",
        "label": "Duffie-Glynn CEV security price",
        "dimension": 1,
        "terminal_time": float(terminal_time),
        "params": {
            "r": R_RATE,
            "sigma": SIGMA,
            "gamma": GAMMA,
            "initial_distribution": normal_initial_spec(INITIAL_MEAN, INITIAL_VARIANCE),
        },
        "cdf": None,
    }


SPEC = SDECase(
    name="cev_security_price",
    label="Duffie-Glynn CEV security price",
    initial_mean=INITIAL_MEAN,
    initial_variance=INITIAL_VARIANCE,
    drift=drift,
    diffusion=diffusion,
    diffusion_derivative=diffusion_derivative,
    target_spec_factory=target_spec,
    description=(
        "Duffie-Glynn equation (8): dX_t=-0.1 X_t dt "
        "+ 0.2 X_t^0.25 dW_t, X_0 ~ N(40, 1)."
    ),
)
