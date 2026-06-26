from __future__ import annotations

import torch

from runners.sde1d.spec import SDECase, normal_initial_spec

CENTER = 0.0
WIDTH = 0.35
KAPPA_LOW = 2.0
MU_LOW = -0.8
KAPPA_HIGH = 1.2
MU_HIGH = 0.9
SIGMA0 = 0.35
RHO = 0.6
INITIAL_MEAN = -0.2
INITIAL_VARIANCE = 0.8


def _switch(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid((x - CENTER) / WIDTH)


def drift(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    switch = _switch(x)
    low_regime = KAPPA_LOW * (MU_LOW - x)
    high_regime = KAPPA_HIGH * (MU_HIGH - x)
    return (1.0 - switch) * low_regime + switch * high_regime


def diffusion(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    return SIGMA0 * (1.0 + RHO * _switch(x))


def diffusion_derivative(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    switch = _switch(x)
    return SIGMA0 * RHO * switch * (1.0 - switch) / WIDTH


def target_spec(terminal_time: float) -> dict:
    return {
        "name": "smooth_threshold_autoregression",
        "label": "Smooth threshold autoregression",
        "dimension": 1,
        "terminal_time": float(terminal_time),
        "params": {
            "center": CENTER,
            "width": WIDTH,
            "kappa_low": KAPPA_LOW,
            "mu_low": MU_LOW,
            "kappa_high": KAPPA_HIGH,
            "mu_high": MU_HIGH,
            "sigma0": SIGMA0,
            "rho": RHO,
            "initial_distribution": normal_initial_spec(
                INITIAL_MEAN,
                INITIAL_VARIANCE,
            ),
        },
        "cdf": None,
    }


SPEC = SDECase(
    name="smooth_threshold_autoregression",
    label="Smooth threshold autoregression",
    initial_mean=INITIAL_MEAN,
    initial_variance=INITIAL_VARIANCE,
    drift=drift,
    diffusion=diffusion,
    diffusion_derivative=diffusion_derivative,
    target_spec_factory=target_spec,
    description=(
        "Smooth threshold autoregression with sigmoid regime switching: "
        "dX_t = [(1-s)2.0(-0.8-X_t) + s1.2(0.9-X_t)]dt "
        "+ 0.35(1 + 0.6s)dW_t, s = sigmoid(X_t / 0.35), "
        "X_0 ~ N(-0.2, 0.8)."
    ),
)
