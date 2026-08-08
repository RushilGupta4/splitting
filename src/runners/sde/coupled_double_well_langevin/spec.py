from __future__ import annotations

import math

import torch

from runners.sde.spec import SDECase, diagonal_normal_initial_spec


BARRIER_COEFFICIENT = 4.0
RING_COUPLING = 0.5
INVERSE_TEMPERATURE = 1.0
INITIAL_MEAN = -1.0
INITIAL_VARIANCE = 0.01


def potential(x: torch.Tensor) -> torch.Tensor:
    local = 0.25 * BARRIER_COEFFICIENT * (x.square() - 1.0).square().sum(dim=-1)
    neighbor_delta = torch.roll(x, shifts=-1, dims=-1) - x
    interaction = 0.5 * RING_COUPLING * neighbor_delta.square().sum(dim=-1)
    return local + interaction


def drift(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    local_gradient = BARRIER_COEFFICIENT * x * (x.square() - 1.0)
    coupling_gradient = RING_COUPLING * (
        2.0 * x
        - torch.roll(x, shifts=1, dims=1)
        - torch.roll(x, shifts=-1, dims=1)
    )
    return -(local_gradient + coupling_gradient)


def diffusion(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    return torch.full_like(x, math.sqrt(2.0 / INVERSE_TEMPERATURE))


def diffusion_derivative(t: float, x: torch.Tensor) -> torch.Tensor:
    del t
    return torch.zeros_like(x)


def target_spec(
    terminal_time: float,
    dimension: int,
    coupling_strength: float,
) -> dict:
    return {
        "name": "coupled_double_well_langevin",
        "label": "Coupled double-well overdamped Langevin",
        "dimension": int(dimension),
        "coupling_strength": float(coupling_strength),
        "terminal_time": float(terminal_time),
        "topology": "periodic_nearest_neighbor_ring",
        "initial_distribution": diagonal_normal_initial_spec(
            INITIAL_MEAN,
            INITIAL_VARIANCE,
            int(dimension),
        ),
        "params": {
            "barrier_coefficient": BARRIER_COEFFICIENT,
            "ring_coupling": RING_COUPLING,
            "inverse_temperature": INVERSE_TEMPERATURE,
            "diffusion_scale": math.sqrt(2.0 / INVERSE_TEMPERATURE),
        },
        "cdf": None,
    }


SPEC = SDECase(
    name="coupled_double_well_langevin",
    label="Coupled double-well overdamped Langevin",
    initial_mean=INITIAL_MEAN,
    initial_variance=INITIAL_VARIANCE,
    drift=drift,
    diffusion=diffusion,
    diffusion_derivative=diffusion_derivative,
    target_spec_factory=target_spec,
    diffusion_structure="diagonal",
    mean_field_coupling_default=None,
    description=(
        "Overdamped Langevin dynamics in a quartic double-well potential "
        "with periodic nearest-neighbor coupling."
    ),
)
