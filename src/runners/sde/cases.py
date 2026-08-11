from __future__ import annotations

from runners.sde.coupled_double_well_langevin.spec import (
    SPEC as COUPLED_DOUBLE_WELL_LANGEVIN_SPEC,
)
from runners.sde.simple_ou.spec import SPEC as SIMPLE_OU_SPEC

SDE_CASES = {
    SIMPLE_OU_SPEC.name: SIMPLE_OU_SPEC,
    COUPLED_DOUBLE_WELL_LANGEVIN_SPEC.name: COUPLED_DOUBLE_WELL_LANGEVIN_SPEC,
}
