from __future__ import annotations

from runners.sde.cev_security_price.spec import SPEC as CEV_SECURITY_PRICE_SPEC
from runners.sde.simple_ou.spec import SPEC as SIMPLE_OU_SPEC
from runners.sde.smooth_threshold_autoregression.spec import (
    SPEC as SMOOTH_THRESHOLD_AUTOREGRESSION_SPEC,
)

SDE_CASES = {
    SIMPLE_OU_SPEC.name: SIMPLE_OU_SPEC,
    CEV_SECURITY_PRICE_SPEC.name: CEV_SECURITY_PRICE_SPEC,
    SMOOTH_THRESHOLD_AUTOREGRESSION_SPEC.name: SMOOTH_THRESHOLD_AUTOREGRESSION_SPEC,
}
