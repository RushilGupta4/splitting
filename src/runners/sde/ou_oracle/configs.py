from copy import deepcopy

from runners.sde.simple_ou.configs import CONFIGS as SIMPLE_OU_CONFIGS

_default = deepcopy(SIMPLE_OU_CONFIGS["default"])
_default.update(
    {
        "B_list": [max(_default["B_list"])],
        "B1_list": [],
        "baselines": ["ou_oracle"],
        "description": "Simple OU finite-query minimax oracle benchmark",
    }
)

CONFIGS = {"default": _default}

__all__ = ["CONFIGS"]
