from runners.sde.configs import (
    SDE_MMD_SELECTION,
    SDE_METADATA_BASE,
    SDE_SAMPLING_BASE,
)

_DEFAULT_CONFIG = {
    **SDE_SAMPLING_BASE,
    **SDE_METADATA_BASE,
    "comparison_mode": "true_samples",
    "description": (
        "Duffie-Glynn CEV security-price SDE adaptive splitting "
        "with Euler and Milstein"
    ),
}

CONFIGS = {
    "default": _DEFAULT_CONFIG,
    "mmd": {
        **_DEFAULT_CONFIG,
        **SDE_MMD_SELECTION,
        "description": (
            "Duffie-Glynn CEV security-price SDE adaptive splitting with "
            "target-space MMD"
        ),
    },
}

__all__ = ["CONFIGS"]
