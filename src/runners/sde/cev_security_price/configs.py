from runners.sde.configs import (
    SDE_METADATA_BASE,
    SDE_SAMPLING_BASE,
)

CONFIGS = {
    "default": {
        **SDE_SAMPLING_BASE,
        **SDE_METADATA_BASE,
        "comparison_mode": "true_samples",
        "description": (
            "Duffie-Glynn CEV security-price SDE adaptive splitting "
            "with Euler and Milstein"
        ),
    },
}

__all__ = ["CONFIGS"]
