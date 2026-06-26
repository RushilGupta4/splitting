from runners.sde1d.configs import (
    SDE1D_METADATA_BASE,
    SDE1D_SAMPLING_BASE,
    make_sde1d_x_grid,
)

X_GRID = make_sde1d_x_grid(-2.75, 2.75, 0.05)

CONFIGS = {
    "default": {
        **SDE1D_SAMPLING_BASE,
        **SDE1D_METADATA_BASE,
        "comparison_mode": "true_samples",
        "description": (
            "Smooth threshold autoregression SDE adaptive splitting with Euler and Milstein"
        ),
        "x_grid": X_GRID,
    },
}

__all__ = ["CONFIGS", "X_GRID"]
