"""Target distribution for the 2D Gaussian-mixture DDPM runner.

First migration: delegate to existing top-level `utils.py`. Later phases may
move the hard-coded TARGET_DISTRIBUTION here.
"""

from utils import (
    compute_target_stats,
    denormalize,
    get_target_distribution_spec,
    normalize,
    sample_target_distribution,
    sample_target_spec,
)

__all__ = [
    "compute_target_stats",
    "denormalize",
    "get_target_distribution_spec",
    "normalize",
    "sample_target_distribution",
    "sample_target_spec",
]
