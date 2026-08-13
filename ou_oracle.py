"""Compatibility exports for the OU oracle runner implementation."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runners.sde.ou_oracle.oracle import (
    ou_oracle_allocation,
    ou_oracle_relative_gap,
    ou_oracle_variance_contributions,
    query_variance_contributions,
)

__all__ = [
    "ou_oracle_allocation",
    "ou_oracle_relative_gap",
    "ou_oracle_variance_contributions",
    "query_variance_contributions",
]
