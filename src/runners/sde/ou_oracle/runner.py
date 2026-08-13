from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from runners.sde.ou_oracle.oracle import (
    CDF_MAX_POINTS,
    CDF_SEED,
    CDF_TOLERANCE,
    QUERY_PROBABILITIES,
    ou_oracle_allocation,
    ou_oracle_relative_gap,
    ou_oracle_variance_contributions,
)
from runners.sde.runner import SimpleOURunner

_FACTOR_TOLERANCE = 1e-10


class OUOracleRunner(SimpleOURunner):
    """Simple OU simulation with deterministic minimax split factors."""

    runner_name = "ou_oracle"
    config_module = "runners.sde.ou_oracle.configs"

    @classmethod
    def default_checkpoint_path(cls) -> str:
        return SimpleOURunner.default_checkpoint_path()

    def parse_baseline_name(self, name: str) -> dict:
        if name == "ou_oracle":
            return {"mode": "ou_oracle"}
        return super().parse_baseline_name(name)

    def reference_cache_key(
        self,
        comparison_mode: str,
        reference_generation_config: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        key = dict(
            super().reference_cache_key(comparison_mode, reference_generation_config)
        )
        key["runner"] = SimpleOURunner.runner_name
        return key

    def oracle_definition(self, split_percentages: Sequence[float]) -> dict[str, Any]:
        if self.input_dim != 2 or self._sampler != "euler":
            raise ValueError("OU oracle requires the 2D Euler Simple OU runner")
        schedule = tuple(float(value) for value in split_percentages)
        self.resolve_split_percentages(schedule)
        cumulative = ou_oracle_allocation(schedule, steps=self._sampling_steps)
        factors = cumulative[1:] / cumulative[:-1]
        if factors.shape != (len(schedule),):
            raise RuntimeError(
                f"OU oracle produced invalid split factors. Expected shape {(len(schedule),)}, got {factors.shape}"
            )

        if np.any(factors < 1.0 - _FACTOR_TOLERANCE):
            raise RuntimeError(
                f"OU oracle produced invalid split factors. Expected all factors >= 1.0, got {factors}"
            )

        return {
            "version": 1,
            "sampling_steps": int(self._sampling_steps),
            "split_percentages": list(schedule),
            "query_probabilities": list(QUERY_PROBABILITIES),
            "cdf_seed": int(CDF_SEED),
            "cdf_max_points": int(CDF_MAX_POINTS),
            "cdf_tolerance": float(CDF_TOLERANCE),
            "relative_gap": float(
                ou_oracle_relative_gap(schedule, steps=self._sampling_steps)
            ),
            "variance_profile": ou_oracle_variance_contributions(
                schedule, steps=self._sampling_steps
            ).tolist(),
            "cumulative_allocation": cumulative.tolist(),
            "split_factors": factors.tolist(),
        }


__all__ = ["OUOracleRunner"]
