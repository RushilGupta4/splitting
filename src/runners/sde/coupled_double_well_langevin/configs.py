from runners.sde.configs import sde_case_configs

CONFIGS = sde_case_configs(
    terminal_time=2.0,
    description="Coupled double-well overdamped Langevin adaptive splitting",
)

__all__ = ["CONFIGS"]
