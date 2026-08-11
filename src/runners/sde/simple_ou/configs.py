from runners.sde.configs import sde_case_configs

CONFIGS = sde_case_configs(
    terminal_time=1.0,
    description="Simple OU SDE adaptive splitting",
)

__all__ = ["CONFIGS"]
