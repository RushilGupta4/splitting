from runners.ddpm.gmm2d.runner import DDPMGMM2DRunner
from runners.ddpm.runner import DDPMRunner

DDPM_RUNNER_CLASSES = {
    DDPMGMM2DRunner.runner_name: DDPMGMM2DRunner,
}

__all__ = ["DDPMGMM2DRunner", "DDPMRunner", "DDPM_RUNNER_CLASSES"]
