from runners.ddpm.gmm2d.runner import DDPMGMM2DRunner
from runners.ddpm.mnist.runner import DDPMMNISTRunner
from runners.ddpm.runner import DDPMRunner

DDPM_RUNNER_CLASSES = {
    DDPMGMM2DRunner.runner_name: DDPMGMM2DRunner,
    DDPMMNISTRunner.runner_name: DDPMMNISTRunner,
}

__all__ = [
    "DDPMGMM2DRunner",
    "DDPMMNISTRunner",
    "DDPMRunner",
    "DDPM_RUNNER_CLASSES",
]
