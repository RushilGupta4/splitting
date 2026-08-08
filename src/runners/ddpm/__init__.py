from runners.ddpm.cifar10.runner import DDPMCIFAR10HFRunner
from runners.ddpm.runner import DDPMRunner

DDPM_RUNNER_CLASSES = {
    DDPMCIFAR10HFRunner.runner_name: DDPMCIFAR10HFRunner,
}

__all__ = [
    "DDPMCIFAR10HFRunner",
    "DDPMRunner",
    "DDPM_RUNNER_CLASSES",
]
