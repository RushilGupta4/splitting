from runners.ddpm.cifar10.runner import DDPMCIFAR10HFRunner
from runners.ddpm.ffhq.runner import LDMFFHQRunner
from runners.ddpm.runner import DDPMRunner

DDPM_RUNNER_CLASSES = {
    DDPMCIFAR10HFRunner.runner_name: DDPMCIFAR10HFRunner,
    LDMFFHQRunner.runner_name: LDMFFHQRunner,
}

__all__ = [
    "DDPMCIFAR10HFRunner",
    "LDMFFHQRunner",
    "DDPMRunner",
    "DDPM_RUNNER_CLASSES",
]
