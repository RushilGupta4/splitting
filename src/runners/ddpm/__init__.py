from runners.ddpm.cifar10.runner import DDPMCIFAR10HFRunner
from runners.ddpm.ffhq.runner import LDMFFHQRunner

DDPM_RUNNER_CLASSES = {
    DDPMCIFAR10HFRunner.runner_name: DDPMCIFAR10HFRunner,
    LDMFFHQRunner.runner_name: LDMFFHQRunner,
}
