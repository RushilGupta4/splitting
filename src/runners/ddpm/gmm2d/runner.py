from runners.ddpm.gmm2d import spec
from runners.ddpm.runner import DDPMRunner


class DDPMGMM2DRunner(DDPMRunner):
    runner_name = "ddpm_gmm2d"
    config_module = "runners.ddpm.gmm2d.configs"
    target_adapter = spec.target
    supported_samplers = spec.SUPPORTED_SAMPLERS
    supported_solvers = spec.SUPPORTED_SOLVERS
    comparison_mode_specs = spec.COMPARISON_MODES

    @classmethod
    def add_train_args(cls, parser) -> None:
        spec.add_train_args(parser)

    @classmethod
    def train_from_args(cls, args) -> None:
        spec.train_from_args(args)

    @classmethod
    def load_model_and_stats(cls, checkpoint_path: str, device: str, *, no_compile: bool):
        return spec.load_model_and_stats(
            checkpoint_path,
            device,
            no_compile=no_compile,
        )

    @staticmethod
    def model_input_dim(model) -> int:
        return spec.model_input_dim(model)

__all__ = ["DDPMGMM2DRunner"]
