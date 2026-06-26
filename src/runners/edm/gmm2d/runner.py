from runners.edm.gmm2d import spec
from runners.edm.runner import EDMRunner


class EDMGMM2DRunner(EDMRunner):
    runner_name = "edm_gmm2d"
    config_module = "runners.edm.gmm2d.configs"
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
    def load_model_from_checkpoint(cls, checkpoint, device: str, no_compile: bool):
        return spec.load_model_from_checkpoint(checkpoint, device, no_compile)

    @staticmethod
    def model_input_dim(model) -> int:
        return spec.model_input_dim(model)

    def observable_values(self, samples, *, comparison_mode: str, x_grid):
        self._validate_mode(comparison_mode)
        return spec.rectangle_indicator_grid(samples, comparison_mode, x_grid)

__all__ = ["EDMGMM2DRunner"]
