from __future__ import annotations

from typing import Any, Mapping

import torch

from runners.base import ComparisonModeSpec
from runners.edm.gmm2d import target
from runners.edm.gmm2d.model import EDMDenoiser
from runners.edm.runner import EDMRunner


class EDMGMM2DRunner(EDMRunner):
    runner_name = "edm_gmm2d"
    config_module = "runners.edm.gmm2d.configs"
    target_adapter = target
    supported_samplers = ("edm_stochastic", "dpmpp_2s", "sde_euler_maruyama")
    supported_solvers = ("edm_stochastic", "dpmpp_2s")
    comparison_mode_specs = (
        ComparisonModeSpec(name="true_dist", requires_reference_cache=False),
        ComparisonModeSpec(name="true_samples", requires_reference_cache=True),
        ComparisonModeSpec(name="edm_samples", requires_reference_cache=True),
    )

    @classmethod
    def add_train_args(cls, parser) -> None:
        parser.add_argument("--sigma_min", type=float, default=0.002)
        parser.add_argument("--sigma_max", type=float, default=80.0)
        parser.add_argument("--rho", type=float, default=7.0)
        parser.add_argument("--sampling_steps", type=int, default=18)
        parser.add_argument("--sigma_data", type=float, default=1.0)
        parser.add_argument("--P_mean", type=float, default=-1.2)
        parser.add_argument("--P_std", type=float, default=1.2)
        parser.add_argument("--num_samples", type=int, default=100_000)
        parser.add_argument("--batch_size", type=int, default=10_000)
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--num_blocks", type=int, default=4)
        parser.add_argument("--num_plot_samples", type=int, default=20_000)
        parser.add_argument("--num_plot_bins", type=int, default=100)
        parser.add_argument("--marginal_plot_path", type=str, default=None)
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0" if torch.cuda.is_available() else "cpu",
        )

    @classmethod
    def train_from_args(cls, args) -> None:
        from runners.edm.gmm2d.train import train

        train(args)

    @classmethod
    def load_model_from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        device: str,
        no_compile: bool,
    ) -> tuple[Any, float]:
        model_args = checkpoint.get("args", {})
        sigma_data = float(checkpoint.get("sigma_data", model_args.get("sigma_data", 1.0)))
        model = EDMDenoiser(
            input_dim=len(checkpoint.get("data_mean", [0.0, 0.0])),
            hidden_dim=int(model_args.get("hidden_dim", 128)),
            num_blocks=int(model_args.get("num_blocks", 4)),
            sigma_data=sigma_data,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        if hasattr(torch, "compile") and not no_compile:
            model = torch.compile(model, dynamic=True)
        return model, sigma_data

    @staticmethod
    def model_input_dim(model) -> int:
        return int(
            getattr(
                model,
                "input_dim",
                getattr(getattr(model, "_orig_mod", None), "input_dim", 2),
            )
        )
