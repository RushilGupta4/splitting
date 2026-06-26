from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from runners.base import ComparisonModeSpec
from runners.edm.gmm2d import target
from runners.edm.gmm2d.model import EDMDenoiser

SUPPORTED_SOLVERS = ("edm_stochastic", "dpmpp_2s")
SUPPORTED_SAMPLERS = ("edm_stochastic", "dpmpp_2s", "sde_euler_maruyama")

COMPARISON_MODES = (
    ComparisonModeSpec(
        name="true_dist",
        requires_reference_cache=False,
        reference_uses_sampling_config=False,
        description="Exact lower-orthant KS against analytic 2D GMM CDF.",
    ),
    ComparisonModeSpec(
        name="true_samples",
        requires_reference_cache=True,
        reference_uses_sampling_config=False,
        description="Two-sample lower-orthant KS against target samples.",
    ),
    ComparisonModeSpec(
        name="edm_samples",
        requires_reference_cache=True,
        reference_uses_sampling_config=True,
        description="Two-sample lower-orthant KS against generated EDM samples.",
    ),
)


def add_train_args(parser) -> None:
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


def train_from_args(args) -> None:
    from runners.edm.gmm2d.train import train

    train(args)


def load_model_from_checkpoint(
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


def model_input_dim(model) -> int:
    return int(
        getattr(
            model,
            "input_dim",
            getattr(getattr(model, "_orig_mod", None), "input_dim", 2),
        )
    )


def rectangle_indicator_grid(
    values: torch.Tensor,
    comparison_mode: str,
    x_grid: Sequence[float],
) -> torch.Tensor:
    del comparison_mode
    thresholds = torch.as_tensor(x_grid, device=values.device, dtype=values.dtype)
    x1_below = values[:, 0:1] <= thresholds.unsqueeze(0)
    x2_below = values[:, 1:2] <= thresholds.unsqueeze(0)
    return (
        (x1_below.unsqueeze(2) & x2_below.unsqueeze(1))
        .to(values.dtype)
        .reshape(values.shape[0], -1)
    )
