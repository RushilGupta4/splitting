from __future__ import annotations

import torch

from runners.base import ComparisonModeSpec
from runners.ddpm.gmm2d import target
from runners.ddpm.gmm2d.model_io import load_model_and_stats, model_input_dim

SUPPORTED_SOLVERS = ("ddim", "dpmpp_2m")
SUPPORTED_SAMPLERS = ("ddim",)

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
        name="ddpm_samples",
        requires_reference_cache=True,
        reference_uses_sampling_config=False,
        description="Two-sample lower-orthant KS against generated DDPM/DDIM samples.",
    ),
)


def add_train_args(parser) -> None:
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=100)
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
    from runners.ddpm.gmm2d.train import train

    train(args)
