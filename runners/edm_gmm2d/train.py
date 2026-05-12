from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from runners.edm_gmm2d.model import EDMDenoiser
from runners.edm_gmm2d.runner import EDMGMM2DRunner
from runners.edm_gmm2d.sampling import EDMSchedule, heun_sample_segment
from runners.edm_gmm2d.target import (
    compute_target_stats,
    get_target_distribution_spec,
    normalize,
    sample_target_distribution,
)

RUNNER_NAME = "edm_gmm2d"
RUNNER_VERSION = 1


def _gaussian_pdf_1d(x, mean, std):
    var = std * std
    return np.exp(-0.5 * ((x - mean) ** 2) / var) / np.sqrt(2.0 * np.pi * var)


def _mixture_marginal_pdf_1d(x, weights, means, stds):
    pdf = np.zeros_like(x, dtype=np.float64)
    for weight, mean, std in zip(weights, means, stds):
        pdf += float(weight) * _gaussian_pdf_1d(x, float(mean), float(std))
    return pdf


def plot_final_marginals(model, args, data_mean, data_std, output_path):
    model.eval()
    schedule = EDMSchedule(
        sampling_steps=int(args.sampling_steps),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        rho=float(args.rho),
        device=args.device,
    )
    with torch.inference_mode():
        x = torch.randn(int(args.num_plot_samples), 2, device=args.device) * float(args.sigma_max)
        generated = heun_sample_segment(model, x, float(args.sigma_max), 0.0, schedule)
        generated = generated * data_std + data_mean
    generated_np = generated.detach().cpu().numpy()
    target_spec = get_target_distribution_spec()
    weights = np.asarray(target_spec["weights"], dtype=np.float64)
    means = np.asarray(target_spec["means"], dtype=np.float64)
    covariances = np.asarray(target_spec["covariances"], dtype=np.float64)
    marginal_stds = np.sqrt(
        np.stack([covariances[:, 0, 0], covariances[:, 1, 1]], axis=1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for dim, label in enumerate(("x1", "x2")):
        ax = axes[dim]
        generated_dim = generated_np[:, dim]
        low = np.percentile(generated_dim, 0.5)
        high = np.percentile(generated_dim, 99.5)
        span = max(high - low, 1e-6)
        low -= 0.1 * span
        high += 0.1 * span
        x_grid = np.linspace(low, high, 800)
        true_pdf = _mixture_marginal_pdf_1d(
            x_grid,
            weights=weights,
            means=means[:, dim],
            stds=marginal_stds[:, dim],
        )
        ax.plot(x_grid, true_pdf, color="black", linewidth=2.0, label="True marginal")
        ax.hist(
            generated_dim,
            bins=int(args.num_plot_bins),
            range=(low, high),
            density=True,
            alpha=0.5,
            color="#1f77b4",
            label="Generated",
        )
        ax.set_title(f"Marginal of {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_xlim(low, high)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved marginal comparison plot to {output_path}")


def train(args):
    target_spec = get_target_distribution_spec()
    print(f"Training runner={RUNNER_NAME} (EDM, 2D Gaussian-mixture target)")
    print(f"Device: {args.device}")
    print(f"Target spec: {target_spec}")
    os.makedirs(EDMGMM2DRunner.runner_dir(), exist_ok=True)

    samples = sample_target_distribution(args.num_samples, args.device)
    data_mean, data_std, data_cov = compute_target_stats(device=args.device)
    samples_normalized = normalize(samples, data_mean, data_std)
    loader = DataLoader(
        TensorDataset(samples_normalized),
        batch_size=int(args.batch_size),
        shuffle=True,
    )

    model = EDMDenoiser(
        input_dim=2,
        hidden_dim=int(args.hidden_dim),
        num_blocks=int(args.num_blocks),
        sigma_data=float(args.sigma_data),
    ).to(args.device)
    optimizer = Adam(model.parameters(), lr=float(args.lr))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=1e-6)

    model.train()
    losses = []
    pbar = tqdm(range(int(args.epochs)), desc="Training")
    for _epoch in pbar:
        epoch_losses = []
        for (y,) in loader:
            log_sigma = float(args.P_mean) + float(args.P_std) * torch.randn(
                y.shape[0], 1, device=args.device
            )
            sigma = torch.exp(log_sigma)
            x = y + torch.randn_like(y) * sigma
            pred = model(x, sigma)
            weight = (sigma ** 2 + float(args.sigma_data) ** 2) / (
                sigma * float(args.sigma_data)
            ) ** 2
            loss = (weight * (pred - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        pbar.set_postfix({"loss": f"{avg_loss:.6f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        scheduler.step()

    checkpoint = {
        "runner": RUNNER_NAME,
        "runner_version": RUNNER_VERSION,
        "epoch": int(args.epochs),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": losses[-1],
        "args": vars(args),
        "sampling_defaults": {
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "rho": float(args.rho),
            "sampling_steps": int(args.sampling_steps),
            "S_churn": 0.0,
        },
        "sigma_data": float(args.sigma_data),
        "target_spec": target_spec,
        "data_mean": data_mean.detach().cpu().tolist(),
        "data_std": data_std.detach().cpu().tolist(),
        "data_cov": data_cov.detach().cpu().tolist(),
    }
    final_path = EDMGMM2DRunner.default_checkpoint_path()
    torch.save(checkpoint, final_path)
    print(f"\nFinal model saved to {final_path}")
    print(f"Final loss: {losses[-1]:.6f}")

    marginal_plot_path = args.marginal_plot_path
    if marginal_plot_path is None:
        marginal_plot_path = os.path.join(EDMGMM2DRunner.runner_dir(), "marginals_final.png")
    plot_final_marginals(model, args, data_mean, data_std, marginal_plot_path)
