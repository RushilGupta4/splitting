"""Training implementation for the DDPM/GMM2D runner."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from runners.ddpm.gmm2d.model import Denoiser
from runners.ddpm.gmm2d.runner import DDPMGMM2DRunner
from runners.ddpm.gmm2d.target import (
    compute_target_stats,
    get_target_distribution_spec,
    normalize,
    sample_target_distribution,
)

RUNNER_NAME = "ddpm_gmm2d"
RUNNER_VERSION = 1


class DDPM:
    """DDPM noise schedule and forward process."""

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

    def p_sample_step(self, model, x_t, t):
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        with torch.no_grad():
            eps_theta = model(x_t, t_tensor)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod_prev[t]

        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (x_t - coef * eps_theta) / torch.sqrt(alpha_t)

        if t > 0:
            posterior_var_t = beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(torch.clamp(posterior_var_t, min=1e-20)) * noise
        return mean

    def sample(self, model, num_samples, input_dim):
        x_t = torch.randn(num_samples, input_dim, device=self.device)
        for t in tqdm(range(self.T - 1, -1, -1), desc="Sampling", leave=False):
            x_t = self.p_sample_step(model, x_t, t)
        return x_t


def _gaussian_pdf_1d(x, mean, std):
    var = std * std
    return np.exp(-0.5 * ((x - mean) ** 2) / var) / np.sqrt(2.0 * np.pi * var)


def _mixture_marginal_pdf_1d(x, weights, means, stds):
    pdf = np.zeros_like(x, dtype=np.float64)
    for weight, mean, std in zip(weights, means, stds):
        pdf += float(weight) * _gaussian_pdf_1d(x, float(mean), float(std))
    return pdf


def plot_final_marginals(
    model,
    ddpm,
    data_mean,
    data_std,
    num_samples,
    num_bins,
    output_path,
):
    """Plot x1/x2 marginals: true distribution vs generated distribution."""
    model.eval()
    with torch.no_grad():
        generated_normalized = ddpm.sample(model, num_samples=num_samples, input_dim=2)
        generated = generated_normalized * data_std + data_mean

    generated_np = generated.detach().cpu().numpy()
    target_spec = get_target_distribution_spec()
    weights = np.asarray(target_spec["weights"], dtype=np.float64)
    means = np.asarray(target_spec["means"], dtype=np.float64)
    covariances = np.asarray(target_spec["covariances"], dtype=np.float64)
    marginal_stds = np.sqrt(
        np.stack([covariances[:, 0, 0], covariances[:, 1, 1]], axis=1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["x1", "x2"]

    for dim in range(2):
        ax = axes[dim]
        generated_dim = generated_np[:, dim]

        low = np.percentile(generated_dim, 0.5)
        high = np.percentile(generated_dim, 99.5)
        span = high - low
        low -= 0.1 * span
        high += 0.1 * span

        plot_points = np.linspace(low, high, 800)
        true_pdf = _mixture_marginal_pdf_1d(
            plot_points,
            weights=weights,
            means=means[:, dim],
            stds=marginal_stds[:, dim],
        )

        ax.plot(plot_points, true_pdf, color="black", linewidth=2.0, label="True marginal")
        ax.hist(
            generated_dim,
            bins=num_bins,
            range=(low, high),
            density=True,
            alpha=0.5,
            color="#1f77b4",
            label="Generated",
        )
        ax.set_title(f"Marginal of {labels[dim]}")
        ax.set_xlabel(labels[dim])
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_xlim(low, high)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved marginal comparison plot to {output_path}")


def create_dataloader(num_samples, batch_size, device):
    print(f"Generating {num_samples} samples from the 2D Gaussian mixture target...")
    samples = sample_target_distribution(num_samples, device)
    data_mean, data_std, data_cov = compute_target_stats(device=device)
    samples_normalized = normalize(samples, data_mean, data_std)

    print(f"  Data mean={data_mean.tolist()}")
    print(f"  Data std={data_std.tolist()}")
    print(f"  Data covariance={data_cov.tolist()}")
    print(
        "  Normalized sample mean="
        f"{samples_normalized.mean(dim=0).tolist()}, std={samples_normalized.std(dim=0).tolist()}"
    )

    dataset = TensorDataset(samples_normalized)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, data_mean, data_std, data_cov


def train(args):
    target_spec = get_target_distribution_spec()
    print(f"Training runner={RUNNER_NAME} (DDPM, 2D Gaussian-mixture target)")
    print(f"Device: {args.device}")
    print(f"Target spec: {target_spec}")

    os.makedirs(DDPMGMM2DRunner.runner_dir(), exist_ok=True)

    model = Denoiser(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    ).to(args.device)

    ddpm = DDPM(T=args.T, device=args.device)

    dataloader, data_mean, data_std, data_cov = create_dataloader(
        args.num_samples, args.batch_size, args.device
    )

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    model.train()
    losses = []

    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        epoch_losses = []
        for (x_0,) in dataloader:
            t = torch.randint(0, args.T, (x_0.shape[0],), device=args.device)
            x_t, noise = ddpm.q_sample(x_0, t)
            noise_pred = model(x_t, t)
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        pbar.set_postfix(
            {"loss": f"{avg_loss:.6f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
        )
        scheduler.step()

    final_checkpoint = {
        "runner": RUNNER_NAME,
        "runner_version": RUNNER_VERSION,
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": losses[-1],
        "args": vars(args),
        "sampling_defaults": {"T": int(args.T)},
        "target_spec": target_spec,
        "data_mean": data_mean.detach().cpu().tolist(),
        "data_std": data_std.detach().cpu().tolist(),
        "data_cov": data_cov.detach().cpu().tolist(),
    }
    final_path = DDPMGMM2DRunner.default_checkpoint_path()
    torch.save(final_checkpoint, final_path)
    print(f"\nFinal model saved to {final_path}")
    print(f"  Normalization mean={data_mean.tolist()}")
    print(f"  Normalization std={data_std.tolist()}")

    print("\nTraining complete!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Mean loss (last 100): {sum(losses[-100:]) / min(100, len(losses)):.6f}")

    marginal_plot_path = args.marginal_plot_path
    if marginal_plot_path is None:
        marginal_plot_path = os.path.join(
            DDPMGMM2DRunner.runner_dir(), "marginals_final.png"
        )

    plot_final_marginals(
        model=model,
        ddpm=ddpm,
        data_mean=data_mean,
        data_std=data_std,
        num_samples=args.num_plot_samples,
        num_bins=args.num_plot_bins,
        output_path=marginal_plot_path,
    )
