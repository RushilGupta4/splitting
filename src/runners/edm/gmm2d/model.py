import torch
import torch.nn as nn

from runners.ddpm.gmm2d.model import Denoiser


class EDMDenoiser(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_blocks=4, sigma_data=1.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.sigma_data = float(sigma_data)
        self.inner = Denoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )

    def forward(self, x, sigma):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0]).view(-1, 1)
        elif sigma.dim() == 1:
            sigma = sigma.view(-1, 1)
        sigma_sq = sigma ** 2
        sigma_data = torch.as_tensor(self.sigma_data, device=x.device, dtype=x.dtype)
        sd_sq = sigma_data ** 2
        c_skip = sd_sq / (sigma_sq + sd_sq)
        c_out = sigma * sigma_data / torch.sqrt(sigma_sq + sd_sq)
        c_in = 1.0 / torch.sqrt(sigma_sq + sd_sq)
        c_noise = 0.25 * torch.log(sigma).view(-1)
        return c_skip * x + c_out * self.inner(c_in * x, c_noise)
