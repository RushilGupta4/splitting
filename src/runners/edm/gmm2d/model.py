import math

import torch
import torch.nn as nn


torch.set_float32_matmul_precision("high")


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalTimeEmbedding dim must be even")
        self.dim = dim
        half_dim = self.dim // 2
        frequency_exponents = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * -frequency_exponents
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, t):
        t = t.to(self.frequencies.dtype)
        embeddings = t[:, None] * self.frequencies[None, :]
        return torch.cat(
            [torch.sin(embeddings), torch.cos(embeddings)], dim=-1
        )


class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim, time_embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.net[0](x)
        h = self.net[1](h)
        h = h + self.time_mlp(t_emb)
        h = self.net[2](h)
        h = self.net[3](h)
        h = self.net[4](h)
        return x + h


class Denoiser(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, time_embed_dim=64, num_blocks=4):
        super().__init__()
        self.input_dim = input_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, time_embed_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embedding(t))
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


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
