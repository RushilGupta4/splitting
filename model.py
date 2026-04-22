import torch
import torch.nn as nn
import math

torch.set_float32_matmul_precision("high")


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        frequency_exponents = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * -frequency_exponents
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, t):
        embeddings = t[:, None] * self.frequencies[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResNetBlock(nn.Module):
    """Residual block with time embedding injection."""

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
        h = self.net(x)
        h = h + self.time_mlp(t_emb)
        return self.activation(x + h)


class Denoiser(nn.Module):
    """
    Simple ResNet-based denoiser for low-dimensional data.
    Predicts noise given noisy sample x_t and timestep t.
    """

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
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    # @torch.compile()
    def forward(self, x, t):
        """
        Args:
            x: noisy sample, shape (batch_size, input_dim)
            t: timestep, shape (batch_size,)
        Returns:
            predicted noise, shape (batch_size, input_dim)
        """
        # Time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Input projection
        h = self.input_proj(x)

        # ResNet blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # Output projection
        return self.output_proj(h)


if __name__ == "__main__":
    # Quick test
    model = Denoiser(input_dim=2, hidden_dim=128, time_embed_dim=64, num_blocks=4)
    x = torch.randn(32, 2)
    t = torch.randint(0, 1000, (32,))
    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
