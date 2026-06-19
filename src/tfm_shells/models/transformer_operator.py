"""Transformer neural operator (estilo DiT) para el surrogate mecanico.

Sesgo inductivo distinto al espectral: autoatencion GLOBAL sobre tokens de parche (captura la
dependencia apoyos->toda la lamina sin truncar modos), con time-conditioning moderno adaLN-Zero
(Peebles & Xie 2023, DiT). Dimensionado a ~params de equino para comparacion justa.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerOperatorOutput:
    sample: torch.Tensor


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    if t.ndim == 0:
        t = t[None]
    t = t.float()
    half = dim // 2
    if half == 0:
        return t.unsqueeze(1)
    exponent = -math.log(float(max_period)) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    freqs = torch.exp(exponent)
    ang = t.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.adaln(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), sh_a, sc_a)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g_a.unsqueeze(1) * a
        x = x + g_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh_m, sc_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch * patch * out_channels)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaln(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class TransformerOperator(nn.Module):
    def __init__(
        self,
        sample_size: int = 64,
        in_channels: int = 2,
        out_channels: int = 13,
        patch_size: int = 4,
        embed_dim: int = 640,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        time_embedding_dim: int = 256,
        use_coordinate_grid: bool = True,
        dropout: float = 0.0,
        **_ignored,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.patch = patch_size
        self.use_coordinate_grid = use_coordinate_grid
        self.grid = sample_size // patch_size
        self.time_embedding_dim = time_embedding_dim
        lifted = in_channels + (2 if use_coordinate_grid else 0)

        self.patch_embed = nn.Conv2d(lifted, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.blocks = nn.ModuleList([DiTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.final = FinalLayer(embed_dim, patch_size, out_channels)
        self._init_zero()

    def _init_zero(self):
        for b in self.blocks:
            nn.init.zeros_(b.adaln[-1].weight); nn.init.zeros_(b.adaln[-1].bias)   # adaLN-Zero
        nn.init.zeros_(self.final.adaln[-1].weight); nn.init.zeros_(self.final.adaln[-1].bias)
        nn.init.zeros_(self.final.linear.weight); nn.init.zeros_(self.final.linear.bias)

    def _grid_coords(self, b, h, w, device):
        y = torch.linspace(-1.0, 1.0, steps=h, device=device)
        x = torch.linspace(-1.0, 1.0, steps=w, device=device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([gx, gy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)

    def _unpatchify(self, x):
        b = x.shape[0]; g = self.grid; p = self.patch; c = self.out_channels
        x = x.view(b, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    def forward(self, sample: torch.Tensor, timestep) -> TransformerOperatorOutput:
        if isinstance(timestep, int):
            timestep = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0]).to(device=sample.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=sample.device, dtype=torch.long)

        x = sample
        if self.use_coordinate_grid:
            x = torch.cat([x, self._grid_coords(sample.shape[0], sample.shape[-2], sample.shape[-1], sample.device)], dim=1)
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed   # (B, N, D)
        c = self.time_mlp(_timestep_embedding(timestep, self.time_embedding_dim))
        for block in self.blocks:
            x = block(x, c)
        x = self.final(x, c)
        return TransformerOperatorOutput(sample=self._unpatchify(x))
