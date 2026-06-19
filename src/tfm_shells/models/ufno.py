"""U-FNO time-conditioned: operador espectral (global, membrana) + camino local (alta frecuencia,
flexion) + rama U-Net multiescala. Aislado respecto a EquiNO: misma estructura y time-conditioning
ADITIVO, anadiendo solo (a) una conv local 3x3 por bloque y (b) una rama U-Net sobre las features
finales, para recuperar las frecuencias que la truncacion espectral pierde (concentraciones de flexion).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UFNOOutput:
    sample: torch.Tensor


def _timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10_000) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float()
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.unsqueeze(1)
    exponent = -math.log(float(max_period)) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / max(half_dim, 1)
    freqs = torch.exp(exponent)
    angles = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_height: int, modes_width: int) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes_height, self.modes_width = modes_height, modes_width
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weight_top = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_height, modes_width, 2))
        self.weight_bottom = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_height, modes_width, 2))

    @staticmethod
    def _cmul(x_ft, w):
        return torch.einsum("bixy,ioxy->boxy", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        mh = min(self.modes_height, h)
        mw = min(self.modes_width, w // 2 + 1)
        wt = torch.view_as_complex(self.weight_top[:, :, :mh, :mw].contiguous())
        wb = torch.view_as_complex(self.weight_bottom[:, :, :mh, :mw].contiguous())
        out_ft[:, :, :mh, :mw] = self._cmul(x_ft[:, :, :mh, :mw], wt)
        out_ft[:, :, -mh:, :mw] = self._cmul(x_ft[:, :, -mh:, :mw], wb)
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class UFNOBlock(nn.Module):
    """Bloque de EquiNO + camino LOCAL 3x3 (la esencia U-FNO: recupera alta frecuencia)."""
    def __init__(self, width: int, mh: int, mw: int, time_dim: int, local_kernel: int, dropout: float) -> None:
        super().__init__()
        self.pre_norm = nn.GroupNorm(1, width)
        self.post_norm = nn.GroupNorm(1, width)
        self.spectral = SpectralConv2d(width, width, mh, mw)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.local = nn.Conv2d(width, width, local_kernel, padding=local_kernel // 2)  # <-- camino local
        self.time_proj = nn.Linear(time_dim, width)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(width * 2, width, 1))

    def forward(self, x, temb):
        time_bias = self.time_proj(temb).unsqueeze(-1).unsqueeze(-1)
        r = x
        x = self.pre_norm(x)
        x = self.spectral(x) + self.pointwise(x) + self.local(x) + time_bias
        x = F.gelu(x) + r
        r = x
        x = self.post_norm(x)
        x = self.channel_mlp(x)
        return F.gelu(x) + r


class UNetBranch(nn.Module):
    """Rama U-Net ligera (multiescala local) -> la 'U' del U-FNO."""
    def __init__(self, width: int) -> None:
        super().__init__()
        self.d1 = nn.Conv2d(width, width, 4, 2, 1)
        self.d2 = nn.Conv2d(width, width, 4, 2, 1)
        self.mid = nn.Conv2d(width, width, 3, 1, 1)
        self.u2 = nn.ConvTranspose2d(width, width, 4, 2, 1)
        self.u1 = nn.ConvTranspose2d(width, width, 4, 2, 1)
        self.out = nn.Conv2d(width, width, 1)

    def forward(self, x):
        s1 = x
        d1 = F.gelu(self.d1(x))
        d2 = F.gelu(self.d2(d1))
        m = F.gelu(self.mid(d2))
        u2 = F.gelu(self.u2(m) + d1)
        u1 = F.gelu(self.u1(u2) + s1)
        return self.out(u1)


class _Head(nn.Module):
    def __init__(self, width: int, out_channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, hidden, 1), nn.GELU(), nn.Conv2d(hidden, out_channels, 1))

    def forward(self, x):
        return self.net(x)


class UFNOModel(nn.Module):
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        operator_width: int = 128,
        num_operator_layers: int = 6,
        spectral_modes_height: int = 16,
        spectral_modes_width: int = 16,
        time_embedding_dim: int = 256,
        head_hidden_channels: int = 128,
        local_kernel: int = 3,
        use_unet_branch: bool = True,
        branch_channels: dict[str, int] | None = None,
        use_coordinate_grid: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_coordinate_grid = use_coordinate_grid
        self.time_embedding_dim = time_embedding_dim

        lifted = in_channels + (2 if use_coordinate_grid else 0)
        self.input_projection = nn.Conv2d(lifted, operator_width, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, operator_width), nn.GELU(), nn.Linear(operator_width, time_embedding_dim))
        self.blocks = nn.ModuleList([
            UFNOBlock(operator_width, spectral_modes_height, spectral_modes_width,
                      time_embedding_dim, local_kernel, dropout)
            for _ in range(num_operator_layers)])
        self.unet_branch = UNetBranch(operator_width) if use_unet_branch else None
        self.output_norm = nn.GroupNorm(1, operator_width)
        self.output_projection = nn.Conv2d(operator_width, operator_width, 1)

        branch_channels = branch_channels or {}
        if branch_channels:
            total = sum(int(v) for v in branch_channels.values())
            if total != out_channels:
                raise ValueError(f"Branch channels sum {total} != out_channels {out_channels}")
            self.branch_names = list(branch_channels.keys())
            self.branch_heads = nn.ModuleDict(
                {n: _Head(operator_width, int(branch_channels[n]), head_hidden_channels) for n in self.branch_names})
            self.single_head = None
        else:
            self.branch_names = []
            self.branch_heads = nn.ModuleDict()
            self.single_head = _Head(operator_width, out_channels, head_hidden_channels)

    def _grid(self, b, h, w, device):
        y = torch.linspace(-1.0, 1.0, steps=h, device=device)
        x = torch.linspace(-1.0, 1.0, steps=w, device=device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([gx, gy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, sample: torch.Tensor, timestep) -> UFNOOutput:
        if isinstance(timestep, int):
            timestep = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0]).to(device=sample.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=sample.device, dtype=torch.long)

        x = sample
        if self.use_coordinate_grid:
            x = torch.cat([x, self._grid(sample.shape[0], sample.shape[-2], sample.shape[-1], sample.device)], dim=1)
        x = self.input_projection(x)
        temb = self.time_mlp(_timestep_embedding(timestep, self.time_embedding_dim))

        for block in self.blocks:
            x = block(x, temb)
        if self.unet_branch is not None:
            x = x + self.unet_branch(x)
        x = self.output_norm(x)
        x = F.gelu(self.output_projection(x))

        if self.single_head is not None:
            out = self.single_head(x)
        else:
            out = torch.cat([self.branch_heads[n](x) for n in self.branch_names], dim=1)
        return UFNOOutput(sample=out)
