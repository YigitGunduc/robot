from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, *, activation=nn.ELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.extend([nn.Linear(d, h), activation()])
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        x = t[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb
