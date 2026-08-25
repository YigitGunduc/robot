from __future__ import annotations

import torch
from torch import nn


class FiniteScalarQuantizer(nn.Module):
    """Small straight-through FSQ bottleneck.

    SONIC uses finite scalar quantization rather than a learned vector codebook.
    This implementation intentionally keeps the idea minimal: every latent scalar
    is bounded with tanh and quantized independently to `levels` uniform values.
    The forward value is discrete; the backward path uses a straight-through
    estimator.
    """

    def __init__(self, dim: int = 64, levels: int = 32):
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.dim = dim
        self.levels = levels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape[-1]}")
        bounded = torch.tanh(x)
        scaled = (bounded + 1.0) * 0.5 * (self.levels - 1)
        indices = torch.round(scaled).clamp_(0, self.levels - 1)
        quantized = indices / (self.levels - 1) * 2.0 - 1.0
        # Straight-through estimator.
        quantized_st = bounded + (quantized - bounded).detach()
        return quantized_st, indices.to(torch.int16)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project already-token-like values to the nearest FSQ grid without tanh."""
        x = x.clamp(-1.0, 1.0)
        scaled = (x + 1.0) * 0.5 * (self.levels - 1)
        idx = torch.round(scaled).clamp_(0, self.levels - 1)
        return idx / (self.levels - 1) * 2.0 - 1.0
