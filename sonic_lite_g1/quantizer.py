from __future__ import annotations

import torch
from torch import nn


class ScalarQuantizer(nn.Module):
    """Tiny FSQ-style scalar bottleneck with a straight-through estimator.

    This intentionally avoids a learned VQ codebook.  Each latent component is
    bounded to [-1, 1] and snapped to ``levels`` uniformly-spaced values during
    the forward pass.  Backpropagation sees the pre-rounded value.

    SONIC uses FSQ for the same engineering reason: no codebook maintenance is
    required while PPO is changing the encoder online.
    """

    def __init__(self, levels: int = 32) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.levels = int(levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(x)
        scale = float(self.levels - 1)
        quantized = torch.round((bounded + 1.0) * 0.5 * scale)
        quantized = quantized / scale * 2.0 - 1.0
        # Straight-through estimator: quantized forward, identity-ish backward.
        return bounded + (quantized - bounded).detach()
