from __future__ import annotations

import torch
from torch import nn


class ScalarFSQFallback(nn.Module):
    """Dependency-free finite scalar quantizer with STE."""

    def __init__(self, dim: int, levels: int = 32):
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.dim, self.levels = int(dim), int(levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(x)
        scale = float(self.levels - 1)
        u = (z + 1.0) * 0.5 * scale
        q = torch.round(u)
        q_ste = u + (q - u).detach()
        return (2.0 * q_ste / scale) - 1.0


class FSQ(nn.Module):
    """SONIC-shaped FSQ wrapper.

    ``vector-quantize-pytorch`` enumerates its implicit codebook during construction. That is useful
    for small FSQ grids but impossible for SONIC's 32-level, 32-D tokens (``32**32`` entries). The
    parameter-free scalar implementation is therefore the normal SONIC path. The upstream package
    is used only when its complete codebook is small enough to enumerate safely.
    """

    MAX_ENUMERATED_CODEBOOK_SIZE = 65_536

    def __init__(
        self,
        dim: int,
        levels: int = 32,
        num_tokens: int = 1,
        token_dim: int | None = None,
        prefer_upstream: bool = True,
    ):
        super().__init__()
        self.dim, self.levels = int(dim), int(levels)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim or (dim // num_tokens))
        if self.num_tokens * self.token_dim != self.dim:
            raise ValueError("num_tokens * token_dim must equal dim")
        self.upstream = None
        codebook_size = self.levels**self.token_dim
        if prefer_upstream and codebook_size <= self.MAX_ENUMERATED_CODEBOOK_SIZE:
            try:
                from vector_quantize_pytorch import FSQ as UpstreamFSQ

                self.upstream = UpstreamFSQ(levels=[self.levels] * self.token_dim)
            except ImportError:
                pass
        self.fallback = ScalarFSQFallback(self.dim, self.levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upstream is None:
            return self.fallback(x)
        original = x.shape
        tok = x.reshape(*original[:-1], self.num_tokens, self.token_dim)
        quantized, _indices = self.upstream(tok)
        return quantized.reshape(original)

    @torch.no_grad()
    def codes(self, x: torch.Tensor) -> torch.Tensor:
        if self.upstream is not None:
            tok = x.reshape(*x.shape[:-1], self.num_tokens, self.token_dim)
            _q, idx = self.upstream(tok)
            return idx
        z = torch.tanh(x)
        return torch.round((z + 1.0) * 0.5 * (self.levels - 1)).long()
