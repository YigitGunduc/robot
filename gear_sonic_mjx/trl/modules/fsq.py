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

    When `vector-quantize-pytorch` is installed this calls the same public FSQ package used by
    NVIDIA, applying a `[levels] * token_dim` scalar grid independently to each token. Without the
    optional dependency, a parameter-free STE fallback preserves the same tensor contract.
    """
    def __init__(self, dim: int, levels: int = 32, num_tokens: int = 1, token_dim: int | None = None, prefer_upstream: bool = True):
        super().__init__()
        self.dim, self.levels = int(dim), int(levels)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim or (dim // num_tokens))
        if self.num_tokens * self.token_dim != self.dim:
            raise ValueError("num_tokens * token_dim must equal dim")
        self.upstream = None
        if prefer_upstream:
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
