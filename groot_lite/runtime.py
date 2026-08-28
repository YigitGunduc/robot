from __future__ import annotations

import torch

from gear_sonic_mjx.trl.modules.universal_token_modules import UniversalTokenModule


class SonicTokenController:
    """Runtime bridge: GR00T motion token -> SONIC dynamic decoder -> 29 normalized actions."""

    def __init__(self, sonic: UniversalTokenModule):
        self.sonic = sonic.eval()

    @torch.no_grad()
    def __call__(
        self, motion_token: torch.Tensor, proprio_history: torch.Tensor
    ) -> torch.Tensor:
        if motion_token.shape[-1] != self.sonic.flat_token_dim:
            raise ValueError(f"expected {self.sonic.flat_token_dim}-D SONIC token")
        return self.sonic.decode(motion_token, proprio_history)


class RecedingHorizonTokenBuffer:
    """Simple action-chunk execution buffer.

    GR00T can predict H tokens at low frequency; SONIC consumes one token per 50-Hz step. Refresh
    this buffer before it empties to use receding-horizon control instead of executing all H open-loop.
    """

    def __init__(self, horizon: int, token_dim: int):
        self.horizon, self.token_dim = horizon, token_dim
        self.chunk: torch.Tensor | None = None
        self.index = 0

    def update(self, chunk: torch.Tensor) -> None:
        if chunk.ndim != 3 or chunk.shape[1:] != (self.horizon, self.token_dim):
            raise ValueError(f"expected [B,{self.horizon},{self.token_dim}]")
        self.chunk = chunk
        self.index = 0

    def next(self) -> torch.Tensor:
        if self.chunk is None:
            raise RuntimeError("no token chunk loaded")
        i = min(self.index, self.horizon - 1)
        token = self.chunk[:, i]
        self.index += 1
        return token

    @property
    def remaining(self) -> int:
        return max(self.horizon - self.index, 0)
