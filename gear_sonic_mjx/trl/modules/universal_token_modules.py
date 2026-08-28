from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gear_sonic_mjx.config import ModelConfig
from .base_module import MLP
from .fsq import FSQ


@dataclass
class UniversalTokenOutput:
    action_mean: torch.Tensor
    token: torch.Tensor
    token_flat: torch.Tensor
    reconstruction: torch.Tensor | None = None


class UniversalTokenModule(nn.Module):
    """G1-only SONIC universal-token module.

    This intentionally mirrors the public SONIC decomposition (G1 encoder -> FSQ -> dynamic and
    kinematic decoders), while omitting SMPL/teleop/SOMA encoders until they are useful in your stack.
    With the NVIDIA release preset: G1 encoder input is 640, token is [B,2,32], actor decoder input
    is 994 when 10-frame proprioception history is used.
    """

    def __init__(self, cfg: ModelConfig, num_future_frames: int = 10, history_length: int = 10):
        super().__init__()
        self.cfg = cfg
        self.num_future_frames = int(num_future_frames)
        self.history_length = int(history_length)
        self.encoder_input_dim = self.num_future_frames * cfg.g1_future_frame_dim
        self.token_shape = (cfg.num_tokens, cfg.token_dim)
        self.flat_token_dim = cfg.flat_token_dim
        self.proprio_dim = history_length * cfg.proprio_frame_dim
        self.dynamic_input_dim = self.flat_token_dim + self.proprio_dim

        self.g1_encoder = MLP(
            self.encoder_input_dim,
            cfg.g1_encoder_hidden,
            self.flat_token_dim,
        )
        self.quantizer = FSQ(self.flat_token_dim, levels=32, num_tokens=cfg.num_tokens, token_dim=cfg.token_dim)
        self.g1_dynamic_decoder = MLP(
            self.dynamic_input_dim,
            cfg.dynamic_decoder_hidden,
            cfg.dof,
        )
        self.g1_kinematic_decoder = MLP(
            self.flat_token_dim,
            cfg.kinematic_decoder_hidden,
            self.encoder_input_dim,
        )

    def encode(self, g1_encoder_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if g1_encoder_obs.shape[-1] != self.encoder_input_dim:
            raise ValueError(f"G1 encoder expected {self.encoder_input_dim}, got {g1_encoder_obs.shape[-1]}")
        latent = self.g1_encoder(g1_encoder_obs)
        token_flat = self.quantizer(latent)
        token = token_flat.reshape(token_flat.shape[0], *self.token_shape)
        return token, token_flat

    def decode(self, token_flat: torch.Tensor, proprio_history: torch.Tensor) -> torch.Tensor:
        x = torch.cat([token_flat, proprio_history], dim=-1)
        if x.shape[-1] != self.dynamic_input_dim:
            raise ValueError(f"Dynamic decoder expected {self.dynamic_input_dim}, got {x.shape[-1]}")
        return self.g1_dynamic_decoder(x)

    def forward(self, g1_encoder_obs: torch.Tensor, proprio_history: torch.Tensor, compute_aux_loss: bool = True) -> UniversalTokenOutput:
        token, token_flat = self.encode(g1_encoder_obs)
        action = self.decode(token_flat, proprio_history)
        recon = self.g1_kinematic_decoder(token_flat) if compute_aux_loss else None
        return UniversalTokenOutput(action, token, token_flat, recon)

    def reconstruction_loss(self, output: UniversalTokenOutput, target_encoder_obs: torch.Tensor) -> torch.Tensor:
        if output.reconstruction is None:
            raise ValueError("forward(..., compute_aux_loss=True) required")
        return torch.mean((output.reconstruction - target_encoder_obs) ** 2)
