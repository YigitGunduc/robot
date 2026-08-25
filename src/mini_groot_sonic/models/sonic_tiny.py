from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from mini_groot_sonic.config import GoalConfig, SonicTinyConfig
from mini_groot_sonic.models.common import mlp
from mini_groot_sonic.models.fsq import FiniteScalarQuantizer


@dataclass
class SonicOutput:
    action_mean: torch.Tensor
    token: torch.Tensor
    token_indices: torch.Tensor
    reconstruction: torch.Tensor


class SparseGoalEncoder(nn.Module):
    """Encodes sparse SE(3) body targets plus per-slot masks into token residuals."""

    def __init__(self, cfg: GoalConfig, token_dim: int):
        super().__init__()
        self.cfg = cfg
        self.net = mlp(cfg.flat_dim, cfg.hidden, token_dim)

    def forward(self, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # targets: [B, S, 7], masks: [B, S]
        if targets.ndim != 3 or masks.ndim != 2:
            raise ValueError("targets must be [B,S,7] and masks [B,S]")
        x = torch.cat([targets, masks[..., None]], dim=-1).flatten(1)
        return self.net(x)


class TinySonicPolicy(nn.Module):
    """Compact SONIC-style universal token controller.

    Reference future q/qdot -> G1 encoder -> FSQ 64D token.
    Quantized token + proprioceptive history -> dynamic decoder -> 29 actions.
    A kinematic decoder reconstructs the future reference for the auxiliary loss.
    """

    def __init__(self, cfg: SonicTinyConfig, goal_cfg: GoalConfig | None = None):
        super().__init__()
        self.cfg = cfg
        self.reference_encoder = mlp(cfg.reference_dim, cfg.encoder_hidden, cfg.token_dim)
        self.quantizer = FiniteScalarQuantizer(cfg.token_dim, cfg.fsq_levels)
        self.dynamic_decoder = mlp(
            cfg.token_dim + cfg.proprio_dim,
            cfg.controller_hidden,
            cfg.dof,
        )
        self.kinematic_decoder = mlp(cfg.token_dim, cfg.recon_hidden, cfg.reference_dim)
        self.log_std = nn.Parameter(torch.full((cfg.dof,), float(torch.log(torch.tensor(cfg.init_action_std)))))
        self.goal_encoder = SparseGoalEncoder(goal_cfg, cfg.token_dim) if goal_cfg is not None else None

    def encode_reference(self, future_reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [B, F, 2*dof] or [B, reference_dim]
        x = future_reference.flatten(1)
        latent = self.reference_encoder(x)
        token, indices = self.quantizer(latent)
        return token, indices, latent

    def decode_token(self, proprio_history: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        return self.dynamic_decoder(torch.cat([token, proprio_history], dim=-1))

    def reconstruct_token(self, token: torch.Tensor) -> torch.Tensor:
        return self.kinematic_decoder(token)

    def project_external_token(self, token: torch.Tensor) -> torch.Tensor:
        return self.quantizer.project(token)

    def forward(
        self,
        proprio_history: torch.Tensor,
        future_reference: torch.Tensor,
        goal_targets: torch.Tensor | None = None,
        goal_masks: torch.Tensor | None = None,
        *,
        goal_residual_scale: float = 1.0,
    ) -> SonicOutput:
        token, indices, _ = self.encode_reference(future_reference)
        if self.goal_encoder is not None and goal_targets is not None and goal_masks is not None:
            goal_residual = self.goal_encoder(goal_targets, goal_masks)
            any_goal = (goal_masks.sum(dim=-1, keepdim=True) > 0).to(token.dtype)
            token = token + goal_residual_scale * any_goal * goal_residual
        action_mean = self.decode_token(proprio_history, token)
        reconstruction = self.reconstruct_token(token)
        return SonicOutput(action_mean, token, indices, reconstruction)

    def distribution(self, action_mean: torch.Tensor) -> Normal:
        std = self.log_std.exp().clamp(self.cfg.min_action_std, self.cfg.max_action_std)
        return Normal(action_mean, std)

    @torch.no_grad()
    def act_deterministic(
        self,
        proprio_history: torch.Tensor,
        future_reference: torch.Tensor,
        goal_targets: torch.Tensor | None = None,
        goal_masks: torch.Tensor | None = None,
    ) -> SonicOutput:
        return self(proprio_history, future_reference, goal_targets, goal_masks)


class TinySonicCritic(nn.Module):
    def __init__(self, cfg: SonicTinyConfig):
        super().__init__()
        self.net = mlp(cfg.proprio_dim + cfg.reference_dim, cfg.critic_hidden, 1)

    def forward(self, proprio_history: torch.Tensor, future_reference: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([proprio_history, future_reference.flatten(1)], dim=-1)).squeeze(-1)
