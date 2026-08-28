from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from gear_sonic_mjx.config import ModelConfig, PPOConfig
from .base_module import MLP
from .universal_token_modules import UniversalTokenModule, UniversalTokenOutput


class Actor(nn.Module):
    def __init__(self, model_cfg: ModelConfig, ppo_cfg: PPOConfig, num_future_frames: int = 10, history_length: int = 10):
        super().__init__()
        self.backbone = UniversalTokenModule(model_cfg, num_future_frames, history_length)
        self.log_std = nn.Parameter(torch.full((model_cfg.dof,), math.log(ppo_cfg.init_noise_std)))
        self.std_min = ppo_cfg.std_clamp_min
        self.std_max = ppo_cfg.std_clamp_max

    def _std(self) -> torch.Tensor:
        return self.log_std.exp().clamp(self.std_min, self.std_max)

    def distribution(self, action_mean: torch.Tensor) -> Normal:
        return Normal(action_mean, self._std().expand_as(action_mean))

    def forward(self, encoder_obs: torch.Tensor, proprio: torch.Tensor, deterministic: bool = False, compute_aux_loss: bool = True):
        out = self.backbone(encoder_obs, proprio, compute_aux_loss=compute_aux_loss)
        dist = self.distribution(out.action_mean)
        action = out.action_mean if deterministic else dist.rsample()
        return action, dist, out


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
