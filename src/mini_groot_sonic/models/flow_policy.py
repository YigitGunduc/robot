from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mini_groot_sonic.config import FlowConfig
from mini_groot_sonic.models.common import SinusoidalTimeEmbedding


@dataclass
class FlowLossOutput:
    loss: torch.Tensor
    predicted_velocity: torch.Tensor
    target_velocity: torch.Tensor
    noisy_actions: torch.Tensor
    t: torch.Tensor


class TinyFlowMotionPolicy(nn.Module):
    """Small GR00T-N1.7-style flow-matching action head.

    It predicts a future chunk of SONIC-compatible 64D motion tokens. Frozen
    language/vision embeddings are treated as context; this model learns only
    robotics fusion and the action vector field.
    """

    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim
        self.action_proj = nn.Linear(cfg.action_dim, d)
        self.text_norm = nn.LayerNorm(cfg.text_dim)
        self.vision_norm = nn.LayerNorm(cfg.vision_dim)
        self.state_proj = nn.Linear(cfg.state_dim, d)
        self.text_proj = nn.Linear(cfg.text_dim, d)
        self.vision_proj = nn.Linear(cfg.vision_dim, d)
        self.goal_proj = nn.Linear(cfg.goal_dim, d) if cfg.goal_dim > 0 else None
        self.time_embed = SinusoidalTimeEmbedding(d)
        self.time_proj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, cfg.action_horizon + 4, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.num_heads,
            dim_feedforward=4 * d,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.out = nn.Linear(d, cfg.action_dim)
        self.register_buffer("state_mean", torch.zeros(cfg.state_dim))
        self.register_buffer("state_std", torch.ones(cfg.state_dim))
        self.register_buffer("goal_mean", torch.zeros(cfg.goal_dim))
        self.register_buffer("goal_std", torch.ones(cfg.goal_dim))

    @torch.no_grad()
    def set_normalization_stats(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        goal_mean: torch.Tensor,
        goal_std: torch.Tensor,
    ) -> None:
        self.state_mean.copy_(state_mean)
        self.state_std.copy_(state_std.clamp_min(1e-4))
        self.goal_mean.copy_(goal_mean)
        self.goal_std.copy_(goal_std.clamp_min(1e-4))

    def _normalize_goal(self, goal: torch.Tensor) -> torch.Tensor:
        normalized = (goal - self.goal_mean) / self.goal_std.clamp_min(1e-4)
        if self.cfg.goal_dim % self.cfg.goal_slot_dim:
            return normalized
        raw_slots = goal.reshape(goal.shape[0], -1, self.cfg.goal_slot_dim)
        slots = normalized.reshape_as(raw_slots)
        mask = raw_slots[..., -1:]
        slots = torch.cat([slots[..., :-1] * mask, mask], dim=-1)
        return slots.flatten(1)

    def _context_tokens(
        self,
        state: torch.Tensor,
        text: torch.Tensor,
        vision: torch.Tensor | None,
        goal: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        normalized_state = (state - self.state_mean) / self.state_std.clamp_min(1e-4)
        state_token = self.state_proj(normalized_state)
        if self.training and self.cfg.state_dropout_prob > 0:
            keep = (torch.rand(state.shape[0], 1, device=state.device) >= self.cfg.state_dropout_prob).to(state.dtype)
            state_token = state_token * keep
        text_token = self.text_proj(self.text_norm(text))
        if self.training and self.cfg.condition_dropout_prob > 0:
            text_keep = (
                torch.rand(text.shape[0], 1, device=text.device) >= self.cfg.condition_dropout_prob
            ).to(text.dtype)
            text_token = text_token * text_keep
        toks = [
            state_token[:, None],
            text_token[:, None],
        ]
        if vision is not None:
            toks.append(self.vision_proj(self.vision_norm(vision))[:, None])
        if self.goal_proj is not None:
            if goal is None:
                goal = torch.zeros(state.shape[0], self.cfg.goal_dim, device=state.device, dtype=state.dtype)
            normalized_goal = self._normalize_goal(goal)
            goal_token = self.goal_proj(normalized_goal)
            if self.training and self.cfg.condition_dropout_prob > 0:
                keep = (
                    torch.rand(goal.shape[0], 1, device=goal.device) >= self.cfg.condition_dropout_prob
                ).to(goal.dtype)
                goal_token = goal_token * keep
            toks.append(goal_token[:, None])
        return toks

    def predict_velocity(
        self,
        noisy_actions: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor,
        text: torch.Tensor,
        vision: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, h, _ = noisy_actions.shape
        action = self.action_proj(noisy_actions)
        time = self.time_proj(self.time_embed(t))[:, None, :]
        action = action + time
        context = self._context_tokens(state, text, vision, goal)
        nctx = len(context)
        x = torch.cat(context + [action], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        x = self.transformer(x)
        return self.out(x[:, nctx : nctx + h])

    def flow_matching_loss(
        self,
        actions: torch.Tensor,
        state: torch.Tensor,
        text: torch.Tensor,
        vision: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> FlowLossOutput:
        b = actions.shape[0]
        beta = torch.distributions.Beta(self.cfg.beta_alpha, self.cfg.beta_beta)
        t = (1.0 - beta.sample((b,))).to(actions.device, actions.dtype) * self.cfg.noise_scale
        noise = torch.randn_like(actions)
        tt = t[:, None, None]
        noisy = (1.0 - tt) * noise + tt * actions
        target_v = actions - noise
        pred_v = self.predict_velocity(noisy, t, state, text, vision, goal)
        sq = (pred_v - target_v).square()
        if valid_mask is not None:
            mask = valid_mask[..., None].to(sq.dtype)
            loss = (sq * mask).sum() / mask.sum().clamp_min(1.0) / sq.shape[-1]
        else:
            loss = sq.mean()
        return FlowLossOutput(loss, pred_v, target_v, noisy, t)

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        text: torch.Tensor,
        vision: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
        *,
        steps: int | None = None,
    ) -> torch.Tensor:
        steps = steps or self.cfg.inference_steps
        b = state.shape[0]
        z = torch.randn(
            b,
            self.cfg.action_horizon,
            self.cfg.action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        dt = 1.0 / steps
        # Current code convention: start at noise (t=0), integrate toward data (t=1).
        for i in range(steps):
            t = torch.full((b,), i / steps, device=z.device, dtype=z.dtype)
            v = self.predict_velocity(z, t, state, text, vision, goal)
            z = z + dt * v
        return z
