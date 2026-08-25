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
        self.state_norm = nn.LayerNorm(cfg.state_dim)
        self.text_norm = nn.LayerNorm(cfg.text_dim)
        self.vision_norm = nn.LayerNorm(cfg.vision_dim)
        self.goal_norm = nn.LayerNorm(cfg.goal_dim) if cfg.goal_dim > 0 else None
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

    def _context_tokens(
        self,
        state: torch.Tensor,
        text: torch.Tensor,
        vision: torch.Tensor | None,
        goal: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        toks = [
            self.state_proj(self.state_norm(state))[:, None],
            self.text_proj(self.text_norm(text))[:, None],
        ]
        if vision is not None:
            toks.append(self.vision_proj(self.vision_norm(vision))[:, None])
        if self.goal_proj is not None and goal is not None:
            toks.append(self.goal_proj(self.goal_norm(goal))[:, None])
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
        b, h, _ = noisy_actions.shape
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
        t = beta.sample((b,)).to(actions.device, actions.dtype)
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
