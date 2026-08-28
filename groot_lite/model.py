from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1)))
    phase = t[:, None] * freqs[None]
    emb = torch.cat([phase.sin(), phase.cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class FrozenSiglip2Backbone(nn.Module):
    """Frozen Hugging Face SigLIP2 image/text feature extractor.

    Default checkpoint is deliberately small relative to GR00T N1.x. It can operate text-only for
    BONES pretraining; visual features are added later when camera demonstrations exist.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        super().__init__()
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError("Install the HF extra: pip install -e '.[hf]'") from exc
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()
        cfg = self.model.config
        self.image_dim = int(getattr(getattr(cfg, "vision_config", cfg), "hidden_size", 768))
        self.text_dim = int(getattr(getattr(cfg, "text_config", cfg), "hidden_size", 768))

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()  # backbone must remain frozen/eval even when action head trains
        return self

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        batch = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        if hasattr(self.model, "get_text_features"):
            feat = self.model.get_text_features(**batch)
        else:
            out = self.model.text_model(**batch)
            feat = getattr(out, "pooler_output", out.last_hidden_state[:, 0])
        return feat.float()

    @torch.no_grad()
    def encode_images(self, images, device: torch.device) -> torch.Tensor:
        batch = self.processor(images=images, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        if hasattr(self.model, "get_image_features"):
            feat = self.model.get_image_features(**batch)
        else:
            out = self.model.vision_model(**batch)
            feat = getattr(out, "pooler_output", out.last_hidden_state[:, 0])
        return feat.float()


class ConditionProjector(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.text = nn.Linear(text_dim, hidden_dim)
        self.image = nn.Linear(image_dim, hidden_dim)
        self.state = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text: torch.Tensor, state: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        x = self.text(text) + self.state(state)
        if image is not None:
            x = x + self.image(image)
        return self.norm(x)


class FlowActionTransformer(nn.Module):
    """Compact GR00T-like flow-matching action head.

    It predicts SONIC universal-token trajectories. Optional extra action dimensions (hands,
    grippers, task-space targets) can be appended and masked per training example.
    """

    def __init__(self, action_dim: int = 64, horizon: int = 16, hidden_dim: int = 512, layers: int = 8, heads: int = 8, ff_mult: int = 4, condition_dim: int = 512):
        super().__init__()
        self.action_dim, self.horizon, self.hidden_dim = action_dim, horizon, hidden_dim
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cond_in = nn.Linear(condition_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, horizon + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * ff_mult, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, noisy_actions: torch.Tensor, flow_time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if noisy_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"expected actions [B,{self.horizon},{self.action_dim}], got {tuple(noisy_actions.shape)}")
        t = self.time_mlp(sinusoidal_time_embedding(flow_time, self.hidden_dim))
        cond = self.cond_in(condition) + t
        tokens = self.action_in(noisy_actions)
        x = torch.cat([cond[:, None], tokens], dim=1) + self.pos
        x = self.transformer(x)
        return self.action_out(self.norm(x[:, 1:]))

    def flow_matching_loss(self, clean_actions: torch.Tensor, condition: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = clean_actions.shape[0]
        eps = torch.randn_like(clean_actions)
        tau = torch.rand(b, device=clean_actions.device, dtype=clean_actions.dtype)
        noisy = tau[:, None, None] * clean_actions + (1.0 - tau[:, None, None]) * eps
        # With x_tau = tau*x + (1-tau)*eps, dx/dtau = x-eps.
        target_v = clean_actions - eps
        pred_v = self(noisy, tau, condition)
        loss = (pred_v - target_v).square()
        if action_mask is None:
            return loss.mean()
        mask = action_mask.to(loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, steps: int = 4, initial_noise: torch.Tensor | None = None, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = condition.shape[0]
        x = torch.randn(b, self.horizon, self.action_dim, device=condition.device, dtype=condition.dtype) if initial_noise is None else initial_noise
        dt = 1.0 / float(steps)
        for i in range(steps):
            tau = torch.full((b,), i / float(steps), device=x.device, dtype=x.dtype)
            x = x + dt * self(x, tau, condition)
        if action_mask is not None:
            x = x * action_mask.to(x.dtype)
        return x


@dataclass
class GrootLiteOutput:
    condition: torch.Tensor
    loss: torch.Tensor | None = None
    actions: torch.Tensor | None = None


class GrootLitePolicy(nn.Module):
    def __init__(self, backbone: FrozenSiglip2Backbone, state_dim: int = 32, action_dim: int = 64, horizon: int = 16, hidden_dim: int = 512, layers: int = 8, heads: int = 8):
        super().__init__()
        self.backbone = backbone
        self.condition = ConditionProjector(backbone.text_dim, backbone.image_dim, state_dim, hidden_dim)
        self.action_head = FlowActionTransformer(action_dim, horizon, hidden_dim, layers, heads, condition_dim=hidden_dim)

    def make_condition(self, text_features: torch.Tensor, state: torch.Tensor, image_features: torch.Tensor | None = None):
        return self.condition(text_features, state, image_features)

    def loss(self, text_features: torch.Tensor, state: torch.Tensor, actions: torch.Tensor, image_features: torch.Tensor | None = None, action_mask: torch.Tensor | None = None):
        cond = self.make_condition(text_features, state, image_features)
        return self.action_head.flow_matching_loss(actions, cond, action_mask)
