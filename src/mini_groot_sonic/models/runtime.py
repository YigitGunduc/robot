from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from mini_groot_sonic.config import FlowConfig, SonicTinyConfig
from mini_groot_sonic.models.flow_policy import TinyFlowMotionPolicy
from mini_groot_sonic.models.frozen_backbones import FrozenSiglip2
from mini_groot_sonic.models.sonic_tiny import TinySonicPolicy


def load_flow_checkpoint(path: str | Path, device: str) -> tuple[TinyFlowMotionPolicy, FlowConfig]:
    ckpt = torch.load(path, map_location=device)
    cfg = FlowConfig(**ckpt["flow_cfg"])
    model = TinyFlowMotionPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_body_checkpoint(path: str | Path, cfg: SonicTinyConfig, device: str) -> TinySonicPolicy:
    ckpt = torch.load(path, map_location=device)
    model = TinySonicPolicy(cfg).to(device)
    model.load_state_dict(ckpt["policy"] if "policy" in ckpt else ckpt)
    model.eval()
    return model


class RecedingHorizonTokenController:
    """Connect a planner command to the small GR00T-style flow model and SONIC decoder.

    The upper model predicts 40 future 64D tokens. At 50 Hz body control and a
    2.5 Hz planner/action-model rate, consume 20 tokens, then replan from the new
    state. This mirrors the useful SONIC/GR00T separation without the large VLM.
    """

    def __init__(
        self,
        flow: TinyFlowMotionPolicy,
        body: TinySonicPolicy,
        backbone: FrozenSiglip2,
        *,
        replan_every: int = 20,
        quantize_external_tokens: bool = True,
    ):
        self.flow = flow
        self.body = body
        self.backbone = backbone
        self.replan_every = replan_every
        self.quantize_external_tokens = quantize_external_tokens
        self._tokens: torch.Tensor | None = None
        self._cursor = 0
        self._text_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def flow_state(obs) -> torch.Tensor:
        return torch.cat(
            [obs.joint_pos, obs.joint_vel, obs.root_quat, obs.root_linvel, obs.root_angvel],
            dim=-1,
        )

    @torch.no_grad()
    def _text(self, command: str) -> torch.Tensor:
        if command not in self._text_cache:
            self._text_cache[command] = self.backbone.encode_text([command])
        return self._text_cache[command]

    @torch.no_grad()
    def replan(self, command: str, obs, goal: torch.Tensor | None = None, vision: torch.Tensor | None = None):
        state = self.flow_state(obs)
        text = self._text(command).to(state.device, state.dtype)
        if text.shape[0] != state.shape[0]:
            text = text.expand(state.shape[0], -1)
        self._tokens = self.flow.sample(state, text, vision, goal)
        if self.quantize_external_tokens:
            self._tokens = self.body.project_external_token(self._tokens)
        self._cursor = 0

    @torch.no_grad()
    def action(self, command: str, obs, proprio_history: torch.Tensor, goal: torch.Tensor | None = None, vision=None):
        if self._tokens is None or self._cursor >= min(self.replan_every, self._tokens.shape[1]):
            self.replan(command, obs, goal, vision)
        token = self._tokens[:, self._cursor]
        self._cursor += 1
        return self.body.decode_token(proprio_history, token).clamp(-1.0, 1.0), token
