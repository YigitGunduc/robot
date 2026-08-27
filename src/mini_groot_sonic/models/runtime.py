from __future__ import annotations

from pathlib import Path

import torch

from mini_groot_sonic.checkpoint import require_current_body_control_stack
from mini_groot_sonic.config import FlowConfig, SonicTinyConfig
from mini_groot_sonic.models.flow_policy import TinyFlowMotionPolicy
from mini_groot_sonic.models.frozen_backbones import FrozenSiglip2
from mini_groot_sonic.models.sonic_tiny import TinySonicPolicy
from mini_groot_sonic.sim.math_utils import quat_rotate_inverse


def load_flow_checkpoint(path: str | Path, device: str) -> tuple[TinyFlowMotionPolicy, FlowConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = FlowConfig(**ckpt["flow_cfg"])
    model = TinyFlowMotionPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_body_checkpoint(
    path: str | Path,
    device: str,
) -> tuple[TinySonicPolicy, SonicTinyConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    require_current_body_control_stack(ckpt)
    cfg = SonicTinyConfig(**ckpt.get("sonic_cfg", {}))
    model = TinySonicPolicy(cfg).to(device)
    model.load_state_dict(ckpt.get("policy", ckpt))
    model.eval()
    return model, cfg


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
        gravity_world = torch.zeros_like(obs.root_angvel)
        gravity_world[:, 2] = -1.0
        return torch.cat(
            [
                obs.joint_pos,
                obs.joint_vel,
                quat_rotate_inverse(obs.root_quat, gravity_world),
                quat_rotate_inverse(obs.root_quat, obs.root_linvel),
                obs.root_angvel,
                obs.root_pos[:, 2:3],
            ],
            dim=-1,
        )

    @torch.no_grad()
    def _text(self, command: str) -> torch.Tensor:
        if command not in self._text_cache:
            self._text_cache[command] = self.backbone.encode_text([command])
        return self._text_cache[command]

    @torch.no_grad()
    def replan(self, command: str, obs, goal: torch.Tensor | None = None, vision: torch.Tensor | None = None):
        previous_tail = None
        if self._tokens is not None and self._cursor < self._tokens.shape[1]:
            previous_tail = self._tokens[:, self._cursor :]
        state = self.flow_state(obs)
        text = self._text(command).to(state.device, state.dtype)
        if text.shape[0] != state.shape[0]:
            text = text.expand(state.shape[0], -1)
        new_tokens = self.flow.sample(state, text, vision, goal)
        if previous_tail is not None and previous_tail.shape[0] == new_tokens.shape[0]:
            overlap = min(previous_tail.shape[1], new_tokens.shape[1], self.replan_every)
            if overlap:
                alpha = torch.linspace(
                    0.0,
                    1.0,
                    overlap + 2,
                    device=new_tokens.device,
                    dtype=new_tokens.dtype,
                )[1:-1]
                new_tokens[:, :overlap] = (
                    previous_tail[:, :overlap] * (1.0 - alpha[None, :, None])
                    + new_tokens[:, :overlap] * alpha[None, :, None]
                )
        self._tokens = new_tokens
        if self.quantize_external_tokens:
            self._tokens = self.body.project_external_token(self._tokens)
        self._cursor = 0

    @torch.no_grad()
    def action(self, command: str, obs, proprio_history: torch.Tensor, goal: torch.Tensor | None = None, vision=None):
        if self._tokens is None or self._cursor >= min(self.replan_every, self._tokens.shape[1]):
            self.replan(command, obs, goal, vision)
        token = self._tokens[:, self._cursor]
        self._cursor += 1
        return torch.tanh(self.body.decode_token(proprio_history, token)), token
