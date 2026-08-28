from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionField:
    name: str
    start: int
    end: int

    @property
    def dim(self) -> int:
        return self.end - self.start


class ActionLayout:
    """GR00T-style packed action vector with explicit per-field masks.

    V1 can be SONIC-token-only (64 D). V2 can append task-space or hand fields without changing
    the low-level 29-DOF body policy. Missing fields are masked from flow-matching loss.
    """
    def __init__(self, token_dim: int = 64, include_task_space_hands: bool = False, include_grippers: bool = False):
        fields = []
        cursor = 0
        fields.append(ActionField("motion_token", cursor, cursor + token_dim)); cursor += token_dim
        if include_task_space_hands:
            # xyz + continuous rotation-6D per hand; task-space targets are checked by IK before execution.
            for side in ["left", "right"]:
                fields.append(ActionField(f"{side}_ee_target", cursor, cursor + 9)); cursor += 9
        if include_grippers:
            for side in ["left", "right"]:
                fields.append(ActionField(f"{side}_gripper", cursor, cursor + 1)); cursor += 1
        self.fields = {f.name: f for f in fields}
        self.dim = cursor

    def pack(self, batch_shape: tuple[int, ...], device: torch.device, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        action = torch.zeros(*batch_shape, self.dim, device=device)
        mask = torch.zeros_like(action, dtype=torch.bool)
        for name, v in values.items():
            f = self.fields[name]
            if v.shape != (*batch_shape, f.dim):
                raise ValueError(f"{name}: expected {(*batch_shape, f.dim)}, got {tuple(v.shape)}")
            action[..., f.start:f.end] = v
            mask[..., f.start:f.end] = True
        return action, mask

    def field(self, action: torch.Tensor, name: str) -> torch.Tensor:
        f = self.fields[name]
        return action[..., f.start:f.end]
