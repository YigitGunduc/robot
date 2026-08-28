from __future__ import annotations

import torch

from gear_sonic_mjx.g1_parameters import ACTION_SCALE_MJ, DEFAULT_ANGLES_MJ


def joint_position_target(
    action: torch.Tensor,
    default_angles: torch.Tensor | None = None,
    action_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    defaults = (
        DEFAULT_ANGLES_MJ.to(action)
        if default_angles is None
        else default_angles.to(action)
    )
    scale = (
        ACTION_SCALE_MJ.to(action) if action_scale is None else action_scale.to(action)
    )
    return defaults + scale * action


def pd_torque(
    target_q: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    target_qd: torch.Tensor | None = None,
    feedforward: torch.Tensor | None = None,
) -> torch.Tensor:
    target_qd = torch.zeros_like(qd) if target_qd is None else target_qd
    ff = torch.zeros_like(q) if feedforward is None else feedforward
    return ff + kp.to(q) * (target_q - q) + kd.to(q) * (target_qd - qd)
