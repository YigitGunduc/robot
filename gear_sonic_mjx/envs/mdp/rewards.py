from __future__ import annotations

from dataclasses import dataclass

import torch

from gear_sonic_mjx.config import RewardConfig
from gear_sonic_mjx.math_utils import quat_angle_error, quat_mul_wxyz, quat_conjugate_wxyz, rotate_inverse_wxyz


def gaussian_reward(error: torch.Tensor, std: float, reduce_dim: int | tuple[int, ...] | None = -1) -> torch.Tensor:
    if reduce_dim is not None:
        error = torch.sum(error * error, dim=reduce_dim)
    else:
        error = error * error
    return torch.exp(-error / max(std * std, 1e-8))


@dataclass
class TrackingState:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    body_pos: torch.Tensor | None = None
    body_quat: torch.Tensor | None = None
    body_linvel: torch.Tensor | None = None
    body_angvel: torch.Tensor | None = None
    reward_points: torch.Tensor | None = None
    action: torch.Tensor | None = None
    prev_action: torch.Tensor | None = None
    joint_pos: torch.Tensor | None = None
    joint_lower: torch.Tensor | None = None
    joint_upper: torch.Tensor | None = None
    undesired_contact: torch.Tensor | None = None
    anti_shake_angvel: torch.Tensor | None = None
    feet_acc: torch.Tensor | None = None


@dataclass
class TrackingReference:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    body_pos: torch.Tensor | None = None
    body_quat: torch.Tensor | None = None
    body_linvel: torch.Tensor | None = None
    body_angvel: torch.Tensor | None = None
    reward_points: torch.Tensor | None = None


class SonicReward:
    """SONIC reward composition with public weights and configurable kernels.

    Exact kernel std values vary across term YAMLs/releases, so they are arguments instead of
    silently guessed constants. The weights match the released base_5point_local_feet_acc config.
    """

    def __init__(self, cfg: RewardConfig, stds: dict[str, float] | None = None):
        self.cfg = cfg
        self.stds = {
            "root_pos": 0.2,
            "root_ori": 0.3,
            "body_pos": 0.15,
            "body_ori": 0.3,
            "linvel": 0.5,
            "angvel": 0.5,
            "fivepoint": 0.15,
            **(stds or {}),
        }

    def __call__(self, s: TrackingState, r: TrackingReference) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        terms: dict[str, torch.Tensor] = {}
        terms["tracking_anchor_pos"] = gaussian_reward(s.root_pos - r.root_pos, self.stds["root_pos"])
        ori_err = quat_angle_error(s.root_quat, r.root_quat)
        terms["tracking_anchor_ori"] = torch.exp(-(ori_err**2) / self.stds["root_ori"]**2)

        if s.body_pos is not None and r.body_pos is not None:
            # NVIDIA's command manager has already heading/translation-aligned the reference bodies.
            per_body = torch.sum((s.body_pos - r.body_pos) ** 2, dim=-1)
            terms["tracking_relative_body_pos"] = torch.exp(-per_body.mean(dim=-1) / self.stds["body_pos"]**2)
        else:
            terms["tracking_relative_body_pos"] = torch.zeros_like(ori_err)

        if s.body_quat is not None and r.body_quat is not None:
            # Reference orientations are already heading-aligned by the command manager.
            e = quat_angle_error(s.body_quat, r.body_quat)
            terms["tracking_relative_body_ori"] = torch.exp(-torch.mean(e*e, dim=-1) / self.stds["body_ori"]**2)
        else:
            terms["tracking_relative_body_ori"] = torch.zeros_like(ori_err)

        if s.body_linvel is not None and r.body_linvel is not None:
            per_body = torch.sum((s.body_linvel - r.body_linvel) ** 2, dim=-1)
            terms["tracking_body_linvel"] = torch.exp(-per_body.mean(dim=-1) / self.stds["linvel"]**2)
        else:
            terms["tracking_body_linvel"] = torch.zeros_like(ori_err)

        if s.body_angvel is not None and r.body_angvel is not None:
            per_body = torch.sum((s.body_angvel - r.body_angvel) ** 2, dim=-1)
            terms["tracking_body_angvel"] = torch.exp(-per_body.mean(dim=-1) / self.stds["angvel"]**2)
        else:
            terms["tracking_body_angvel"] = torch.zeros_like(ori_err)

        if s.reward_points is not None and r.reward_points is not None:
            sp = s.reward_points - s.root_pos[:, None]
            rp = r.reward_points - r.root_pos[:, None]
            sq = s.root_quat[:, None].expand(sp.shape[:-1] + (4,))
            rq = r.root_quat[:, None].expand(rp.shape[:-1] + (4,))
            sp_local = rotate_inverse_wxyz(sq, sp)
            rp_local = rotate_inverse_wxyz(rq, rp)
            per_point = torch.sum((sp_local - rp_local) ** 2, dim=-1)
            terms["tracking_vr_5point_local"] = torch.exp(-per_point.mean(dim=-1) / self.stds["fivepoint"]**2)
        else:
            terms["tracking_vr_5point_local"] = torch.zeros_like(ori_err)

        terms["action_rate_l2"] = torch.zeros_like(ori_err) if s.action is None or s.prev_action is None else torch.sum((s.action-s.prev_action)**2, dim=-1)
        if s.joint_pos is not None and s.joint_lower is not None and s.joint_upper is not None:
            below = (s.joint_lower - s.joint_pos).clamp_min(0)
            above = (s.joint_pos - s.joint_upper).clamp_min(0)
            terms["joint_limit"] = torch.sum(below + above, dim=-1)
        else:
            terms["joint_limit"] = torch.zeros_like(ori_err)
        terms["undesired_contacts"] = torch.zeros_like(ori_err) if s.undesired_contact is None else s.undesired_contact.float()
        if s.anti_shake_angvel is None:
            terms["anti_shake_ang_vel"] = torch.zeros_like(ori_err)
        else:
            # NVIDIA applies a 1.5 rad/s deadzone then mean-squares the excess per body.
            speed = torch.linalg.vector_norm(s.anti_shake_angvel, dim=-1)
            terms["anti_shake_ang_vel"] = torch.relu(speed - 1.5).square().mean(dim=-1)
        terms["feet_acc"] = torch.zeros_like(ori_err) if s.feet_acc is None else torch.sum(s.feet_acc**2, dim=(-1,-2) if s.feet_acc.ndim == 3 else -1)

        total = (
            self.cfg.tracking_anchor_pos * terms["tracking_anchor_pos"]
            + self.cfg.tracking_anchor_ori * terms["tracking_anchor_ori"]
            + self.cfg.tracking_relative_body_pos * terms["tracking_relative_body_pos"]
            + self.cfg.tracking_relative_body_ori * terms["tracking_relative_body_ori"]
            + self.cfg.tracking_body_linvel * terms["tracking_body_linvel"]
            + self.cfg.tracking_body_angvel * terms["tracking_body_angvel"]
            + self.cfg.tracking_vr_5point_local * terms["tracking_vr_5point_local"]
            + self.cfg.action_rate_l2 * terms["action_rate_l2"]
            + self.cfg.joint_limit * terms["joint_limit"]
            + self.cfg.undesired_contacts * terms["undesired_contacts"]
            + self.cfg.anti_shake_ang_vel * terms["anti_shake_ang_vel"]
            + self.cfg.feet_acc * terms["feet_acc"]
        )
        return total, terms
