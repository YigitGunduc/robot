from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_groot_sonic.config import RewardConfig
from mini_groot_sonic.sim.math_utils import (
    quat_conjugate,
    quat_distance_angle,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)


def exp_reward(error_sq: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-error_sq / (std * std))


def mean_sq(x: torch.Tensor, dim=None) -> torch.Tensor:
    return x.square().mean(dim=dim)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    """Return the normalized yaw-only component of a wxyz quaternion."""

    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def reanchor_reference_bodies(obs, ref: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Align reference XY and heading to the robot, preserving reference height."""

    delta_heading = heading_quat(quat_mul(obs.root_quat, quat_conjugate(ref["root_quat"])))
    reference_origin = obs.root_pos.clone()
    reference_origin[:, 2] = ref["root_pos"][:, 2]
    body_offset = ref["body_pos"] - ref["root_pos"][:, None]
    ref_body_pos = reference_origin[:, None] + quat_rotate(
        delta_heading[:, None].expand_as(ref["body_quat"]), body_offset
    )
    ref_body_quat = quat_mul(
        delta_heading[:, None].expand_as(ref["body_quat"]), ref["body_quat"]
    )
    return ref_body_pos, ref_body_quat


@dataclass
class RewardOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    done_tracking: torch.Tensor


class SonicStyleReward:
    """Compact implementation of the published SONIC tracking reward composition."""

    def __init__(
        self,
        cfg: RewardConfig,
        body_names: list[str],
        keypoint_names: list[str],
        joint_names: list[str] | None = None,
    ):
        self.cfg = cfg
        self.body_names = body_names
        self.keypoint_indices = torch.tensor([body_names.index(n) for n in keypoint_names], dtype=torch.long)
        self.foot_indices = torch.tensor(
            [i for i, n in enumerate(keypoint_names) if "ankle" in n.lower() or "foot" in n.lower()],
            dtype=torch.long,
        )
        self.head_wrist_indices = torch.tensor(
            [i for i, n in enumerate(keypoint_names) if "head" in n.lower() or "wrist" in n.lower() or "hand" in n.lower()],
            dtype=torch.long,
        )
        self.ee_indices = torch.tensor(
            [
                i
                for i, n in enumerate(keypoint_names)
                if any(part in n.lower() for part in ("wrist", "hand", "ankle", "foot"))
            ],
            dtype=torch.long,
        )
        keypoint_offsets = torch.zeros(len(keypoint_names), 3)
        if not any("head" in name.lower() for name in keypoint_names):
            for index, name in enumerate(keypoint_names):
                if "torso" in name.lower():
                    keypoint_offsets[index, 2] = 0.5
                    break
        self.keypoint_offsets = keypoint_offsets
        self.ankle_joint_indices = torch.tensor(
            [i for i, n in enumerate(joint_names or []) if "ankle" in n.lower()],
            dtype=torch.long,
        )

    def __call__(
        self,
        obs,
        ref: dict[str, torch.Tensor],
        action: torch.Tensor,
        previous_action: torch.Tensor,
        joint_low: torch.Tensor,
        joint_high: torch.Tensor,
        previous_joint_vel: torch.Tensor | None,
        dt: float,
        undesired_contact_count: torch.Tensor | None = None,
    ) -> RewardOutput:
        cfg = self.cfg
        dev = action.device
        kidx = self.keypoint_indices.to(dev)

        anchor_pos_err_sq = (obs.root_pos - ref["root_pos"]).square().sum(-1)
        anchor_ori_err = quat_distance_angle(obs.root_quat, ref["root_quat"])

        ref_body_pos, ref_body_quat = reanchor_reference_bodies(obs, ref)
        rel_pos_err_sq = (obs.body_pos - ref_body_pos).square().sum(-1).mean(-1)
        rel_ori_err = quat_distance_angle(obs.body_quat, ref_body_quat)
        rel_ori_err_sq = rel_ori_err.square().mean(-1)

        linvel_err_sq = (obs.body_linvel - ref["body_linvel"]).square().sum(-1).mean(-1)
        angvel_err_sq = (obs.body_angvel - ref["body_angvel"]).square().sum(-1).mean(-1)

        # 5-point local positions (head, wrists, ankles), root-frame coordinates.
        offsets = self.keypoint_offsets.to(dev, dtype=obs.body_pos.dtype)
        cur_points = obs.body_pos[:, kidx] + quat_rotate(
            obs.body_quat[:, kidx], offsets[None].expand(obs.body_pos.shape[0], -1, -1)
        )
        ref_points = ref["body_pos"][:, kidx] + quat_rotate(
            ref["body_quat"][:, kidx], offsets[None].expand(obs.body_pos.shape[0], -1, -1)
        )
        cur_k_world = cur_points - obs.root_pos[:, None]
        ref_k_world = ref_points - ref["root_pos"][:, None]
        cur_root = obs.root_quat[:, None].expand(-1, len(kidx), -1)
        ref_root = ref["root_quat"][:, None].expand(-1, len(kidx), -1)
        cur_k = quat_rotate_inverse(cur_root, cur_k_world)
        ref_k = quat_rotate_inverse(ref_root, ref_k_world)
        key_err_sq = (cur_k - ref_k).square().sum(-1).mean(-1)

        terms = {
            "anchor_pos": cfg.anchor_pos_weight * exp_reward(anchor_pos_err_sq, cfg.anchor_pos_std),
            "anchor_ori": cfg.anchor_ori_weight * exp_reward(anchor_ori_err.square(), cfg.anchor_ori_std),
            "relative_body_pos": cfg.relative_body_pos_weight * exp_reward(rel_pos_err_sq, cfg.relative_body_pos_std),
            "relative_body_ori": cfg.relative_body_ori_weight * exp_reward(rel_ori_err_sq, cfg.relative_body_ori_std),
            "body_linvel": cfg.body_linvel_weight * exp_reward(linvel_err_sq, cfg.body_linvel_std),
            "body_angvel": cfg.body_angvel_weight * exp_reward(angvel_err_sq, cfg.body_angvel_std),
            "keypoint": cfg.keypoint_weight * exp_reward(key_err_sq, cfg.keypoint_std),
            "action_rate": cfg.action_rate_weight * (action - previous_action).square().sum(-1),
        }

        lower_violation = (joint_low - obs.joint_pos).clamp_min(0)
        upper_violation = (obs.joint_pos - joint_high).clamp_min(0)
        terms["joint_limit"] = cfg.joint_limit_weight * (lower_violation + upper_violation).sum(-1)

        if undesired_contact_count is None:
            undesired_contact_count = torch.zeros_like(anchor_pos_err_sq)
        terms["undesired_contacts"] = cfg.undesired_contact_weight * undesired_contact_count

        hwidx = kidx[self.head_wrist_indices.to(dev)] if len(self.head_wrist_indices) else kidx[:0]
        if hwidx.numel():
            shake = (obs.body_angvel[:, hwidx].norm(dim=-1) - 1.5).clamp_min(0).square().mean(-1)
        else:
            shake = torch.zeros_like(anchor_pos_err_sq)
        terms["anti_shake"] = cfg.anti_shake_weight * shake

        if previous_joint_vel is not None and len(self.ankle_joint_indices):
            aidx = self.ankle_joint_indices.to(dev)
            ankle_acc = (obs.joint_vel[:, aidx] - previous_joint_vel[:, aidx]) / dt
            feet_acc_term = ankle_acc.square().sum(-1)
        else:
            feet_acc_term = torch.zeros_like(anchor_pos_err_sq)
        terms["feet_acc"] = cfg.feet_acc_weight * feet_acc_term

        total = torch.stack(list(terms.values()), dim=0).sum(0)

        # SONIC terminates on anchor/end-effector height and full foot position,
        # while local keypoint MPJPE remains a dense reward rather than a failure.
        height_threshold = torch.full_like(anchor_ori_err, cfg.terminate_anchor_pos)
        low_motion = ref["root_pos"][:, 2] < cfg.low_motion_root_height
        height_threshold = torch.where(
            low_motion,
            torch.full_like(height_threshold, cfg.terminate_low_motion_height),
            height_threshold,
        )
        anchor_height_err = (obs.root_pos[:, 2] - ref["root_pos"][:, 2]).abs()
        ee_height_err = torch.zeros_like(anchor_height_err)
        if len(self.ee_indices):
            eidx = self.ee_indices.to(dev)
            ee_height_err = (cur_points[:, eidx, 2] - ref_points[:, eidx, 2]).abs().max(-1).values

        foot_err = torch.zeros_like(anchor_height_err)
        if len(self.foot_indices):
            foot_body_ids = kidx[self.foot_indices.to(dev)]
            foot_err = (obs.body_pos[:, foot_body_ids] - ref_body_pos[:, foot_body_ids]).norm(dim=-1).max(-1).values
        done = (
            (anchor_height_err > height_threshold)
            | (anchor_ori_err > cfg.terminate_anchor_ori)
            | (
                ee_height_err
                > torch.where(
                    low_motion,
                    torch.full_like(height_threshold, cfg.terminate_low_motion_height),
                    torch.full_like(height_threshold, cfg.terminate_ee_pos),
                )
            )
            | (foot_err > cfg.terminate_foot_pos)
        )
        return RewardOutput(total, terms, done)
