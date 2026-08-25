from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_groot_sonic.config import RewardConfig
from mini_groot_sonic.sim.math_utils import quat_conjugate, quat_distance_angle, quat_mul, quat_rotate_inverse


def exp_reward(error_sq: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-error_sq / (std * std))


def mean_sq(x: torch.Tensor, dim=None) -> torch.Tensor:
    return x.square().mean(dim=dim)


@dataclass
class RewardOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    done_tracking: torch.Tensor


class SonicStyleReward:
    """Compact implementation of the published SONIC tracking reward composition."""

    def __init__(self, cfg: RewardConfig, body_names: list[str], keypoint_names: list[str]):
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

    def __call__(
        self,
        obs,
        ref: dict[str, torch.Tensor],
        action: torch.Tensor,
        previous_action: torch.Tensor,
        joint_low: torch.Tensor,
        joint_high: torch.Tensor,
        previous_body_linvel: torch.Tensor | None,
        dt: float,
        undesired_contact_count: torch.Tensor | None = None,
    ) -> RewardOutput:
        cfg = self.cfg
        dev = action.device
        kidx = self.keypoint_indices.to(dev)

        anchor_pos_err_sq = (obs.root_pos - ref["root_pos"]).square().sum(-1)
        anchor_ori_err = quat_distance_angle(obs.root_quat, ref["root_quat"])

        cur_rel_pos = obs.body_pos - obs.root_pos[:, None, :]
        ref_rel_pos = ref["body_pos"] - ref["root_pos"][:, None, :]
        rel_pos_err_sq = (cur_rel_pos - ref_rel_pos).square().sum(-1).mean(-1)

        cur_root_inv = quat_conjugate(obs.root_quat)[:, None, :].expand_as(obs.body_quat)
        ref_root_inv = quat_conjugate(ref["root_quat"])[:, None, :].expand_as(ref["body_quat"])
        cur_rel_q = quat_mul(cur_root_inv, obs.body_quat)
        ref_rel_q = quat_mul(ref_root_inv, ref["body_quat"])
        rel_ori_err = quat_distance_angle(cur_rel_q, ref_rel_q)
        rel_ori_err_sq = rel_ori_err.square().mean(-1)

        linvel_err_sq = (obs.body_linvel - ref["body_linvel"]).square().sum(-1).mean(-1)
        angvel_err_sq = (obs.body_angvel - ref["body_angvel"]).square().sum(-1).mean(-1)

        # 5-point local positions (head, wrists, ankles), root-frame coordinates.
        cur_k_world = obs.body_pos[:, kidx] - obs.root_pos[:, None]
        ref_k_world = ref["body_pos"][:, kidx] - ref["root_pos"][:, None]
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
            "action_rate": cfg.action_rate_weight * (action - previous_action).square().mean(-1),
        }

        lower_violation = (joint_low - obs.joint_pos).clamp_min(0)
        upper_violation = (obs.joint_pos - joint_high).clamp_min(0)
        terms["joint_limit"] = cfg.joint_limit_weight * (lower_violation + upper_violation).square().sum(-1)

        if undesired_contact_count is None:
            undesired_contact_count = torch.zeros_like(anchor_pos_err_sq)
        terms["undesired_contacts"] = cfg.undesired_contact_weight * undesired_contact_count

        hwidx = kidx[self.head_wrist_indices.to(dev)] if len(self.head_wrist_indices) else kidx[:0]
        if hwidx.numel():
            shake = (obs.body_angvel[:, hwidx].norm(dim=-1) - 1.5).clamp_min(0).square().mean(-1)
        else:
            shake = torch.zeros_like(anchor_pos_err_sq)
        terms["anti_shake"] = cfg.anti_shake_weight * shake

        if previous_body_linvel is not None and len(self.foot_indices):
            fidx = kidx[self.foot_indices.to(dev)]
            feet_acc = (obs.body_linvel[:, fidx] - previous_body_linvel[:, fidx]) / dt
            feet_acc_term = feet_acc.square().sum(-1).mean(-1)
        else:
            feet_acc_term = torch.zeros_like(anchor_pos_err_sq)
        terms["feet_acc"] = cfg.feet_acc_weight * feet_acc_term

        total = torch.stack(list(terms.values()), dim=0).sum(0)

        key_err = key_err_sq.sqrt()
        # Foot-specific threshold if foot keypoints are configured.
        foot_err = torch.zeros_like(key_err)
        if len(self.foot_indices):
            foot_local = (cur_k[:, self.foot_indices.to(dev)] - ref_k[:, self.foot_indices.to(dev)]).norm(dim=-1)
            foot_err = foot_local.max(-1).values
        done = (
            (anchor_pos_err_sq.sqrt() > cfg.terminate_anchor_pos)
            | (anchor_ori_err > cfg.terminate_anchor_ori)
            | (key_err > cfg.terminate_ee_pos)
            | (foot_err > cfg.terminate_foot_pos)
        )
        return RewardOutput(total, terms, done)
