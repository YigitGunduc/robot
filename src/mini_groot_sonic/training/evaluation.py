from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from mini_groot_sonic.config import SimConfig, SonicTinyConfig
from mini_groot_sonic.data.motion_bank import MotionBank
from mini_groot_sonic.models.sonic_tiny import TinySonicPolicy
from mini_groot_sonic.sim.math_utils import quat_distance_angle
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv


@torch.no_grad()
def evaluate_body_controller(
    paths: list[str | Path],
    sim_cfg: SimConfig,
    sonic_cfg: SonicTinyConfig,
    policy: TinySonicPolicy,
    *,
    max_motions: int = 16,
) -> dict[str, float]:
    """Run deterministic, non-randomized held-out motion tracking."""
    selected = list(paths[:max_motions])
    if not selected:
        return {}
    eval_sim = replace(sim_cfg, enable_randomization=False)
    bank = MotionBank(selected, sonic_cfg, eval_sim.device)
    env = MJWarpG1VecEnv(eval_sim, sonic_cfg, len(selected))
    device = torch.device(eval_sim.device)
    motion_ids = torch.arange(len(selected), device=device)
    frame_ids = torch.zeros(len(selected), dtype=torch.long, device=device)
    ref = bank.current_reference(motion_ids, frame_ids)
    obs = env.reset(
        ref["root_pos"],
        ref["root_quat"],
        ref["joint_pos"],
        ref["root_linvel"],
        ref["root_angvel"],
        ref["joint_vel"],
    )
    active = torch.ones(len(selected), dtype=torch.bool, device=device)
    failed = torch.zeros_like(active)
    previous_action = torch.zeros(len(selected), sonic_cfg.dof, device=device)
    sums = {name: torch.zeros((), device=device) for name in (
        "root_position_error",
        "root_orientation_error",
        "mpjpe",
        "joint_position_error",
        "action_rate",
        "undesired_contacts",
        "mean_abs_actuator_force",
    )}
    samples = torch.zeros((), device=device)
    max_steps = int(bank.lengths.max().item())
    was_training = policy.training
    policy.eval()
    for _ in range(max_steps):
        if not active.any():
            break
        max_start = (
            bank.lengths[motion_ids]
            - (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride
            - 1
        ).clamp_min(0)
        future_ids = torch.minimum(frame_ids, max_start)
        future = bank.future_reference(motion_ids, future_ids)
        out = policy(env.proprio_history(), future)
        action = torch.tanh(out.action_mean)
        action = torch.where(active[:, None], action, torch.zeros_like(action))
        obs = env.step(action)
        frame_ids = frame_ids + active.long()
        safe_frame = torch.minimum(frame_ids, bank.lengths[motion_ids] - 1)
        ref = bank.current_reference(motion_ids, safe_frame)
        mask = active.float()
        root_error = (obs.root_pos - ref["root_pos"]).norm(dim=-1)
        ori_error = quat_distance_angle(obs.root_quat, ref["root_quat"])
        mpjpe = (obs.body_pos - ref["body_pos"]).norm(dim=-1).mean(-1)
        joint_error = (obs.joint_pos - ref["joint_pos"]).abs().mean(-1)
        action_rate = (action - previous_action).square().sum(-1)
        contacts = env.undesired_contact_count()
        force = env.mean_abs_actuator_force()
        for name, value in (
            ("root_position_error", root_error),
            ("root_orientation_error", ori_error),
            ("mpjpe", mpjpe),
            ("joint_position_error", joint_error),
            ("action_rate", action_rate),
            ("undesired_contacts", contacts),
            ("mean_abs_actuator_force", force),
        ):
            sums[name] += (value * mask).sum()
        samples += mask.sum()
        tracking_failure = (root_error > 0.15) | (ori_error > 0.2) | (mpjpe > 0.15)
        failed |= active & tracking_failure
        motion_end = frame_ids + (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride + 1 >= bank.lengths[motion_ids]
        active &= ~(tracking_failure | motion_end)
        previous_action = action
    policy.train(was_training)
    denom = samples.clamp_min(1)
    metrics = {name: float(value / denom) for name, value in sums.items()}
    metrics["success_rate"] = float((~failed).float().mean())
    metrics["evaluated_motions"] = float(len(selected))
    return metrics
