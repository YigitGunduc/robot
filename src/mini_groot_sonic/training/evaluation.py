from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from mini_groot_sonic.config import RewardConfig, SimConfig, SonicTinyConfig
from mini_groot_sonic.data.motion_bank import MotionBank
from mini_groot_sonic.models.sonic_tiny import TinySonicPolicy
from mini_groot_sonic.sim.math_utils import quat_distance_angle
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv
from mini_groot_sonic.training.rewards import (
    SonicStyleReward,
    reanchor_reference_bodies,
)


@torch.no_grad()
def evaluate_body_controller(
    paths: list[str | Path],
    sim_cfg: SimConfig,
    sonic_cfg: SonicTinyConfig,
    policy: TinySonicPolicy,
    *,
    reward_cfg: RewardConfig | None = None,
    max_motions: int = 16,
) -> dict[str, float]:
    """Run deterministic, non-randomized held-out motion tracking."""
    selected = list(paths[:max_motions])
    if not selected:
        return {}
    eval_sim = replace(
        sim_cfg,
        enable_randomization=False,
        enable_observation_noise=False,
    )
    bank = MotionBank(selected, sonic_cfg, eval_sim.device)
    env = MJWarpG1VecEnv(eval_sim, sonic_cfg, len(selected))
    reward_cfg = reward_cfg or RewardConfig()
    reward_fn = SonicStyleReward(
        reward_cfg,
        env.body_names,
        list(eval_sim.keypoint_body_names),
        env.joint_names,
    )
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
    previous_joint_vel = obs.joint_vel.clone()
    sums = {name: torch.zeros((), device=device) for name in (
        "root_position_error",
        "root_xy_error",
        "root_orientation_error",
        "mpjpe",
        "joint_position_error",
        "action_rate",
        "undesired_contacts",
        "mean_abs_actuator_force",
    )}
    samples = torch.zeros((), device=device)
    active_steps = torch.zeros(len(selected), device=device)
    termination_maxima = {
        name: torch.zeros((), device=device)
        for name in (
            "anchor_height_error",
            "anchor_orientation_error",
            "end_effector_height_error",
            "foot_position_error",
        )
    }
    termination_counts = {
        name: torch.zeros((), device=device)
        for name in (
            "anchor_height_failure",
            "anchor_orientation_failure",
            "end_effector_height_failure",
            "foot_position_failure",
        )
    }
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
        future = bank.future_reference(motion_ids, future_ids, obs.root_quat)
        out = policy(env.proprio_history(), future)
        action = torch.where(active[:, None], out.action_mean, torch.zeros_like(out.action_mean))
        action = action.clamp(-env.action_clip_value, env.action_clip_value)
        obs = env.step(action)
        frame_ids = frame_ids + active.long()
        safe_frame = torch.minimum(frame_ids, bank.lengths[motion_ids] - 1)
        ref = bank.current_reference(motion_ids, safe_frame)
        mask = active.float()
        root_error = (obs.root_pos[:, 2] - ref["root_pos"][:, 2]).abs()
        root_xy_error = (obs.root_pos[:, :2] - ref["root_pos"][:, :2]).norm(dim=-1)
        ori_error = quat_distance_angle(obs.root_quat, ref["root_quat"])
        reanchored_body_pos, _ = reanchor_reference_bodies(obs, ref)
        mpjpe = (obs.body_pos - reanchored_body_pos).norm(dim=-1).mean(-1)
        joint_error = (obs.joint_pos - ref["joint_pos"]).abs().mean(-1)
        action_rate = (action - previous_action).square().sum(-1)
        contacts = env.undesired_contact_count()
        force = env.mean_abs_actuator_force()
        reward = reward_fn(
            obs,
            ref,
            action,
            previous_action,
            env.soft_joint_low,
            env.soft_joint_high,
            previous_joint_vel,
            env.control_dt,
            contacts,
        )
        active_steps += active.float()
        for name, current_maximum in termination_maxima.items():
            active_values = reward.termination[name][active]
            if active_values.numel():
                termination_maxima[name] = torch.maximum(
                    current_maximum, active_values.max()
                )
        for name, current_count in termination_counts.items():
            termination_counts[name] = current_count + (
                active & reward.termination[name]
            ).float().sum()
        for name, value in (
            ("root_position_error", root_error),
            ("root_xy_error", root_xy_error),
            ("root_orientation_error", ori_error),
            ("mpjpe", mpjpe),
            ("joint_position_error", joint_error),
            ("action_rate", action_rate),
            ("undesired_contacts", contacts),
            ("mean_abs_actuator_force", force),
        ):
            sums[name] += (value * mask).sum()
        samples += mask.sum()
        tracking_failure = reward.done_tracking
        failed |= active & tracking_failure
        motion_end = frame_ids + (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride + 1 >= bank.lengths[motion_ids]
        active &= ~(tracking_failure | motion_end)
        previous_action = action
        previous_joint_vel = obs.joint_vel.clone()
    policy.train(was_training)
    denom = samples.clamp_min(1)
    metrics = {name: float(value / denom) for name, value in sums.items()}
    metrics["success_rate"] = float((~failed).float().mean())
    metrics["evaluated_motions"] = float(len(selected))
    future_span = (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride + 1
    available_steps = (bank.lengths[motion_ids] - future_span).clamp_min(1).float()
    survival_fraction = (active_steps / available_steps).clamp_max(1.0)
    metrics["mean_survival_fraction"] = float(survival_fraction.mean())
    metrics["minimum_survival_fraction"] = float(survival_fraction.min())
    for name, value in termination_maxima.items():
        metrics[f"max_{name}"] = float(value)
    for name, value in termination_counts.items():
        metrics[f"{name}_rate"] = float(value / len(selected))
    return metrics
