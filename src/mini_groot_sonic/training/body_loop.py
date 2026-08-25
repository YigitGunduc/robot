from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from mini_groot_sonic.config import PPOConfig, RewardConfig, SimConfig, SonicTinyConfig
from mini_groot_sonic.data.motion_bank import MotionBank
from mini_groot_sonic.models.sonic_tiny import TinySonicCritic, TinySonicPolicy
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv
from mini_groot_sonic.training.ppo import PPOAuxTrainer, Rollout
from mini_groot_sonic.training.rewards import SonicStyleReward


@dataclass
class BodyTrainState:
    motion_ids: torch.Tensor
    frame_ids: torch.Tensor
    previous_action: torch.Tensor
    previous_body_linvel: torch.Tensor | None


def _reset_envs(env: MJWarpG1VecEnv, bank: MotionBank, state: BodyTrainState, ids: torch.Tensor) -> None:
    if ids.numel() == 0:
        return
    sampled = bank.sample_start(ids.numel())
    state.motion_ids[ids] = sampled.motion_ids
    state.frame_ids[ids] = sampled.frame_ids
    ref = bank.current_reference(sampled.motion_ids, sampled.frame_ids)
    env.reset_idx(
        ids,
        ref["root_pos"],
        ref["root_quat"],
        ref["joint_pos"],
        joint_vel=ref["joint_vel"],
    )
    state.previous_action[ids] = 0
    if state.previous_body_linvel is not None:
        state.previous_body_linvel[ids] = env.observe().body_linvel[ids]


def train_body_controller(
    preprocessed_paths: list[str | Path],
    sim_cfg: SimConfig,
    sonic_cfg: SonicTinyConfig,
    reward_cfg: RewardConfig,
    ppo_cfg: PPOConfig,
    iterations: int,
    checkpoint_dir: str | Path,
) -> TinySonicPolicy:
    device = torch.device(sim_cfg.device)
    env = MJWarpG1VecEnv(sim_cfg, sonic_cfg, ppo_cfg.num_envs)
    bank = MotionBank(preprocessed_paths, sonic_cfg, sim_cfg.device)
    if bank.body_names != env.body_names:
        raise ValueError("Preprocessed body_names do not match the MJCF used for training")

    policy = TinySonicPolicy(sonic_cfg).to(device)
    critic = TinySonicCritic(sonic_cfg).to(device)
    trainer = PPOAuxTrainer(policy, critic, sonic_cfg, ppo_cfg)
    reward_fn = SonicStyleReward(reward_cfg, env.body_names, list(sim_cfg.keypoint_body_names))

    state = BodyTrainState(
        motion_ids=torch.zeros(ppo_cfg.num_envs, dtype=torch.long, device=device),
        frame_ids=torch.zeros(ppo_cfg.num_envs, dtype=torch.long, device=device),
        previous_action=torch.zeros(ppo_cfg.num_envs, sonic_cfg.dof, device=device),
        previous_body_linvel=torch.zeros(ppo_cfg.num_envs, len(env.body_names), 3, device=device),
    )
    _reset_envs(env, bank, state, torch.arange(ppo_cfg.num_envs, device=device))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(iterations):
        prop_buf, ref_buf, action_buf = [], [], []
        logp_buf, value_buf, reward_buf, done_buf = [], [], [], []
        reward_accum = 0.0

        for _ in range(ppo_cfg.rollout_steps):
            prop = env.proprio_history()
            future = bank.future_reference(state.motion_ids, state.frame_ids)
            with torch.no_grad():
                out = policy(prop, future)
                dist = policy.distribution(out.action_mean)
                action = dist.sample().clamp(-1.0, 1.0)
                logp = dist.log_prob(action).sum(-1)
                value = critic(prop, future)

            obs = env.step(action)
            state.frame_ids += 1
            ref_now = bank.current_reference(state.motion_ids, state.frame_ids)
            rew = reward_fn(
                obs,
                ref_now,
                action,
                state.previous_action,
                env.joint_low,
                env.joint_high,
                state.previous_body_linvel,
                env.control_dt,
            )
            state.previous_action = action.detach()
            state.previous_body_linvel.copy_(obs.body_linvel.detach())
            motion_end = state.frame_ids + (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride + 1 >= bank.lengths[state.motion_ids]
            done = rew.done_tracking | motion_end

            prop_buf.append(prop)
            ref_buf.append(future)
            action_buf.append(action)
            logp_buf.append(logp)
            value_buf.append(value)
            reward_buf.append(rew.total)
            done_buf.append(done)
            reward_accum += float(rew.total.mean())

            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            _reset_envs(env, bank, state, reset_ids)

        with torch.no_grad():
            last_prop = env.proprio_history()
            last_ref = bank.future_reference(state.motion_ids, state.frame_ids)
            last_value = critic(last_prop, last_ref)

        roll = Rollout(
            prop=torch.stack(prop_buf),
            ref=torch.stack(ref_buf),
            action=torch.stack(action_buf),
            old_logp=torch.stack(logp_buf),
            value=torch.stack(value_buf),
            reward=torch.stack(reward_buf),
            done=torch.stack(done_buf),
        )
        trainer.compute_gae(roll, last_value)
        stats = trainer.update(roll)
        print(
            f"iter={iteration:05d} reward={reward_accum/ppo_cfg.rollout_steps:.3f} "
            f"policy={stats['policy']:.4f} value={stats['value']:.4f} "
            f"recon={stats['recon']:.4f} kl={stats['kl']:.5f}"
        )

        if iteration % 100 == 0 or iteration == iterations - 1:
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "critic": critic.state_dict(),
                    "iteration": iteration,
                    "sonic_cfg": sonic_cfg.__dict__,
                },
                checkpoint_dir / f"body_{iteration:06d}.pt",
            )
    return policy
