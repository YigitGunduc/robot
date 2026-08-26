from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from mini_groot_sonic.config import PPOConfig, RewardConfig, SimConfig, SonicTinyConfig
from mini_groot_sonic.data.motion_bank import MotionBank
from mini_groot_sonic.models.sonic_tiny import TinySonicCritic, TinySonicPolicy
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv
from mini_groot_sonic.training.evaluation import evaluate_body_controller
from mini_groot_sonic.training.ppo import PPOAuxTrainer, Rollout
from mini_groot_sonic.training.rewards import SonicStyleReward
from mini_groot_sonic.training.utils import (
    restore_rng_state,
    rng_state,
    save_config_snapshot,
    seed_everything,
)


@dataclass
class BodyTrainState:
    motion_ids: torch.Tensor
    frame_ids: torch.Tensor
    previous_action: torch.Tensor
    previous_joint_vel: torch.Tensor | None


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
        root_linvel=ref["root_linvel"],
        root_angvel=ref["root_angvel"],
        joint_vel=ref["joint_vel"],
    )
    state.previous_action[ids] = 0
    if state.previous_joint_vel is not None:
        state.previous_joint_vel[ids] = env.observe().joint_vel[ids]


def train_body_controller(
    preprocessed_paths: list[str | Path],
    sim_cfg: SimConfig,
    sonic_cfg: SonicTinyConfig,
    reward_cfg: RewardConfig,
    ppo_cfg: PPOConfig,
    iterations: int,
    checkpoint_dir: str | Path,
    *,
    validation_paths: list[str | Path] | None = None,
    resume_from: str | Path | None = None,
) -> TinySonicPolicy:
    seed_everything(ppo_cfg.seed)
    device = torch.device(sim_cfg.device)
    env = MJWarpG1VecEnv(sim_cfg, sonic_cfg, ppo_cfg.num_envs)
    bank = MotionBank(preprocessed_paths, sonic_cfg, sim_cfg.device)
    if bank.body_names != env.body_names:
        raise ValueError("Preprocessed body_names do not match the MJCF used for training")

    policy = TinySonicPolicy(sonic_cfg).to(device)
    critic = TinySonicCritic(sonic_cfg).to(device)
    reference_mean, reference_std = bank.reference_stats()
    policy.set_reference_stats(reference_mean, reference_std)
    critic.set_reference_stats(reference_mean, reference_std)
    trainer = PPOAuxTrainer(policy, critic, sonic_cfg, ppo_cfg)
    reward_fn = SonicStyleReward(
        reward_cfg,
        env.body_names,
        list(sim_cfg.keypoint_body_names),
        env.joint_names,
    )

    start_iteration = 0
    best_success = -1.0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["policy"])
        critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            trainer.optim.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint.get("iteration", -1)) + 1
        best_success = float(checkpoint.get("best_success", best_success))
        restore_rng_state(checkpoint.get("rng_state"))

    state = BodyTrainState(
        motion_ids=torch.zeros(ppo_cfg.num_envs, dtype=torch.long, device=device),
        frame_ids=torch.zeros(ppo_cfg.num_envs, dtype=torch.long, device=device),
        previous_action=torch.zeros(ppo_cfg.num_envs, sonic_cfg.dof, device=device),
        previous_joint_vel=torch.zeros(ppo_cfg.num_envs, sonic_cfg.dof, device=device),
    )
    _reset_envs(env, bank, state, torch.arange(ppo_cfg.num_envs, device=device))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(
        checkpoint_dir / "config.json",
        sonic=sonic_cfg,
        sim=sim_cfg,
        reward=reward_cfg,
        ppo=ppo_cfg,
    )

    for iteration in range(start_iteration, iterations):
        prop_buf, ref_buf, privileged_buf, action_buf = [], [], [], []
        logp_buf, value_buf, reward_buf, done_buf, token_index_buf = [], [], [], [], []
        reward_accum = torch.zeros((), device=device)
        reward_terms: dict[str, torch.Tensor] = {}

        for _ in range(ppo_cfg.rollout_steps):
            prop = env.proprio_history()
            future = bank.future_reference(state.motion_ids, state.frame_ids)
            if sim_cfg.enable_randomization:
                future = future.clone()
                future[..., : sonic_cfg.dof] += (
                    sim_cfg.reference_joint_noise
                    * torch.randn_like(future[..., : sonic_cfg.dof])
                )
                root_start = sonic_cfg.dof * 2
                future[..., root_start:] += (
                    sim_cfg.reference_root_noise
                    * torch.randn_like(future[..., root_start:])
                )
            with torch.no_grad():
                out = policy(prop, future)
                dist = policy.distribution(out.action_mean)
                action = dist.rsample()
                logp = dist.log_prob(action).sum(-1)
                privileged = env.privileged_observation()
                value = critic(prop, future, privileged)

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
                state.previous_joint_vel,
                env.control_dt,
                env.undesired_contact_count(),
            )
            state.previous_action = action.detach()
            state.previous_joint_vel.copy_(obs.joint_vel.detach())
            motion_end = state.frame_ids + (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride + 1 >= bank.lengths[state.motion_ids]
            done = rew.done_tracking | motion_end

            prop_buf.append(prop)
            ref_buf.append(future)
            privileged_buf.append(privileged)
            action_buf.append(action)
            logp_buf.append(logp)
            value_buf.append(value)
            reward_buf.append(rew.total)
            done_buf.append(done)
            token_index_buf.append(out.token_indices)
            reward_accum += rew.total.mean()
            for name, value in rew.terms.items():
                mean_value = value.mean()
                reward_terms[name] = reward_terms.get(name, torch.zeros_like(mean_value)) + mean_value

            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel():
                bank.update_failures(
                    state.motion_ids[reset_ids],
                    rew.done_tracking[reset_ids],
                    ppo_cfg.failure_sampling_alpha,
                    ppo_cfg.failure_sampling_cap,
                )
            _reset_envs(env, bank, state, reset_ids)

        with torch.no_grad():
            last_prop = env.proprio_history()
            last_ref = bank.future_reference(state.motion_ids, state.frame_ids)
            last_value = critic(last_prop, last_ref, env.privileged_observation())

        roll = Rollout(
            prop=torch.stack(prop_buf),
            ref=torch.stack(ref_buf),
            privileged=torch.stack(privileged_buf),
            action=torch.stack(action_buf),
            old_logp=torch.stack(logp_buf),
            value=torch.stack(value_buf),
            reward=torch.stack(reward_buf),
            done=torch.stack(done_buf),
        )
        trainer.compute_gae(roll, last_value)
        stats = trainer.update(roll)
        token_indices = torch.stack(token_index_buf).flatten(0, 1).long()
        occupancy = torch.nn.functional.one_hot(
            token_indices,
            num_classes=sonic_cfg.fsq_levels,
        ).any(dim=0).float().mean()
        saturation = (
            (token_indices == 0) | (token_indices == sonic_cfg.fsq_levels - 1)
        ).float().mean()
        print(
            f"iter={iteration:05d} reward={float(reward_accum/ppo_cfg.rollout_steps):.3f} "
            f"policy={stats['policy']:.4f} value={stats['value']:.4f} "
            f"recon={stats['recon']:.4f} kl={stats['kl']:.5f} "
            f"fsq_occupancy={float(occupancy):.3f} fsq_saturation={float(saturation):.3f}"
        )
        if iteration % 20 == 0:
            summary = " ".join(
                f"{name}={float(value / ppo_cfg.rollout_steps):.3f}"
                for name, value in sorted(reward_terms.items())
            )
            print(f"reward_terms {summary}")

        eval_metrics = {}
        if validation_paths and (
            iteration % ppo_cfg.eval_interval == 0 or iteration == iterations - 1
        ):
            eval_metrics = evaluate_body_controller(
                validation_paths,
                sim_cfg,
                sonic_cfg,
                policy,
                max_motions=8,
            )
            print("validation " + " ".join(f"{k}={v:.5f}" for k, v in sorted(eval_metrics.items())))

        if iteration % ppo_cfg.checkpoint_interval == 0 or iteration == iterations - 1:
            success = float(eval_metrics.get("success_rate", best_success))
            checkpoint = {
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": trainer.optim.state_dict(),
                "iteration": iteration,
                "best_success": max(best_success, success),
                "sonic_cfg": asdict(sonic_cfg),
                "sim_cfg": asdict(sim_cfg),
                "reward_cfg": asdict(reward_cfg),
                "ppo_cfg": asdict(ppo_cfg),
                "rng_state": rng_state(),
                "validation": eval_metrics,
            }
            torch.save(checkpoint, checkpoint_dir / f"body_{iteration:06d}.pt")
            if eval_metrics and success >= best_success:
                best_success = success
                torch.save(checkpoint, checkpoint_dir / "body_best.pt")
    return policy
