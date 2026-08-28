from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from gear_sonic_mjx.checkpoint_utils import capture_rng_state, restore_rng_state
from gear_sonic_mjx.config import ModelConfig, SonicConfig
from gear_sonic_mjx.envs.g1_tracking_task import G1SonicTrackingTask
from gear_sonic_mjx.envs.motion_library import open_motion_library
from gear_sonic_mjx.g1_parameters import (
    SONIC_FOOT_BODY_NAMES,
    SONIC_REWARD_POINT_BODY_NAMES,
)
from gear_sonic_mjx.sim.mjwarp_backend import MjWarpBatchSim
from gear_sonic_mjx.trl.modules.base_module import MLP
from gear_sonic_mjx.trl.modules.universal_token_modules import UniversalTokenModule
from gear_sonic_mjx.trl.trainer.ppo_trainer_aux_loss import (
    PPOTrainer,
    RolloutStorage,
    SonicActorCritic,
)

DEFAULT_REWARD_POINTS = SONIC_REWARD_POINT_BODY_NAMES
DEFAULT_FEET = SONIC_FOOT_BODY_NAMES


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    path: Path, iteration: int, model, trainer, task, cfg, observations
) -> None:
    payload = {
        "training_state_version": 2,
        "iteration": iteration,
        "model": model.state_dict(),
        "token_module": model.token_module.state_dict(),
        "actor_optimizer": trainer.actor_opt.state_dict(),
        "critic_optimizer": trainer.critic_opt.state_dict(),
        "task": task.state_dict(),
        "observations": tuple(x.detach().cpu() for x in observations),
        "rng": capture_rng_state(),
        "config": cfg,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(
        description="Train SONIC-like G1 tracking policy with BONES-SEED + MuJoCo-Warp"
    )
    ap.add_argument("--mjcf", required=True)
    ap.add_argument(
        "--motions", required=True, help="preprocessed + FK-augmented BONES directory"
    )
    ap.add_argument(
        "--config",
        default=str(
            Path(__file__).parents[1] / "gear_sonic_mjx/config/sonic_release_mjx.yaml"
        ),
    )
    ap.add_argument("--output", default="runs/sonic_mjwarp")
    ap.add_argument("--network", choices=["small", "nvidia"], default=None)
    ap.add_argument(
        "--num-envs",
        type=int,
        help="override config for smoke tests/benchmark-selected scale",
    )
    ap.add_argument("--iterations", type=int, help="override maximum PPO iterations")
    ap.add_argument("--save-interval", type=int, help="override checkpoint interval")
    ap.add_argument("--seed", type=int, help="override random seed")
    ap.add_argument(
        "--reward-point-bodies",
        "--fivepoint-bodies",
        dest="reward_point_bodies",
        nargs="*",
        default=DEFAULT_REWARD_POINTS,
    )
    ap.add_argument("--foot-bodies", nargs="*", default=DEFAULT_FEET)
    ap.add_argument("--allow-missing-fk", action="store_true")
    ap.add_argument("--no-domain-randomization", action="store_true")
    ap.add_argument(
        "--dr-variants",
        type=int,
        default=64,
        help="compiled MJWarp physics variants distributed over worlds",
    )
    ap.add_argument("--resume")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = SonicConfig.from_yaml(args.config)
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_cfg = resume_ckpt.get("config")
        if isinstance(saved_cfg, SonicConfig):
            cfg = saved_cfg
    if args.network == "nvidia":
        cfg.model = ModelConfig.nvidia_release()
    elif args.network == "small" and cfg.model.preset == "nvidia_release":
        raise ValueError(
            "cannot resume an NVIDIA-width checkpoint with --network small"
        )
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.iterations is not None:
        cfg.ppo.num_learning_iterations = args.iterations
    if args.save_interval is not None:
        cfg.ppo.save_interval = args.save_interval
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_domain_randomization:
        cfg.domain_randomization = {}
    cfg.validate()
    _seed_everything(cfg.seed)

    sim = MjWarpBatchSim(
        args.mjcf,
        cfg.num_envs,
        cfg.sim.sim_dt,
        cfg.sim.nconmax,
        cfg.sim.njmax,
        cfg.sim.naconmax,
    )
    device = sim.device
    if not args.no_domain_randomization:
        # NVIDIA level0_4 includes physical-property randomization. Select torso/wrist bodies by
        # semantic name so this works across common Unitree G1 MJCF variants.
        names = []
        for bid in range(1, sim.mj_model.nbody):
            name = (
                sim.mujoco.mj_id2name(sim.mj_model, sim.mujoco.mjtObj.mjOBJ_BODY, bid)
                or ""
            )
            if any(k in name.lower() for k in ("wrist", "torso", "waist")):
                names.append(name)
        sim.configure_startup_domain_randomization(
            mass_body_names=names, num_variants=args.dr_variants, seed=cfg.seed
        )
    motions = open_motion_library(args.motions, cfg.motion.target_fps)
    task = G1SonicTrackingTask(
        sim,
        motions,
        cfg,
        reward_point_names=args.reward_point_bodies,
        foot_names=args.foot_bodies,
        require_fk_cache=not args.allow_missing_fk,
    )
    token = UniversalTokenModule(
        cfg.model, cfg.motion.num_future_frames, cfg.motion.actor_prop_history_length
    ).to(device)
    encoder_dim = token.encoder_input_dim
    proprio_dim = token.proprio_dim
    critic_dim = task.critic_dim
    critic = MLP(critic_dim, cfg.model.critic_hidden, 1).to(device)
    model = SonicActorCritic(
        token,
        critic,
        cfg.model.dof,
        cfg.ppo.init_noise_std,
        cfg.ppo.std_clamp_min,
        cfg.ppo.std_clamp_max,
    ).to(device)
    trainer = PPOTrainer(model, cfg.ppo)
    start_iter = 0
    restored_task = False
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        trainer.actor_opt.load_state_dict(resume_ckpt["actor_optimizer"])
        trainer.critic_opt.load_state_dict(resume_ckpt["critic_optimizer"])
        start_iter = int(resume_ckpt.get("iteration", 0)) + 1
        if "task" in resume_ckpt:
            task.load_state_dict(resume_ckpt["task"])
            restored_task = True
        if "rng" in resume_ckpt:
            restore_rng_state(resume_ckpt["rng"])

    storage = RolloutStorage(
        cfg.ppo.num_steps_per_env,
        cfg.num_envs,
        encoder_dim,
        proprio_dim,
        critic_dim,
        cfg.model.dof,
        device,
    )
    if restored_task and "observations" in resume_ckpt:
        enc, prop, critic_obs = tuple(x.to(device) for x in resume_ckpt["observations"])
    else:
        enc, prop, critic_obs = (
            task._obs(advance_history=False) if restored_task else task.reset()
        )
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    last_iteration = start_iter - 1
    for iteration in range(start_iter, cfg.ppo.num_learning_iterations):
        started = time.perf_counter()
        storage.clear()
        for _ in range(cfg.ppo.num_steps_per_env):
            with torch.no_grad():
                action, logp, _ = model.act(enc, prop)
                value = model.value(critic_obs)
                step = task.step(action)
            if (
                not torch.isfinite(action).all()
                or not torch.isfinite(step.reward).all()
            ):
                raise FloatingPointError(
                    f"non-finite action/reward at iteration {iteration}; refusing to continue"
                )
            storage.add(
                enc, prop, critic_obs, action, logp, value, step.reward, step.done
            )
            enc, prop, critic_obs = step.encoder_obs, step.proprio_obs, step.critic_obs
        with torch.no_grad():
            last_value = model.value(critic_obs)
        storage.compute_returns(last_value, cfg.ppo.gamma, cfg.ppo.lam)
        stats = trainer.update(storage)
        if not all(np.isfinite(value) for value in stats.values()):
            raise FloatingPointError(
                f"non-finite PPO statistics at iteration {iteration}: {stats}"
            )
        last_iteration = iteration

        if iteration % 10 == 0:
            samples = cfg.ppo.num_steps_per_env * cfg.num_envs
            samples_per_second = samples / max(time.perf_counter() - started, 1e-9)
            print(
                f"it={iteration:06d} reward={storage.rewards.mean().item():.4f} "
                f"pi={stats['policy_loss']:.4f} v={stats['value_loss']:.4f} "
                f"aux={stats['aux_loss']:.5f} kl={stats['kl']:.5f} "
                f"lr={stats['actor_lr']:.2e} samples/s={samples_per_second:,.0f}"
            )
        if iteration and iteration % cfg.ppo.save_interval == 0:
            path = outdir / f"checkpoint_{iteration:07d}.pt"
            _save_checkpoint(
                path, iteration, model, trainer, task, cfg, (enc, prop, critic_obs)
            )

    if last_iteration >= start_iter:
        final_path = outdir / f"checkpoint_{last_iteration:07d}.pt"
        _save_checkpoint(
            final_path,
            last_iteration,
            model,
            trainer,
            task,
            cfg,
            (enc, prop, critic_obs),
        )
        print(f"saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
