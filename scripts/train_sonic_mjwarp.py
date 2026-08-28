from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gear_sonic_mjx.config import ModelConfig, SonicConfig
from gear_sonic_mjx.envs.g1_tracking_task import G1SonicTrackingTask
from gear_sonic_mjx.envs.motion_library import open_motion_library
from gear_sonic_mjx.sim.mjwarp_backend import MjWarpBatchSim
from gear_sonic_mjx.trl.modules.base_module import MLP
from gear_sonic_mjx.trl.modules.universal_token_modules import UniversalTokenModule
from gear_sonic_mjx.trl.trainer.ppo_trainer_aux_loss import PPOTrainer, RolloutStorage, SonicActorCritic

from gear_sonic_mjx.g1_parameters import SONIC_REWARD_POINT_BODY_NAMES, SONIC_FOOT_BODY_NAMES

DEFAULT_REWARD_POINTS = SONIC_REWARD_POINT_BODY_NAMES
DEFAULT_FEET = SONIC_FOOT_BODY_NAMES


def main():
    ap = argparse.ArgumentParser(description="Train SONIC-like G1 tracking policy with BONES-SEED + MuJoCo-Warp")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--motions", required=True, help="preprocessed + FK-augmented BONES directory")
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "gear_sonic_mjx/config/sonic_release_mjx.yaml"))
    ap.add_argument("--output", default="runs/sonic_mjwarp")
    ap.add_argument("--network", choices=["small", "nvidia"], default="small")
    ap.add_argument("--reward-point-bodies", "--fivepoint-bodies", dest="reward_point_bodies", nargs="*", default=DEFAULT_REWARD_POINTS)
    ap.add_argument("--foot-bodies", nargs="*", default=DEFAULT_FEET)
    ap.add_argument("--allow-missing-fk", action="store_true")
    ap.add_argument("--no-domain-randomization", action="store_true")
    ap.add_argument("--dr-variants", type=int, default=64, help="compiled MJWarp physics variants distributed over worlds")
    ap.add_argument("--resume")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = SonicConfig.from_yaml(args.config)
    if args.network == "nvidia":
        cfg.model = ModelConfig.nvidia_release()
    sim = MjWarpBatchSim(args.mjcf, cfg.num_envs, cfg.sim.sim_dt, cfg.sim.nconmax, cfg.sim.njmax)
    device = sim.device
    if not args.no_domain_randomization:
        # NVIDIA level0_4 includes physical-property randomization. Select torso/wrist bodies by
        # semantic name so this works across common Unitree G1 MJCF variants.
        names = []
        for bid in range(1, sim.mj_model.nbody):
            name = sim.mujoco.mj_id2name(sim.mj_model, sim.mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if any(k in name.lower() for k in ("wrist", "torso", "waist")):
                names.append(name)
        sim.configure_startup_domain_randomization(mass_body_names=names, num_variants=args.dr_variants, seed=cfg.seed)
    motions = open_motion_library(args.motions, cfg.motion.target_fps)
    task = G1SonicTrackingTask(
        sim, motions, cfg, reward_point_names=args.reward_point_bodies, foot_names=args.foot_bodies,
        require_fk_cache=not args.allow_missing_fk,
    )
    token = UniversalTokenModule(cfg.model, cfg.motion.num_future_frames, cfg.motion.actor_prop_history_length).to(device)
    encoder_dim = token.encoder_input_dim
    proprio_dim = token.proprio_dim
    critic_dim = task.critic_dim
    critic = MLP(critic_dim, cfg.model.critic_hidden, 1).to(device)
    model = SonicActorCritic(token, critic, cfg.model.dof, cfg.ppo.init_noise_std, cfg.ppo.std_clamp_min, cfg.ppo.std_clamp_max).to(device)
    trainer = PPOTrainer(model, cfg.ppo)
    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        trainer.actor_opt.load_state_dict(ckpt["actor_optimizer"])
        trainer.critic_opt.load_state_dict(ckpt["critic_optimizer"])
        start_iter = int(ckpt.get("iteration", 0)) + 1

    storage = RolloutStorage(cfg.ppo.num_steps_per_env, cfg.num_envs, encoder_dim, proprio_dim, critic_dim, cfg.model.dof, device)
    enc, prop, critic_obs = task.reset()
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)

    for iteration in range(start_iter, cfg.ppo.num_learning_iterations):
        storage.clear()
        for _ in range(cfg.ppo.num_steps_per_env):
            with torch.no_grad():
                action, logp, _ = model.act(enc, prop)
                value = model.value(critic_obs)
                step = task.step(action)
            storage.add(enc, prop, critic_obs, action, logp, value, step.reward, step.done)
            enc, prop, critic_obs = step.encoder_obs, step.proprio_obs, step.critic_obs
        with torch.no_grad():
            last_value = model.value(critic_obs)
        storage.compute_returns(last_value, cfg.ppo.gamma, cfg.ppo.lam)
        stats = trainer.update(storage)

        if iteration % 10 == 0:
            print(
                f"it={iteration:06d} reward={storage.rewards.mean().item():.4f} "
                f"pi={stats['policy_loss']:.4f} v={stats['value_loss']:.4f} "
                f"aux={stats['aux_loss']:.5f} kl={stats['kl']:.5f} lr={stats['actor_lr']:.2e}"
            )
        if iteration and iteration % cfg.ppo.save_interval == 0:
            path = outdir / f"checkpoint_{iteration:07d}.pt"
            torch.save({
                "iteration": iteration, "model": model.state_dict(), "token_module": model.token_module.state_dict(),
                "actor_optimizer": trainer.actor_opt.state_dict(), "critic_optimizer": trainer.critic_opt.state_dict(),
                "config": cfg,
            }, path)


if __name__ == "__main__":
    main()
