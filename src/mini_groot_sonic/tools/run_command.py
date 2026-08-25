from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_groot_sonic.config import SimConfig, SonicTinyConfig
from mini_groot_sonic.models.frozen_backbones import FrozenSiglip2
from mini_groot_sonic.models.runtime import RecedingHorizonTokenController, load_body_checkpoint, load_flow_checkpoint
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a natural-language body command through flow tokens -> tiny SONIC controller")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hf-model", default="google/siglip2-base-patch16-224")
    args = ap.parse_args()

    sonic_cfg = SonicTinyConfig()
    sim_cfg = SimConfig(mjcf=Path(args.mjcf), device=args.device)
    env = MJWarpG1VecEnv(sim_cfg, sonic_cfg, 1)
    body = load_body_checkpoint(args.body, sonic_cfg, args.device)
    flow, _ = load_flow_checkpoint(args.flow, args.device)
    backbone = FrozenSiglip2(args.hf_model, args.device)
    runner = RecedingHorizonTokenController(flow, body, backbone)

    # Neutral G1 start. Production should reset from a known stand checkpoint/reference.
    root_pos = torch.tensor([[0.0, 0.0, 0.78]], device=args.device)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args.device)
    q = env._default_joint_pos[None].clone()
    obs = env.reset(root_pos, root_quat, q)
    for i in range(args.steps):
        action, token = runner.action(args.command, obs, env.proprio_history())
        obs = env.step(action)
        if i % 50 == 0:
            print(f"step={i} root={obs.root_pos[0].tolist()} token_norm={float(token.norm()):.3f}")


if __name__ == "__main__":
    main()
