from __future__ import annotations

import argparse
from pathlib import Path

from mini_groot_sonic.config import PPOConfig, RewardConfig, SimConfig, SonicTinyConfig
from mini_groot_sonic.training.body_loop import train_body_controller


def main() -> None:
    ap = argparse.ArgumentParser(description="Train compact SONIC-style G1 universal motion-token controller")
    ap.add_argument("--motions", required=True, help="Directory containing preprocessed BONES .npz clips")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", default="runs/body")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--max-motions", type=int, default=512)
    args = ap.parse_args()

    paths = sorted(Path(args.motions).glob("*.npz"))[: args.max_motions]
    if not paths:
        raise SystemExit("No preprocessed motion NPZ files found")
    sonic = SonicTinyConfig()
    sim = SimConfig(mjcf=Path(args.mjcf), device=args.device)
    ppo = PPOConfig(num_envs=args.num_envs)
    train_body_controller(paths, sim, sonic, RewardConfig(), ppo, args.iterations, args.out)


if __name__ == "__main__":
    main()
