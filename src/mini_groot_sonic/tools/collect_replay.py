from __future__ import annotations

import argparse
import random
from pathlib import Path

from mini_groot_sonic.config import load_project_config
from mini_groot_sonic.data.episode_writer import EpisodeWriter
from mini_groot_sonic.data.replay import (
    collect_preprocessed_episode,
    load_policy_checkpoint,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay BONES clips in MJWarp and collect GR00T-like token datasets")
    ap.add_argument("--motions", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", default="replays")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--mode", choices=["policy", "reference_pd"], default="policy")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="Deterministic replay subset seed")
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--camera", default=None)
    ap.add_argument(
        "--randomized",
        action="store_true",
        help="Collect recovery states with training-time noise, latency, pushes, and dynamics randomization",
    )
    args = ap.parse_args()

    cfg = load_project_config(args.config)
    cfg.sim.mjcf = Path(args.mjcf)
    cfg.sim.device = args.device or cfg.sim.device
    cfg.sim.enable_randomization = args.randomized
    cfg.replay.output_dir = Path(args.out)
    cfg.replay.save_rgb = args.rgb
    cfg.replay.camera_name = args.camera
    sonic = cfg.sonic
    policy = None
    if args.checkpoint:
        policy, sonic = load_policy_checkpoint(args.checkpoint, cfg.sim.device)
    if args.mode == "policy" and policy is None:
        raise SystemExit("--mode policy requires --checkpoint")

    writer = EpisodeWriter(args.out)
    paths = sorted(Path(args.motions).glob("*.npz"))
    random.Random(args.seed).shuffle(paths)
    for i, p in enumerate(paths[: args.limit], start=1):
        out = collect_preprocessed_episode(
            p, cfg.sim, sonic, cfg.replay, writer, policy=policy, mode=args.mode
        )
        print(f"[{i}] wrote {out}")


if __name__ == "__main__":
    main()
