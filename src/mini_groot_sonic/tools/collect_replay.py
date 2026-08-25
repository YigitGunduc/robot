from __future__ import annotations

import argparse
from pathlib import Path

from mini_groot_sonic.config import ReplayConfig, SimConfig, SonicTinyConfig
from mini_groot_sonic.data.episode_writer import EpisodeWriter
from mini_groot_sonic.data.replay import collect_preprocessed_episode, load_policy_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay BONES clips in MJWarp and collect GR00T-like token datasets")
    ap.add_argument("--motions", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", default="replays")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--mode", choices=["policy", "reference_pd"], default="policy")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--camera", default=None)
    args = ap.parse_args()

    sonic = SonicTinyConfig()
    sim = SimConfig(mjcf=Path(args.mjcf), device=args.device)
    replay = ReplayConfig(output_dir=Path(args.out), save_rgb=args.rgb, camera_name=args.camera)
    policy = None
    if args.checkpoint:
        policy = load_policy_checkpoint(args.checkpoint, sonic, args.device)
    if args.mode == "policy" and policy is None:
        raise SystemExit("--mode policy requires --checkpoint")

    writer = EpisodeWriter(args.out)
    for i, p in enumerate(sorted(Path(args.motions).glob("*.npz"))[: args.limit], start=1):
        out = collect_preprocessed_episode(p, sim, sonic, replay, writer, policy=policy, mode=args.mode)
        print(f"[{i}] wrote {out}")


if __name__ == "__main__":
    main()
