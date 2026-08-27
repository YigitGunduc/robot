from __future__ import annotations

import argparse
import random
from pathlib import Path

from mini_groot_sonic.config import load_project_config
from mini_groot_sonic.training.body_loop import train_body_controller
from mini_groot_sonic.training.utils import split_motion_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Train compact SONIC-style G1 universal motion-token controller")
    ap.add_argument("--motions", required=True, help="Directory containing preprocessed BONES .npz clips")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mjcf", default=None)
    ap.add_argument("--out", default="runs/body")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=100_000)
    ap.add_argument("--max-motions", type=int, default=256)
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--reset-best",
        action="store_true",
        help="Reset best-checkpoint comparison when resuming onto a new curriculum dataset",
    )
    randomization = ap.add_mutually_exclusive_group()
    randomization.add_argument(
        "--randomization", dest="randomization", action="store_true"
    )
    randomization.add_argument(
        "--no-randomization", dest="randomization", action="store_false"
    )
    ap.set_defaults(randomization=None)
    args = ap.parse_args()

    cfg = load_project_config(args.config)
    if args.mjcf is not None:
        cfg.sim.mjcf = Path(args.mjcf)
    if not cfg.sim.mjcf.exists():
        raise SystemExit("Provide --mjcf or set sim.mjcf in the config")
    if args.device is not None:
        cfg.sim.device = args.device
    if args.num_envs is not None:
        cfg.ppo.num_envs = args.num_envs
    if args.randomization is not None:
        cfg.sim.enable_randomization = args.randomization
    paths = sorted(Path(args.motions).glob("*.npz"))
    random.Random(cfg.ppo.seed).shuffle(paths)
    paths = paths[: args.max_motions]
    if not paths:
        raise SystemExit("No preprocessed motion NPZ files found")
    training, validation = split_motion_paths(
        paths,
        args.validation_fraction,
        cfg.ppo.seed,
    )
    train_body_controller(
        training,
        cfg.sim,
        cfg.sonic,
        cfg.reward,
        cfg.ppo,
        args.iterations,
        args.out,
        validation_paths=validation,
        resume_from=args.resume,
        reset_best_on_resume=args.reset_best,
    )


if __name__ == "__main__":
    main()
