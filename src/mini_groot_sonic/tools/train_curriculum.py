from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mini_groot_sonic.config import load_project_config
from mini_groot_sonic.training.curriculum import (
    PromotionCriteria,
    train_dynamic_curriculum,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train and dynamically promote a compact BONES motion curriculum"
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--motions", default=None, help="Override motion directory stored in manifest")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--evaluation-chunk-iterations", type=int, default=100)
    ap.add_argument("--minimum-stage-iterations", type=int, default=1000)
    ap.add_argument("--maximum-stage-iterations", type=int, default=20_000)
    ap.add_argument("--promotion-patience", type=int, default=2)
    ap.add_argument("--promotion-success-rate", type=float, default=0.80)
    ap.add_argument("--promotion-mpjpe", type=float, default=0.08)
    ap.add_argument("--promotion-root-position-error", type=float, default=0.08)
    ap.add_argument("--promotion-root-orientation-error", type=float, default=0.20)
    ap.add_argument("--minimum-evaluated-motions", type=int, default=1)
    ap.add_argument("--randomization-start-stage", type=int, default=3)
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
    cfg.sim.mjcf = Path(args.mjcf)
    cfg.sim.device = args.device or cfg.sim.device
    if args.num_envs is not None:
        cfg.ppo.num_envs = args.num_envs
    if args.randomization is not None:
        cfg.sim.enable_randomization = args.randomization
    state = train_dynamic_curriculum(
        args.manifest,
        cfg.sim,
        cfg.sonic,
        cfg.reward,
        cfg.ppo,
        args.out,
        motion_dir_override=args.motions,
        validation_fraction=args.validation_fraction,
        evaluation_chunk_iterations=args.evaluation_chunk_iterations,
        minimum_stage_iterations=args.minimum_stage_iterations,
        maximum_stage_iterations=args.maximum_stage_iterations,
        promotion_patience=args.promotion_patience,
        criteria=PromotionCriteria(
            success_rate=args.promotion_success_rate,
            mpjpe=args.promotion_mpjpe,
            root_position_error=args.promotion_root_position_error,
            root_orientation_error=args.promotion_root_orientation_error,
            minimum_evaluated_motions=args.minimum_evaluated_motions,
        ),
        randomization_start_stage=args.randomization_start_stage,
        resume_from=args.resume,
    )
    print(json.dumps(asdict(state), indent=2))


if __name__ == "__main__":
    main()
