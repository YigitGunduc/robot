from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_groot_sonic.config import load_project_config
from mini_groot_sonic.models.runtime import load_body_checkpoint
from mini_groot_sonic.training.evaluation import evaluate_body_controller


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a body checkpoint on held-out motion clips")
    ap.add_argument("--motions", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-motions", type=int, default=16)
    args = ap.parse_args()

    cfg = load_project_config(args.config)
    cfg.sim.mjcf = Path(args.mjcf)
    cfg.sim.device = args.device or cfg.sim.device
    cfg.sim.enable_randomization = False
    policy, sonic_cfg = load_body_checkpoint(args.body, cfg.sim.device)
    paths = sorted(Path(args.motions).glob("*.npz"))[: args.max_motions]
    if not paths:
        raise SystemExit("No preprocessed motion NPZ files found")
    metrics = evaluate_body_controller(paths, cfg.sim, sonic_cfg, policy, max_motions=args.max_motions)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
