from __future__ import annotations

import argparse

from mini_groot_sonic.config import load_project_config
from mini_groot_sonic.training.flow_loop import train_flow_policy


def main() -> None:
    ap = argparse.ArgumentParser(description="Train compact GR00T-style flow model over 64D body tokens")
    ap.add_argument("--replays", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="runs/flow")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--hf-model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    cfg = load_project_config(args.config)
    device = args.device or cfg.sim.device
    train_flow_policy(
        args.replays,
        cfg.flow,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hf_model=args.hf_model,
        device=device,
        use_vision=args.vision,
        num_workers=args.workers,
        use_amp=not args.no_amp,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
