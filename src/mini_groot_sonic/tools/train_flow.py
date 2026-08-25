from __future__ import annotations

import argparse

from mini_groot_sonic.config import FlowConfig
from mini_groot_sonic.training.flow_loop import train_flow_policy


def main() -> None:
    ap = argparse.ArgumentParser(description="Train compact GR00T-style flow model over 64D body tokens")
    ap.add_argument("--replays", required=True)
    ap.add_argument("--out", default="runs/flow")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--hf-model", default="google/siglip2-base-patch16-224")
    args = ap.parse_args()
    train_flow_policy(
        args.replays,
        FlowConfig(),
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hf_model=args.hf_model,
        device=args.device,
        use_vision=args.vision,
    )


if __name__ == "__main__":
    main()
