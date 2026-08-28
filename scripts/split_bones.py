from __future__ import annotations

import argparse
from pathlib import Path

from gear_sonic_mjx.data_process.splits import (
    build_split_manifest,
    save_split_manifest,
    validate_split_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic clip-level BONES train/validation/test splits"
    )
    parser.add_argument("--motions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()

    manifest = build_split_manifest(
        args.motions,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    validate_split_manifest(manifest)
    save_split_manifest(manifest, args.output)
    counts = {name: len(paths) for name, paths in manifest["splits"].items()}
    print(f"wrote {Path(args.output)}: {counts}")


if __name__ == "__main__":
    main()
