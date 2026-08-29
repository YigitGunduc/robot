from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear_sonic_mjx.data_process.subset import (
    build_easy_subset_manifest,
    materialize_subset,
    save_subset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic, mirror-safe easy BONES development subset"
    )
    parser.add_argument(
        "--motions", required=True, help="full preprocessed BONES library"
    )
    parser.add_argument(
        "--metadata", required=True, help="official BONES parquet/csv metadata"
    )
    parser.add_argument(
        "--output", required=True, help="subset motion-library directory"
    )
    parser.add_argument(
        "--manifest", required=True, help="auditable subset JSON manifest"
    )
    parser.add_argument("--max-clips", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("version") != 2 or manifest.get("preset") != "easy":
            raise ValueError(f"unsupported existing subset manifest: {manifest_path}")
        if int(manifest.get("max_clips", -1)) != args.max_clips:
            raise ValueError(
                f"existing subset max_clips={manifest.get('max_clips')} does not match "
                f"requested {args.max_clips}; use a new output/manifest name"
            )
        if int(manifest.get("seed", -1)) != args.seed:
            raise ValueError(
                f"existing subset seed={manifest.get('seed')} does not match requested "
                f"{args.seed}; use a new output/manifest name"
            )
    else:
        manifest = build_easy_subset_manifest(
            args.motions, args.metadata, max_clips=args.max_clips, seed=args.seed
        )
        save_subset_manifest(manifest, manifest_path)
    stats = materialize_subset(args.motions, args.output, manifest)
    print(
        {
            **stats,
            "class_counts": manifest["class_counts"],
            "manifest": str(manifest_path),
        }
    )


if __name__ == "__main__":
    main()
