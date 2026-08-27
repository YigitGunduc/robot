from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_groot_sonic.data.curriculum import (
    STAGE_NAMES,
    build_curriculum,
    write_curriculum_artifacts,
)


def _stage_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Stage sizes must be comma-separated integers") from exc
    if len(sizes) != len(STAGE_NAMES):
        raise argparse.ArgumentTypeError(
            f"Expected {len(STAGE_NAMES)} sizes for {', '.join(STAGE_NAMES)}"
        )
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build an auditable structured-metadata and kinematic BONES curriculum"
    )
    ap.add_argument("--motions", required=True, help="Directory of preprocessed BONES NPZ files")
    ap.add_argument("--out", required=True, help="Curriculum manifest/audit output directory")
    ap.add_argument(
        "--stage-sizes",
        type=_stage_sizes,
        default=_stage_sizes("8,20,32,48,64"),
        help="Cumulative sizes for balance, neutral walk, walk variations, turns, jog/run",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    motion_dir = Path(args.motions).resolve()
    paths = sorted(motion_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"No preprocessed NPZ motions found in {motion_dir}")
    stages, records = build_curriculum(paths, args.stage_sizes, seed=args.seed)
    manifest = write_curriculum_artifacts(
        args.out,
        motion_dir,
        stages,
        records,
        seed=args.seed,
    )
    report = {
        "manifest": str(manifest),
        "analyzed": len(records),
        "quality_passed": sum(record.quality_passed for record in records),
        "semantically_eligible": sum(record.semantic_stage is not None for record in records),
        "stages": {
            stage.name: {"motions": len(stage.filenames), "new": len(stage.new_filenames)}
            for stage in stages
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
