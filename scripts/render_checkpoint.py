#!/usr/bin/env python3
"""Render one trained checkpoint without rerunning training."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


TASK = "Mjlab-SonicLite-Tracking-Flat-Unitree-G1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-length", type=int, default=600)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    motion_file = args.motion_file.expanduser().resolve()
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    if not motion_file.exists():
        raise SystemExit(f"Motion file does not exist: {motion_file}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MUJOCO_GL"] = "egl"
    command = [
        "play",
        TASK,
        "--checkpoint-file",
        str(checkpoint),
        "--motion-file",
        str(motion_file),
        "--num-envs",
        "1",
        "--device",
        "cpu",
        "--video",
        "True",
        "--video-length",
        str(args.video_length),
    ]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, env=env, check=False)
    print(f"RETURN CODE (play): {result.returncode}", flush=True)

    videos = list(checkpoint.parent.rglob("*.mp4"))
    if result.returncode == 0 and videos:
        latest = max(videos, key=lambda path: path.stat().st_mtime)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest, args.output)
        print(f"copied rendered MP4: {args.output}", flush=True)
        return 0

    print(
        "Rendering did not produce an MP4. Training/checkpoint files are unchanged.",
        file=sys.stderr,
        flush=True,
    )
    return result.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
