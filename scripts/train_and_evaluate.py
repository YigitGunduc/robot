#!/usr/bin/env python3
"""Run the small SONIC-Lite G1 train/evaluate/render workflow.

This script only orchestrates the existing data tools and mjlab commands. It
does not change the model, environment, rewards, or training logic.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


TASK = "Mjlab-SonicLite-Tracking-Flat-Unitree-G1"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def select_clips(args: argparse.Namespace, repo: Path) -> Path:
    selected = args.work_dir / "selected_bones.json"
    run(
        [
            sys.executable,
            "-m",
            "sonic_lite_g1.data.select_bones",
            "--root",
            str(args.csv_root),
            "--out",
            str(selected),
            "--max-stand",
            str(args.max_stand),
            "--max-walk",
            str(args.max_walk),
            "--max-turn",
            str(args.max_turn),
            "--max-crouch",
            str(args.max_crouch),
            "--max-jog",
            str(args.max_jog),
        ],
        cwd=repo,
    )
    return selected


def convert_and_pack(args: argparse.Namespace, selected_file: Path, repo: Path) -> Path:
    payload = json.loads(selected_file.read_text())
    selected = [item["path"] for group in payload["groups"].values() for item in group]
    if not selected:
        raise RuntimeError("The selector returned no clips")

    csv_dir = args.work_dir / "mjlab_csv"
    npz_dir = args.work_dir / "converted_npz"
    csv_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(selected):
        source_path = Path(source)
        stem = f"{index:04d}_{source_path.stem}"
        converted_csv = csv_dir / f"{stem}.csv"
        converted_npz = npz_dir / f"{stem}.npz"

        if not converted_csv.exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "sonic_lite_g1.data.bones_to_mjlab_csv",
                    str(source_path),
                    str(converted_csv),
                ],
                cwd=repo,
            )

        if not converted_npz.exists():
            env = os.environ.copy()
            env.setdefault("WANDB_MODE", "offline")
            env.setdefault("MUJOCO_GL", "egl")
            run(
                [
                    sys.executable,
                    "-m",
                    "mjlab.scripts.csv_to_npz",
                    "--input-file",
                    str(converted_csv),
                    "--output-name",
                    stem,
                    "--input-fps",
                    str(args.input_fps),
                    "--output-fps",
                    "50",
                    "--render",
                    "False",
                ],
                cwd=repo,
                env=env,
            )
            generated = Path("/tmp/motion.npz")
            if not generated.exists():
                raise FileNotFoundError(
                    "mjlab conversion completed but /tmp/motion.npz was not found"
                )
            shutil.copy2(generated, converted_npz)

    packed = args.work_dir / "stand_walk.npz"
    run(
        [
            sys.executable,
            "-m",
            "sonic_lite_g1.data.pack_motions",
            "--dir",
            str(npz_dir),
            "--out",
            str(packed),
        ],
        cwd=repo,
    )
    return packed


def newest_checkpoint(repo: Path) -> Path:
    checkpoints = list((repo / "logs").rglob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("No model_*.pt checkpoint found under logs/")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def summarize_tensorboard(repo: Path, output: Path) -> None:
    """Write the last/min/max scalar values from the training event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard is not installed; skipping analytical summary")
        return

    event_files = list((repo / "logs").rglob("events.out.tfevents.*"))
    if not event_files:
        print("No TensorBoard event files found; skipping analytical summary")
        return

    summary: dict[str, dict[str, float | int]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values = [event.value for event in accumulator.Scalars(tag)]
            if not values:
                continue
            summary[tag] = {
                "steps": len(values),
                "first": float(values[0]),
                "last": float(values[-1]),
                "min": float(min(values)),
                "max": float(max(values)),
            }

    output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote analytical summary: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("data/colab_run"))
    parser.add_argument("--max-stand", type=int, default=30)
    parser.add_argument("--max-walk", type=int, default=100)
    parser.add_argument("--max-turn", type=int, default=0)
    parser.add_argument("--max-crouch", type=int, default=0)
    parser.add_argument("--max-jog", type=int, default=0)
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--video-length", type=int, default=600)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.csv_root = args.dataset_root / "g1" / "csv"
    if not args.csv_root.exists():
        raise SystemExit(
            f"Missing extracted BONES CSV directory: {args.csv_root}\n"
            "Extract g1.tar.gz under the dataset root first."
        )

    selected = select_clips(args, repo)
    motion_file = convert_and_pack(args, selected, repo)

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    train_command = [
        "train",
        TASK,
        "--env.commands.motion.motion-file",
        str(motion_file),
        "--env.scene.num-envs",
        str(args.num_envs),
        "--agent.max-iterations",
        str(args.max_iterations),
        "--video",
        "True",
    ]
    run(train_command, cwd=repo, env=env)

    checkpoint = newest_checkpoint(repo)
    print("checkpoint:", checkpoint)
    summarize_tensorboard(repo, args.work_dir / "training_metrics.json")

    play_command = [
        "play",
        TASK,
        "--checkpoint-file",
        str(checkpoint),
        "--motion-file",
        str(motion_file),
        "--num-envs",
        "1",
        "--video",
        "True",
        "--video-length",
        str(args.video_length),
    ]
    run(play_command, cwd=repo, env=env)

    videos = list((repo / "logs").rglob("*.mp4"))
    if videos:
        latest_video = max(videos, key=lambda path: path.stat().st_mtime)
        target = args.work_dir / "trained_action.mp4"
        shutil.copy2(latest_video, target)
        print(f"copied rendered MP4: {target}")
    else:
        print("No MP4 found under logs/")


if __name__ == "__main__":
    main()
