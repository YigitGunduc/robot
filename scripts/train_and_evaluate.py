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

import numpy as np


TASK = "Mjlab-SonicLite-Tracking-Flat-Unitree-G1"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    print(f"RETURN CODE ({command[0]}): {return_code}", flush=True)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


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
    selected_npz_files: list[Path] = []

    for index, source in enumerate(selected):
        source_path = Path(source)
        cached_npz = next(
            (
                candidate
                for candidate in args.work_dir.rglob("*.npz")
                if candidate.name.endswith(f"_{source_path.stem}.npz")
                or candidate.name == f"{source_path.stem}.npz"
            ),
            None,
        )
        if cached_npz is not None:
            print(f"reusing cached NPZ: {cached_npz}", flush=True)
            selected_npz_files.append(cached_npz)
            continue
        stem = f"{index:04d}_{source_path.stem}"
        cached_csv = next(
            (
                candidate
                for candidate in args.work_dir.rglob("*.csv")
                if candidate.name.endswith(f"_{source_path.stem}.csv")
                or candidate.name == f"{source_path.stem}.csv"
            ),
            None,
        )
        converted_csv = cached_csv or (csv_dir / f"{stem}.csv")
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
            generated = Path("/tmp/motion.npz")
            generated.unlink(missing_ok=True)
            try:
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
            except subprocess.CalledProcessError:
                # mjlab writes the local NPZ before attempting its optional W&B
                # upload. Keep using the local artifact when that upload fails.
                if not generated.exists():
                    raise
                print(
                    "WARNING: mjlab conversion returned nonzero after creating "
                    f"{generated}; continuing with the local NPZ",
                    flush=True,
                )
            if not generated.exists():
                raise FileNotFoundError(
                    "mjlab conversion completed but /tmp/motion.npz was not found"
                )
            shutil.copy2(generated, converted_npz)
        selected_npz_files.append(converted_npz)

    packed = args.work_dir / "stand_walk.npz"
    selected_npz_list = args.work_dir / "selected_npz.txt"
    selected_npz_list.write_text("\n".join(str(path) for path in selected_npz_files) + "\n")
    run(
        [
            sys.executable,
            "-m",
            "sonic_lite_g1.data.pack_motions",
            "--list",
            str(selected_npz_list),
            "--out",
            str(packed),
        ],
        cwd=repo,
    )
    return packed


def reuse_clean_packed_motion(args: argparse.Namespace, selected_file: Path) -> Path | None:
    """Filter an existing packed motion file without rerunning FK conversion."""
    packed = args.work_dir / "stand_walk.npz"
    manifest = packed.with_suffix(packed.suffix + ".manifest.json")
    if not packed.exists() or not manifest.exists():
        return None

    selected_payload = json.loads(selected_file.read_text())
    selected_stems = {
        Path(item["path"]).stem
        for group in selected_payload["groups"].values()
        for item in group
    }
    source_files = json.loads(manifest.read_text())["source_files"]
    keep: list[int] = []
    for index, source_file in enumerate(source_files):
        stem = Path(source_file).stem
        if "inj" in stem.lower() or "injured" in stem.lower():
            continue
        if any(stem == selected or stem.endswith(f"_{selected}") for selected in selected_stems):
            keep.append(index)

    if len(keep) != len(selected_stems):
        print(
            f"existing packed motion has {len(keep)}/{len(selected_stems)} clean selected clips; "
            "falling back to per-clip conversion",
            flush=True,
        )
        return None

    with np.load(packed, allow_pickle=False) as source:
        starts = np.asarray(source["clip_starts"], dtype=np.int64)
        lengths = np.asarray(source["clip_lengths"], dtype=np.int64)
        payload: dict[str, np.ndarray] = {}
        for key in source.files:
            if key in {"clip_starts", "clip_lengths"}:
                continue
            chunks = [
                np.asarray(source[key])[starts[i] : starts[i] + lengths[i]]
                for i in keep
            ]
            payload[key] = np.concatenate(chunks, axis=0)

    new_starts: list[int] = []
    cursor = 0
    new_lengths: list[int] = []
    for index in keep:
        new_starts.append(cursor)
        new_lengths.append(int(lengths[index]))
        cursor += int(lengths[index])
    payload["clip_starts"] = np.asarray(new_starts, dtype=np.int64)
    payload["clip_lengths"] = np.asarray(new_lengths, dtype=np.int64)

    clean_packed = args.work_dir / "clean_stand_walk.npz"
    np.savez_compressed(clean_packed, **payload)
    clean_packed.with_suffix(clean_packed.suffix + ".manifest.json").write_text(
        json.dumps(
            {
                "source_files": [source_files[i] for i in keep],
                "num_clips": len(keep),
                "num_frames": cursor,
                "filtered_from": str(packed),
            },
            indent=2,
        )
    )
    print(f"reused {len(keep)} clean clips from existing packed motion: {clean_packed}")
    return clean_packed


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
    parser.add_argument("--num-envs", type=int, default=256)
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
    motion_file = reuse_clean_packed_motion(args, selected)
    if motion_file is None:
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
        "False",
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
