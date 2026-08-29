from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from gear_sonic_mjx.data_process.bones import MotionClip, should_filter_out
from gear_sonic_mjx.data_process.splits import mirror_group_key

EASY_MOTION_CLASSES = ("idle", "walk", "turn", "gesture")
_FORBIDDEN_EASY_TERMS = (
    "jog",
    "jump",
    "run",
    "crouch",
    "kneel",
    "crawl",
    "dance",
    "acrobat",
    "kick",
    "throw",
    "flip",
    "roll",
    "climb",
    "injured",
    "limp",
    "fall",
    "lunge",
    "squat",
    "boxing",
    "exercise",
    "handstand",
    "plank",
    "sit",
    "chair",
    "crutch",
    "bottle",
    "phone",
    "object",
    "mic_",
    "sword",
    "gun",
    "ball",
    "fast",
    "one_leg",
)
_SAFE_GESTURE_TERMS = (
    "wave",
    "point",
    "clap",
    "nod",
    "shake_head",
    "salute",
    "thumb",
    "shrug",
    "bow",
    "beckon",
    "raise_your_hand",
    "raise_hand",
    "hello",
    "goodbye",
    "listening",
    "yawn",
)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def easy_motion_class(row: Mapping[str, object]) -> str | None:
    """Classify conservative, upright motions for a small SONIC development run."""
    filename = _text(row.get("filename"))
    movement = _text(row.get("content_type_of_movement"))
    body = _text(row.get("content_body_position"))
    category = _text(row.get("category"))
    props = _text(row.get("content_props"))
    description = _text(row.get("content_short_description"))
    combined = f"{filename} {movement} {description}"
    if should_filter_out(filename) or any(
        term in combined for term in _FORBIDDEN_EASY_TERMS
    ):
        return None
    if body != "standing" or props not in {"", "0", "none", "nan"}:
        return None
    if int(row.get("content_complex_action", 0) or 0) != 0:
        return None
    neutral = row.get("is_neutral", 1)
    if not pd.isna(neutral) and float(neutral) != 1.0:
        return None
    frames = int(row.get("move_duration_frames", 0) or 0)
    if not 120 <= frames <= 1200:
        return None

    if (
        "idle" in filename
        and category in {"baseline", "basic locomotion neutral"}
        and movement in {"standing idle", "standing", "gesture", "action", "transition"}
    ):
        return "idle"
    if (
        category == "basic locomotion neutral"
        and movement == "walking"
        and ("walk_ff" in filename or "walk_forward" in filename)
    ):
        return "walk"
    if (
        category == "basic locomotion neutral"
        and movement in {"turning", "walking, turning"}
        and "270" not in filename
        and "360" not in filename
    ):
        return "turn"
    if (
        category in {"gestures", "communication", "looking and pointing"}
        and movement in {"gesture", "action", "pointing", "looking"}
        and any(term in combined for term in _SAFE_GESTURE_TERMS)
    ):
        return "gesture"
    return None


def _stable_order(seed: int, label: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{label}:{value}".encode()).digest()


def _read_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError(f"Unsupported BONES metadata format: {path}")


def build_easy_subset_manifest(
    motions: str | Path,
    metadata: str | Path,
    max_clips: int = 512,
    seed: int = 0,
) -> dict[str, object]:
    """Select a deterministic, mirror-safe easy-motion subset from a preprocessed library."""
    if max_clips < 20:
        raise ValueError(
            "easy subset needs at least 20 clips for non-empty data splits"
        )
    root = Path(motions)
    available = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.npz")
        if path.name not in {"_manifest.npz", "_packed_metadata.npz"}
    )
    if not available:
        raise FileNotFoundError(f"no preprocessed BONES clips under {root}")
    available_set = set(available)
    by_stem: dict[str, str] = {}
    ambiguous_stems: set[str] = set()
    for relative in available:
        stem = Path(relative).stem
        if stem in by_stem:
            ambiguous_stems.add(stem)
        else:
            by_stem[stem] = relative
    for stem in ambiguous_stems:
        by_stem.pop(stem, None)

    frame = _read_metadata(metadata)
    required = {
        "filename",
        "move_g1_path",
        "category",
        "content_type_of_movement",
        "content_body_position",
        "move_duration_frames",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"BONES metadata is missing easy-subset fields: {missing}")

    grouped: dict[str, dict[str, list[str]]] = {
        name: defaultdict(list) for name in EASY_MOTION_CLASSES
    }
    for row in frame.to_dict(orient="records"):
        motion_class = easy_motion_class(row)
        if motion_class is None:
            continue
        expected = Path(str(row["move_g1_path"])).with_suffix(".npz").as_posix()
        relative = (
            expected if expected in available_set else by_stem.get(Path(expected).stem)
        )
        if relative is None:
            continue
        # Select actor/mirror pairs independently so a very common source take cannot
        # dominate the small curriculum. The later split key still groups all actor
        # retargets of the same content take to prevent validation leakage.
        group = mirror_group_key(str(row["filename"]))
        if relative not in grouped[motion_class][group]:
            grouped[motion_class][group].append(relative)

    selected: list[str] = []
    selected_set: set[str] = set()
    class_by_relpath: dict[str, str] = {}
    selected_groups: set[tuple[str, str]] = set()
    quota = max_clips // len(EASY_MOTION_CLASSES)

    def add_group(
        motion_class: str, group: str, class_limit: int | None = None
    ) -> bool:
        paths = sorted(set(grouped[motion_class][group]))
        fresh = [path for path in paths if path not in selected_set]
        if not fresh or len(selected) + len(fresh) > max_clips:
            return False
        if class_limit is not None:
            count = sum(value == motion_class for value in class_by_relpath.values())
            if count + len(fresh) > class_limit:
                return False
        for path in fresh:
            selected.append(path)
            selected_set.add(path)
            class_by_relpath[path] = motion_class
        selected_groups.add((motion_class, group))
        return True

    for motion_class in EASY_MOTION_CLASSES:
        groups = sorted(
            grouped[motion_class],
            key=lambda group: _stable_order(seed, motion_class, group),
        )
        for group in groups:
            add_group(motion_class, group, quota)

    # Fill any quota shortfall in deterministic round-robin order. A single global
    # hash sort biases classes with many more candidate groups (walking in BONES).
    remaining = {
        motion_class: sorted(
            (
                group
                for group in grouped[motion_class]
                if (motion_class, group) not in selected_groups
            ),
            key=lambda group: _stable_order(seed, motion_class, group),
        )
        for motion_class in EASY_MOTION_CLASSES
    }
    positions = {motion_class: 0 for motion_class in EASY_MOTION_CLASSES}
    fill_order = sorted(
        EASY_MOTION_CLASSES,
        key=lambda motion_class: _stable_order(seed, "fill", motion_class),
    )
    while len(selected) < max_clips:
        progress = False
        for motion_class in fill_order:
            groups = remaining[motion_class]
            while positions[motion_class] < len(groups):
                group = groups[positions[motion_class]]
                positions[motion_class] += 1
                if add_group(motion_class, group):
                    progress = True
                    break
            if len(selected) >= max_clips:
                break
        if not progress:
            break

    if len(selected) < 20:
        raise RuntimeError(
            f"easy-motion filters selected only {len(selected)} available clips"
        )
    counts = {
        motion_class: sum(value == motion_class for value in class_by_relpath.values())
        for motion_class in EASY_MOTION_CLASSES
    }
    return {
        "version": 1,
        "preset": "easy",
        "seed": int(seed),
        "max_clips": int(max_clips),
        "selected_relpaths": selected,
        "class_by_relpath": class_by_relpath,
        "class_counts": counts,
    }


def materialize_subset(
    motions: str | Path, output: str | Path, manifest: Mapping[str, object]
) -> dict[str, int]:
    """Hard-link a selected subset on one filesystem, copying only when linking is unavailable."""
    source_root, output_root = Path(motions), Path(output)
    relpaths = [str(path) for path in manifest["selected_relpaths"]]
    source_manifest_path = source_root / "_manifest.npz"
    source_metadata: dict[str, tuple[int, float]] = {}
    if source_manifest_path.is_file():
        source_manifest = np.load(source_manifest_path, allow_pickle=False)
        source_metadata = {
            str(path): (int(frames), float(fps))
            for path, frames, fps in zip(
                source_manifest["relpaths"].tolist(),
                source_manifest["num_frames"].tolist(),
                source_manifest["fps"].tolist(),
                strict=True,
            )
        }
    output_paths: list[str] = []
    output_frames: list[int] = []
    output_fps: list[float] = []
    linked = copied = reused = 0
    for relative in relpaths:
        source = source_root / relative
        destination = output_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if destination.stat().st_size != source.stat().st_size:
                raise RuntimeError(
                    f"existing subset file differs from source: {destination}"
                )
            reused += 1
        else:
            try:
                os.link(source, destination)
                linked += 1
            except OSError:
                shutil.copy2(source, destination)
                copied += 1
        if relative in source_metadata:
            frames, fps = source_metadata[relative]
        else:
            clip = MotionClip.load_npz(source)
            frames, fps = clip.num_frames, clip.fps
        output_paths.append(relative)
        output_frames.append(frames)
        output_fps.append(fps)
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "_manifest.npz",
        relpaths=np.asarray(output_paths, dtype="U512"),
        num_frames=np.asarray(output_frames, dtype=np.int32),
        fps=np.asarray(output_fps, dtype=np.float32),
    )
    return {
        "clips": len(output_paths),
        "linked": linked,
        "copied": copied,
        "reused": reused,
    }


def save_subset_manifest(manifest: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
