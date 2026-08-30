from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

# Exact public blacklist from NVIDIA's current SONIC BONES filtering script.
NVIDIA_REJECT = (
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on",
    "seat", "table", "_sit_", "sit_", "ladder", "crutch", "_bed_", "_ride_",
    "scooter", "stepdown", "acrobatics_", "box_hspu", "cartwheel", "50cm_box_",
    "on_box", "fall_from", "handstand_ff_", "on_1m", "form_box", "off_1m",
    "230m", "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll",
    "kozak_dance", "stair", "handstand", "box_jump", "monkey_jump", "safety_roll",
    "box_dips", "walking_on_edge", "push_obstacle",
)

# Extra conservative V1 exclusions.  We want boring free-space locomotion only.
V1_REJECT = (
    "inj", "injured", "jump", "hop", "leap", "flip", "roll", "fall", "crawl", "kneel", "dance",
    "kick", "punch", "throw", "catch", "carry", "crate", "box", "object", "push",
    "pull", "lift", "car", "vehicle", "door", "weapon", "ball", "sport", "vault",
    "rope", "bench", "sofa", "desk", "shelf", "edge",
)

GROUPS = {
    "stand": ("stand", "standing", "idle"),
    "walk": ("walk", "walking"),
    "turn": ("turn", "pivot", "start", "stop"),
    "crouch": ("crouch", "squat", "squatting"),
    "jog": ("jog", "jogging"),
}

DEFAULT_QUOTAS = {
    "stand": 150,
    "walk": 700,
    "turn": 200,
    "crouch": 300,
    "jog": 500,
}


def classify(name: str) -> str | None:
    s = name.lower()
    if any(k in s for k in NVIDIA_REJECT) or any(k in s for k in V1_REJECT):
        return None
    # Specific groups first so "walking ... turn" becomes turn, not plain walk.
    for group in ("stand", "turn", "crouch", "jog", "walk"):
        if any(k in s for k in GROUPS[group]):
            return group
    return None


def _load_motion_for_difficulty(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load native BONES or already-converted generalized-coordinate CSV.

    Returns root position in metres, yaw in radians, and joint angles in radians.
    """
    # Native BONES format has a header and stores cm/degrees.
    with path.open(newline="") as f:
        first = f.readline().strip().split(",")
    if "root_translateX" in first:
        from .bones_to_mjlab_csv import load_native_bones

        root, euler_deg, joints = load_native_bones(path)
        yaw = np.unwrap(np.deg2rad(euler_deg[:, 2]))
        return root, yaw, joints

    # Converted mjlab input: xyz [m], quaternion xyzw, 29 joints [rad].
    x = np.loadtxt(path, delimiter=",")
    x = np.atleast_2d(x)
    if x.shape[1] != 36:
        raise ValueError(f"expected 36 generalized coordinates, got {x.shape[1]}")
    root = x[:, :3]
    q = x[:, 3:7]
    xx, yy, zz, ww = [q[:, i] for i in range(4)]
    yaw = np.unwrap(np.arctan2(2.0 * (ww * zz + xx * yy), 1.0 - 2.0 * (yy * yy + zz * zz)))
    return root, yaw, x[:, 7:]


def difficulty(path: Path, fps: float) -> float:
    """Cheap physics proxy used only to rank motions within one skill bucket."""
    try:
        root, yaw, joints = _load_motion_for_difficulty(path)
        if root.shape[0] < 5:
            return math.inf
        dt = 1.0 / fps
        qd = np.diff(joints, axis=0) / dt
        qdd = np.diff(qd, axis=0) / dt if len(qd) > 1 else np.zeros_like(qd)
        yaw_rate = np.diff(yaw) / dt
        root_vel = np.diff(root[:, :2], axis=0) / dt

        p95_qd = float(np.percentile(np.abs(qd), 95)) if qd.size else 0.0
        p95_qdd = float(np.percentile(np.abs(qdd), 95)) if qdd.size else 0.0
        p95_yaw = float(np.percentile(np.abs(yaw_rate), 95)) if yaw_rate.size else 0.0
        p95_speed = float(np.percentile(np.linalg.norm(root_vel, axis=1), 95)) if root_vel.size else 0.0
        z_range = float(np.ptp(root[:, 2]))

        return (
            p95_qd / 6.0
            + p95_qdd / 60.0
            + p95_yaw / 2.5
            + p95_speed / 3.0
            + z_range / 0.40
        )
    except Exception:
        return math.inf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="BONES robot CSV root")
    ap.add_argument("--out", type=Path, default=Path("selected_bones.json"))
    ap.add_argument("--input-fps", type=float, default=120.0)
    ap.add_argument("--max-stand", type=int, default=DEFAULT_QUOTAS["stand"])
    ap.add_argument("--max-walk", type=int, default=DEFAULT_QUOTAS["walk"])
    ap.add_argument("--max-turn", type=int, default=DEFAULT_QUOTAS["turn"])
    ap.add_argument("--max-crouch", type=int, default=DEFAULT_QUOTAS["crouch"])
    ap.add_argument("--max-jog", type=int, default=DEFAULT_QUOTAS["jog"])
    args = ap.parse_args()

    quotas = {g: getattr(args, f"max_{g}") for g in GROUPS}
    candidates: dict[str, list[dict]] = {g: [] for g in GROUPS}

    paths = sorted(args.root.rglob("*.csv"))
    if not paths:
        raise SystemExit(f"No CSV files found under {args.root}")

    for i, path in enumerate(paths, 1):
        group = classify(str(path.relative_to(args.root)))
        if group is None:
            continue
        score = difficulty(path, args.input_fps)
        if not math.isfinite(score):
            continue
        candidates[group].append({"path": str(path.resolve()), "difficulty": score})
        if i % 1000 == 0:
            print(f"scanned {i}/{len(paths)}")

    selected: dict[str, list[dict]] = {}
    for group, items in candidates.items():
        items.sort(key=lambda x: x["difficulty"])
        selected[group] = items[: quotas[group]]
        print(f"{group:>7}: candidates={len(items):5d}, selected={len(selected[group]):4d}")

    payload = {
        "input_fps": args.input_fps,
        "groups": selected,
        "ordered_curriculum": ["stand", "walk", "turn", "crouch", "jog"],
        "notes": "NVIDIA blacklist + conservative V1 semantic filter + within-skill kinematic difficulty ranking",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
