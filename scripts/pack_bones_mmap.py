from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gear_sonic_mjx.data_process.bones import MotionClip, resample_motion, _finite_difference
from gear_sonic_mjx.data_process.fk_cache import augment_clip_with_mujoco_fk
from gear_sonic_mjx.g1_parameters import SONIC_TRACKED_BODY_NAMES


FIELDS = [
    "root_pos", "root_quat_wxyz", "joint_pos", "joint_vel",
    "body_pos", "body_quat_wxyz", "body_linvel", "body_angvel",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack FK-augmented BONES clips into vectorized memory-mapped arrays")
    ap.add_argument("--motions", required=True, help="directory containing preprocessed/FK-augmented .npz clips")
    ap.add_argument("--output", required=True, help="new packed directory")
    ap.add_argument("--fps", type=float, default=50.0, help="pack directly at SONIC control/reference FPS")
    ap.add_argument("--nvidia-upper-body-augment", action="store_true", help="apply NVIDIA-style load-time upper-body donor augmentation to eligible planner-generated motion names")
    ap.add_argument("--upper-body-prob", type=float, default=0.5)
    ap.add_argument("--mjcf", help="required for upper-body augmentation so body FK can be recomputed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.motions), Path(args.output)
    files = sorted(p for p in src.rglob("*.npz") if p.name not in {"_manifest.npz", "_packed_metadata.npz"})
    if not files:
        raise FileNotFoundError(f"no motion clips under {src}")
    if args.nvidia_upper_body_augment and not args.mjcf:
        raise ValueError("--nvidia-upper-body-augment requires --mjcf to recompute consistent body FK")
    rng = np.random.default_rng(args.seed)

    # First pass: target lengths and body contract. This only reads each compressed source once to
    # compute exact resampled lengths; the expensive result is then written sequentially below.
    lengths, names = [], []
    body_names = None
    for p in tqdm(files, desc="scan"):
        clip = MotionClip.load_npz(p)
        n = max(2, int(round(clip.duration * args.fps)) + 1)
        lengths.append(n); names.append(clip.name)
        if clip.body_names is not None:
            if body_names is None:
                body_names = tuple(clip.body_names)
            elif tuple(clip.body_names) != body_names:
                raise ValueError(f"body_names mismatch in {p}")
    prefixes = ("2025", "walking_2025", "running_2025", "slow_walk_2025")
    donor_indices = [i for i, name in enumerate(names) if not name.startswith(prefixes)]
    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total = int(offsets[-1])
    dst.mkdir(parents=True, exist_ok=True)

    has_fk = body_names is not None
    if has_fk:
        missing = [n for n in SONIC_TRACKED_BODY_NAMES if n not in body_names]
        if missing:
            raise ValueError(f"FK cache is missing canonical SONIC bodies: {missing}")

    # Preserve the source body ordering; the training task resolves the 14 canonical names by name.
    shapes = {
        "root_pos": (total, 3), "root_quat_wxyz": (total, 4),
        "joint_pos": (total, 29), "joint_vel": (total, 29),
    }
    if has_fk:
        nb = len(body_names)
        shapes.update({
            "body_pos": (total, nb, 3), "body_quat_wxyz": (total, nb, 4),
            "body_linvel": (total, nb, 3), "body_angvel": (total, nb, 3),
        })
    arrays = {k: np.lib.format.open_memmap(dst / f"{k}.npy", mode="w+", dtype=np.float32, shape=shape) for k, shape in shapes.items()}

    for i, p in enumerate(tqdm(files, desc="pack")):
        clip = resample_motion(MotionClip.load_npz(p), args.fps)
        upper_augmented = False
        if (
            args.nvidia_upper_body_augment
            and clip.name.startswith(prefixes)
            and donor_indices
            and rng.random() < args.upper_body_prob
        ):
            donor_idx = int(rng.choice(donor_indices))
            donor = resample_motion(MotionClip.load_npz(files[donor_idx]), args.fps)
            T = clip.num_frames
            if donor.num_frames >= T:
                start = int(rng.integers(0, donor.num_frames - T + 1))
                didx = np.arange(start, start + T)
            else:
                # NVIDIA ping-pongs short donor sequences to match the navigation clip length.
                if donor.num_frames <= 1:
                    didx = np.zeros(T, dtype=np.int64)
                else:
                    period = 2 * donor.num_frames - 2
                    x = np.arange(T, dtype=np.int64) % period
                    didx = np.where(x < donor.num_frames, x, period - x)
            clip.joint_pos = clip.joint_pos.copy()
            clip.joint_pos[:, 12:] = donor.joint_pos[didx, 12:]
            clip.joint_vel = _finite_difference(clip.joint_pos, clip.fps)
            # Mixed upper-body pose changes body kinematics; recompute them from the exact MJCF.
            clip.body_names = None; clip.body_pos = None; clip.body_quat_wxyz = None
            clip.body_linvel = None; clip.body_angvel = None
            augment_clip_with_mujoco_fk(clip, args.mjcf, list(body_names or SONIC_TRACKED_BODY_NAMES))
            upper_augmented = True
        if clip.num_frames != lengths[i]:
            raise RuntimeError(f"length changed unexpectedly for {p}: {clip.num_frames} != {lengths[i]}")
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        for key in arrays:
            value = getattr(clip, key)
            if value is None:
                raise ValueError(f"{p}: missing {key}; run augment_bones_fk.py first")
            arrays[key][sl] = value
    for arr in arrays.values():
        arr.flush()
    np.savez_compressed(
        dst / "_packed_metadata.npz",
        fps=np.asarray(args.fps, np.float32),
        lengths=lengths,
        offsets=offsets,
        names=np.asarray(names, dtype="U512"),
        body_names=np.asarray(body_names or (), dtype="U128"),
    )
    gib = sum(Path(dst / f"{k}.npy").stat().st_size for k in arrays) / (1024**3)
    print(f"packed {len(files)} clips, {total:,} frames at {args.fps:g} Hz, {gib:.1f} GiB")


if __name__ == "__main__":
    main()
