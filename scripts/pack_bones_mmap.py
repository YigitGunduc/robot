from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gear_sonic_mjx.data_process.bones import (
    MotionClip,
    _finite_difference,
    resample_motion,
    resampled_frame_count,
)
from gear_sonic_mjx.data_process.fk_cache import MujocoFKCache
from gear_sonic_mjx.data_process.splits import SPLIT_NAMES, load_split_files
from gear_sonic_mjx.g1_parameters import SONIC_TRACKED_BODY_NAMES

FIELDS = [
    "root_pos",
    "root_quat_wxyz",
    "joint_pos",
    "joint_vel",
    "body_pos",
    "body_quat_wxyz",
    "body_linvel",
    "body_angvel",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pack FK-augmented BONES clips into vectorized memory-mapped arrays"
    )
    ap.add_argument(
        "--motions",
        required=True,
        help="directory containing preprocessed/FK-augmented .npz clips",
    )
    ap.add_argument("--output", required=True, help="new packed directory")
    ap.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="pack directly at SONIC control/reference FPS",
    )
    ap.add_argument(
        "--nvidia-upper-body-augment",
        action="store_true",
        help="apply NVIDIA-style load-time upper-body donor augmentation to eligible planner-generated motion names",
    )
    ap.add_argument("--upper-body-prob", type=float, default=0.5)
    ap.add_argument(
        "--mjcf",
        required=True,
        help="exact G1 MJCF used to recompute physically consistent FK after 50-Hz resampling",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--split-manifest", help="JSON manifest created by scripts/split_bones.py"
    )
    ap.add_argument("--split", choices=SPLIT_NAMES, help="manifest split to pack")
    args = ap.parse_args()

    src, dst = Path(args.motions), Path(args.output)
    if bool(args.split_manifest) != bool(args.split):
        raise ValueError("--split-manifest and --split must be provided together")
    files = (
        load_split_files(src, args.split_manifest, args.split)
        if args.split_manifest
        else sorted(
            p
            for p in src.rglob("*.npz")
            if p.name not in {"_manifest.npz", "_packed_metadata.npz"}
        )
    )
    if not files:
        raise FileNotFoundError(f"no motion clips under {src}")
    rng = np.random.default_rng(args.seed)

    # First pass: target lengths and body contract. This only reads each compressed source once to
    # compute exact resampled lengths; the expensive result is then written sequentially below.
    lengths, names = [], []
    body_names = None
    for p in tqdm(files, desc="scan"):
        clip = MotionClip.load_npz(p)
        n = resampled_frame_count(clip.num_frames, clip.fps, args.fps)
        lengths.append(n)
        names.append(clip.name)
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

    if body_names is None:
        body_names = tuple(SONIC_TRACKED_BODY_NAMES)
    has_fk = True
    if body_names is not None:
        missing = [n for n in SONIC_TRACKED_BODY_NAMES if n not in body_names]
        if missing:
            raise ValueError(f"FK cache is missing canonical SONIC bodies: {missing}")

    # Preserve the source body ordering; the training task resolves the 14 canonical names by name.
    shapes = {
        "root_pos": (total, 3),
        "root_quat_wxyz": (total, 4),
        "joint_pos": (total, 29),
        "joint_vel": (total, 29),
    }
    if has_fk:
        nb = len(body_names)
        shapes.update(
            {
                "body_pos": (total, nb, 3),
                "body_quat_wxyz": (total, nb, 4),
                "body_linvel": (total, nb, 3),
                "body_angvel": (total, nb, 3),
            }
        )
    arrays = {
        k: np.lib.format.open_memmap(
            dst / f"{k}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        for k, shape in shapes.items()
    }
    fk = MujocoFKCache(args.mjcf, list(body_names))

    for i, p in enumerate(tqdm(files, desc="pack")):
        clip = resample_motion(MotionClip.load_npz(p), args.fps)
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
        # FK must be generated from the final 50-Hz root/joint trajectory. Interpolating a
        # 30-Hz body cache independently would not equal FK(resampled joint angles).
        clip.body_names = None
        clip.body_pos = None
        clip.body_quat_wxyz = None
        clip.body_linvel = None
        clip.body_angvel = None
        fk.augment(clip)
        if clip.num_frames != lengths[i]:
            raise RuntimeError(
                f"length changed unexpectedly for {p}: {clip.num_frames} != {lengths[i]}"
            )
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        for key, destination in arrays.items():
            value = getattr(clip, key)
            if value is None:
                raise ValueError(f"{p}: missing {key}; run augment_bones_fk.py first")
            destination[sl] = value
    for arr in arrays.values():
        arr.flush()
    np.savez_compressed(
        dst / "_packed_metadata.npz",
        fps=np.asarray(args.fps, np.float32),
        lengths=lengths,
        offsets=offsets,
        names=np.asarray(names, dtype="U512"),
        source_relpaths=np.asarray(
            [p.relative_to(src).as_posix() for p in files], dtype="U1024"
        ),
        body_names=np.asarray(body_names or (), dtype="U128"),
    )
    gib = sum(Path(dst / f"{k}.npy").stat().st_size for k in arrays) / (1024**3)
    print(
        f"packed {len(files)} clips, {total:,} frames at {args.fps:g} Hz, {gib:.1f} GiB"
    )


if __name__ == "__main__":
    main()
