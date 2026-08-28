from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gear_sonic_mjx.g1_parameters import BONES_CSV_JOINT_NAMES


NVIDIA_FILTER_KEYWORDS = [
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table",
    "_sit_", "sit_", "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown",
    "acrobatics_", "box_hspu", "cartwheel", "50cm_box_", "on_box", "fall_from",
    "handstand_ff_", "on_1m", "form_box", "off_1m", "230m", "jump_over_obstacle_",
    "lift_crate_come_up_", "jump_to_shoulder_roll", "kozak_dance", "stair", "handstand",
    "box_jump", "monkey_jump", "safety_roll", "box_dips", "walking_on_edge", "push_obstacle",
]


@dataclass
class MotionClip:
    name: str
    fps: float
    root_pos: np.ndarray      # [T,3] meters
    root_quat_wxyz: np.ndarray  # [T,4]
    joint_pos: np.ndarray     # [T,29] radians, MuJoCo order
    joint_vel: np.ndarray     # [T,29] rad/s, MuJoCo order
    body_names: tuple[str, ...] | None = None
    body_pos: np.ndarray | None = None
    body_quat_wxyz: np.ndarray | None = None
    body_linvel: np.ndarray | None = None
    body_angvel: np.ndarray | None = None

    def __post_init__(self) -> None:
        t = self.joint_pos.shape[0]
        assert self.root_pos.shape == (t, 3)
        assert self.root_quat_wxyz.shape == (t, 4)
        assert self.joint_pos.shape == (t, 29)
        assert self.joint_vel.shape == (t, 29)

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / float(self.fps)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            name=np.asarray(self.name),
            fps=np.asarray(self.fps, dtype=np.float32),
            root_pos=self.root_pos.astype(np.float32),
            root_quat_wxyz=self.root_quat_wxyz.astype(np.float32),
            joint_pos=self.joint_pos.astype(np.float32),
            joint_vel=self.joint_vel.astype(np.float32),
        )
        if self.body_names is not None:
            payload["body_names"] = np.asarray(self.body_names, dtype="U128")
        for key in ["body_pos", "body_quat_wxyz", "body_linvel", "body_angvel"]:
            value = getattr(self, key)
            if value is not None:
                payload[key] = value.astype(np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> "MotionClip":
        d = np.load(path, allow_pickle=False)
        return cls(
            name=str(d["name"].item()),
            fps=float(d["fps"].item()),
            root_pos=d["root_pos"],
            root_quat_wxyz=d["root_quat_wxyz"],
            joint_pos=d["joint_pos"],
            joint_vel=d["joint_vel"],
            body_names=tuple(str(x) for x in d["body_names"].tolist()) if "body_names" in d else None,
            body_pos=d["body_pos"] if "body_pos" in d else None,
            body_quat_wxyz=d["body_quat_wxyz"] if "body_quat_wxyz" in d else None,
            body_linvel=d["body_linvel"] if "body_linvel" in d else None,
            body_angvel=d["body_angvel"] if "body_angvel" in d else None,
        )


def should_filter_out(name: str, extra_keywords: list[str] | None = None) -> bool:
    text = name.lower()
    keywords = NVIDIA_FILTER_KEYWORDS + [k.lower() for k in (extra_keywords or [])]
    return any(k in text for k in keywords)


def _finite_difference(q: np.ndarray, fps: float) -> np.ndarray:
    if len(q) < 2:
        return np.zeros_like(q, dtype=np.float32)
    # np.gradient gives centered differences internally and one-sided endpoints.
    return np.gradient(q, 1.0 / fps, axis=0, edge_order=1).astype(np.float32)


def resample_motion(clip: MotionClip, target_fps: float) -> MotionClip:
    if abs(target_fps - clip.fps) < 1e-6:
        return clip
    duration = clip.duration
    n = max(2, int(round(duration * target_fps)) + 1)
    old_t = np.arange(clip.num_frames, dtype=np.float64) / clip.fps
    new_t = np.linspace(0.0, duration, n, dtype=np.float64)

    root_pos = np.stack([np.interp(new_t, old_t, clip.root_pos[:, i]) for i in range(3)], axis=-1)
    joint_pos = np.stack([np.interp(new_t, old_t, clip.joint_pos[:, i]) for i in range(29)], axis=-1)

    # Slerp root orientation using scipy RotationSpline-like piecewise interpolation.
    rots = Rotation.from_quat(clip.root_quat_wxyz[:, [1, 2, 3, 0]])
    # Manual per interval slerp to avoid depending on scipy Slerp behavior across versions.
    from scipy.spatial.transform import Slerp
    root_xyzw = Slerp(old_t, rots)(new_t).as_quat().astype(np.float32)
    root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
    joint_pos = joint_pos.astype(np.float32)
    out = MotionClip(
        name=clip.name,
        fps=float(target_fps),
        root_pos=root_pos.astype(np.float32),
        root_quat_wxyz=root_wxyz,
        joint_pos=joint_pos,
        joint_vel=_finite_difference(joint_pos, target_fps),
        body_names=clip.body_names,
    )
    if clip.body_pos is not None:
        out.body_pos = np.stack([
            np.stack([np.interp(new_t, old_t, clip.body_pos[:, j, k]) for k in range(3)], axis=-1)
            for j in range(clip.body_pos.shape[1])
        ], axis=1).astype(np.float32)
        out.body_linvel = np.gradient(out.body_pos, 1.0 / target_fps, axis=0, edge_order=1).astype(np.float32)
    if clip.body_quat_wxyz is not None:
        qs = []
        from scipy.spatial.transform import Slerp
        for j in range(clip.body_quat_wxyz.shape[1]):
            rj = Rotation.from_quat(clip.body_quat_wxyz[:, j][:, [1,2,3,0]])
            qxyzw = Slerp(old_t, rj)(new_t).as_quat().astype(np.float32)
            qs.append(qxyzw[:, [3,0,1,2]])
        out.body_quat_wxyz = np.stack(qs, axis=1)
        # Angular velocity is optional; recompute in the FK-cache helper when exact body velocities matter.
    if clip.body_angvel is not None:
        out.body_angvel = np.stack([
            np.stack([np.interp(new_t, old_t, clip.body_angvel[:, j, k]) for k in range(3)], axis=-1)
            for j in range(clip.body_angvel.shape[1])
        ], axis=1).astype(np.float32)
    return out


def load_bones_seed_csv(path: str | Path, source_fps: float = 120.0) -> MotionClip:
    """Load the public Bones-SEED flat G1 CSV format.

    Expected columns are the same ones used by NVIDIA's converter:
    Frame, root_translate{X,Y,Z} [cm], root_rotate{X,Y,Z} [deg], and 29 *_dof columns [deg].
    """
    path = Path(path)
    df = pd.read_csv(path)
    required_root = [
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
    ]
    missing = [c for c in required_root if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing Bones-SEED columns: {missing}")

    # Prefer NVIDIA's documented names, but accept any 29 *_dof columns in file order.
    if all(c in df.columns for c in BONES_CSV_JOINT_NAMES):
        joint_cols = BONES_CSV_JOINT_NAMES
    else:
        joint_cols = [c for c in df.columns if c.endswith("_dof")]
        if len(joint_cols) != 29:
            raise ValueError(f"{path}: expected 29 *_dof columns, found {len(joint_cols)}")

    root_pos = df[["root_translateX", "root_translateY", "root_translateZ"]].to_numpy(np.float32) / 100.0
    euler = df[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy(np.float64)
    quat_xyzw = Rotation.from_euler("xyz", euler, degrees=True).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    q = np.deg2rad(df[joint_cols].to_numpy(np.float32)).astype(np.float32)
    qd = _finite_difference(q, source_fps)
    return MotionClip(path.stem, float(source_fps), root_pos, quat_wxyz, q, qd)


def discover_bones_csvs(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    yield from sorted(root.rglob("*.csv"))


def preprocess_bones_tree(
    source: str | Path,
    dest: str | Path,
    source_fps: float = 120.0,
    preprocess_fps: float = 30.0,
    extra_filter_keywords: list[str] | None = None,
    skip_filtered: bool = True,
) -> dict[str, int]:
    source = Path(source)
    dest = Path(dest)
    stats = {"total": 0, "written": 0, "filtered": 0, "failed": 0}
    manifest_paths: list[str] = []
    manifest_frames: list[int] = []
    manifest_fps: list[float] = []
    for csv in discover_bones_csvs(source):
        stats["total"] += 1
        rel = csv.relative_to(source)
        if skip_filtered and should_filter_out(str(rel), extra_filter_keywords):
            stats["filtered"] += 1
            continue
        try:
            clip = load_bones_seed_csv(csv, source_fps)
            clip = resample_motion(clip, preprocess_fps)
            out = (dest / rel).with_suffix(".npz")
            clip.save_npz(out)
            manifest_paths.append(str(out.relative_to(dest)))
            manifest_frames.append(clip.num_frames)
            manifest_fps.append(clip.fps)
            stats["written"] += 1
        except Exception:
            stats["failed"] += 1
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / "_manifest.npz",
        relpaths=np.asarray(manifest_paths, dtype="U512"),
        num_frames=np.asarray(manifest_frames, dtype=np.int32),
        fps=np.asarray(manifest_fps, dtype=np.float32),
    )
    return stats
