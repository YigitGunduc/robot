from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _euler_xyz_extrinsic_deg_to_xyzw(euler_deg: np.ndarray) -> np.ndarray:
    """Convert fixed-axis/extrinsic XYZ Euler degrees to xyzw quaternions.

    BONES-SEED documents root rotations as extrinsic XYZ Euler angles.  For
    column vectors this is Rz(z) @ Ry(y) @ Rx(x), i.e. the standard
    roll-pitch-yaw quaternion below.
    """
    r, p, y = np.deg2rad(euler_deg).T * 0.5
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yy = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.stack([x, yy, z, w], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
    return q


def load_native_bones(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return root position [m], root euler [deg], 29 joint angles [rad]."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc
        rows = [row for row in reader if row]

    names = [h.strip() for h in header]
    required = [
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
    ]
    missing = [x for x in required if x not in names]
    if missing:
        raise ValueError(
            f"{path} does not look like native BONES-SEED CSV; missing {missing}"
        )
    data = np.asarray(rows, dtype=np.float64)
    index = {name: i for i, name in enumerate(names)}
    root_cm = data[:, [index[x] for x in required[:3]]]
    euler_deg = data[:, [index[x] for x in required[3:]]]

    joint_cols = [i for i, name in enumerate(names) if name.endswith("_dof")]
    if len(joint_cols) != 29:
        # Some releases have the same 29 joint columns but slightly different
        # suffixes. Fall back to every non-root, non-Frame column.
        protected = {"Frame", *required}
        joint_cols = [i for i, name in enumerate(names) if name not in protected]
    if len(joint_cols) != 29:
        raise ValueError(f"expected 29 G1 joint columns, found {len(joint_cols)} in {path}")

    root_m = root_cm / 100.0
    joints_rad = np.deg2rad(data[:, joint_cols])
    return root_m, euler_deg, joints_rad


def convert_file(src: Path, dst: Path) -> None:
    root_m, euler_deg, joints_rad = load_native_bones(src)
    quat_xyzw = _euler_xyz_extrinsic_deg_to_xyzw(euler_deg)
    out = np.concatenate([root_m, quat_xyzw, joints_rad], axis=1)
    if out.shape[1] != 36:
        raise AssertionError(f"expected 36 generalized coordinates, got {out.shape[1]}")
    if not np.isfinite(out).all():
        raise ValueError(f"non-finite values in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst, out, delimiter=",", fmt="%.9g")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert native BONES-SEED G1 CSV (cm/deg/Euler) to mjlab input CSV (m/rad/xyzw)."
    )
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    convert_file(args.src, args.dst)
    print(f"converted {args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
