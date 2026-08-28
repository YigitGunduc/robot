from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic_mjx.data_process.bones import MotionClip
from gear_sonic_mjx.g1_parameters import G1_MUJOCO_JOINT_NAMES


def _quat_angvel(q_wxyz: np.ndarray, fps: float) -> np.ndarray:
    """Finite-difference world-frame angular velocity for body quaternions [T,N,4]."""
    t, n, _ = q_wxyz.shape
    out = np.zeros((t, n, 3), np.float32)
    if t < 2:
        return out
    xyzw = q_wxyz[..., [1, 2, 3, 0]]
    r = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(t, n, 3, 3)
    dt = 1.0 / fps
    for i in range(t - 1):
        rel = np.einsum("nij,njk->nik", np.transpose(r[i], (0, 2, 1)), r[i + 1])
        rotvec_local = Rotation.from_matrix(rel).as_rotvec()
        out[i] = np.einsum("nij,nj->ni", r[i], rotvec_local / dt).astype(np.float32)
    out[-1] = out[-2]
    return out


def augment_clip_with_mujoco_fk(clip: MotionClip, mjcf_path: str | Path, body_names: list[str]) -> MotionClip:
    """Cache reference body kinematics once so GPU rollouts do not run reference FK every step."""
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("Install mujoco to build FK caches") from exc
    m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    d = mujoco.MjData(m)
    free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    if len(free) != 1:
        raise ValueError("Expected one free joint")
    root_qadr = int(m.jnt_qposadr[free[0]])
    qadr = []
    for name in G1_MUJOCO_JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(name)
        qadr.append(int(m.jnt_qposadr[jid]))
    bids = []
    for name in body_names:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"MJCF missing body {name!r}")
        bids.append(bid)

    body_pos = np.empty((clip.num_frames, len(bids), 3), np.float32)
    body_quat = np.empty((clip.num_frames, len(bids), 4), np.float32)
    for i in range(clip.num_frames):
        d.qpos[root_qadr:root_qadr+3] = clip.root_pos[i]
        d.qpos[root_qadr+3:root_qadr+7] = clip.root_quat_wxyz[i]
        d.qpos[qadr] = clip.joint_pos[i]
        mujoco.mj_forward(m, d)
        body_pos[i] = d.xpos[bids]
        body_quat[i] = d.xquat[bids]
    body_linvel = np.gradient(body_pos, 1.0 / clip.fps, axis=0, edge_order=1).astype(np.float32)
    body_angvel = _quat_angvel(body_quat, clip.fps)
    clip.body_names = tuple(body_names)
    clip.body_pos = body_pos
    clip.body_quat_wxyz = body_quat
    clip.body_linvel = body_linvel
    clip.body_angvel = body_angvel
    return clip
