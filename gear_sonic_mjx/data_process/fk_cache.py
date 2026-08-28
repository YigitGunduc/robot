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


class MujocoFKCache:
    """Reusable CPU MuJoCo FK evaluator for a fixed MJCF/body contract."""

    def __init__(self, mjcf_path: str | Path, body_names: list[str]):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("Install mujoco to build FK caches") from exc
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.body_names = tuple(body_names)
        free = [
            joint
            for joint in range(self.model.njnt)
            if int(self.model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if len(free) != 1:
            raise ValueError("Expected one free joint")
        self.root_qadr = int(self.model.jnt_qposadr[free[0]])
        self.qadr = []
        for name in G1_MUJOCO_JOINT_NAMES:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise KeyError(name)
            self.qadr.append(int(self.model.jnt_qposadr[joint]))
        self.body_ids = []
        for name in self.body_names:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                raise KeyError(f"MJCF missing body {name!r}")
            self.body_ids.append(body)

    def augment(self, clip: MotionClip) -> MotionClip:
        body_pos = np.empty((clip.num_frames, len(self.body_ids), 3), np.float32)
        body_quat = np.empty((clip.num_frames, len(self.body_ids), 4), np.float32)
        self.mujoco.mj_resetData(self.model, self.data)
        for frame in range(clip.num_frames):
            self.data.qpos[self.root_qadr : self.root_qadr + 3] = clip.root_pos[frame]
            self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7] = (
                clip.root_quat_wxyz[frame]
            )
            self.data.qpos[self.qadr] = clip.joint_pos[frame]
            self.mujoco.mj_forward(self.model, self.data)
            body_pos[frame] = self.data.xpos[self.body_ids]
            body_quat[frame] = self.data.xquat[self.body_ids]
        clip.body_names = self.body_names
        clip.body_pos = body_pos
        clip.body_quat_wxyz = body_quat
        clip.body_linvel = np.gradient(
            body_pos, 1.0 / clip.fps, axis=0, edge_order=1
        ).astype(np.float32)
        clip.body_angvel = _quat_angvel(body_quat, clip.fps)
        return clip


def augment_clip_with_mujoco_fk(
    clip: MotionClip, mjcf_path: str | Path, body_names: list[str]
) -> MotionClip:
    """One-shot compatibility wrapper around :class:`MujocoFKCache`."""
    return MujocoFKCache(mjcf_path, body_names).augment(clip)
