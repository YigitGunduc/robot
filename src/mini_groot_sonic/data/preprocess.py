from __future__ import annotations

from pathlib import Path

import numpy as np

from mini_groot_sonic.data.bones import BonesClip


def precompute_mujoco_kinematics(clip: BonesClip, mjcf: str | Path, joint_names: list[str]) -> dict[str, np.ndarray]:
    """Compute MuJoCo body pose/velocity reference tracks once on CPU.

    This is deliberately preprocessing, not part of the GPU PPO loop. Keeping the
    reference body tracks on disk makes the MJWarp training loop simple and fast.
    """
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("Install simulator extras: pip install -e '.[sim]'") from exc

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    joint_qpos = []
    joint_dof = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint {name!r} missing from MJCF")
        joint_qpos.append(int(model.jnt_qposadr[jid]))
        joint_dof.append(int(model.jnt_dofadr[jid]))
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise ValueError("Expected one free root joint")
    root_jid = int(free_joints[0])
    rq = int(model.jnt_qposadr[root_jid])
    rv = int(model.jnt_dofadr[root_jid])

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(1, model.nbody)]
    body_ids = np.arange(1, model.nbody)
    t = clip.length
    body_pos = np.zeros((t, len(body_ids), 3), np.float32)
    body_quat = np.zeros((t, len(body_ids), 4), np.float32)
    body_linvel = np.zeros((t, len(body_ids), 3), np.float32)
    body_angvel = np.zeros((t, len(body_ids), 3), np.float32)
    root_linvel = np.zeros((t, 3), np.float32)
    root_angvel = np.zeros((t, 3), np.float32)

    qpos_track = np.repeat(model.qpos0[None], t, axis=0)
    qpos_track[:, rq : rq + 3] = clip.root_pos
    qpos_track[:, rq + 3 : rq + 7] = clip.root_quat
    qpos_track[:, joint_qpos] = clip.joint_pos

    # Let MuJoCo differentiate quaternion coordinates using its own tangent-space
    # convention. Central differences are used away from the boundaries.
    differentiated = np.zeros((t, model.nv), np.float64)
    for i in range(t):
        lo = max(0, i - 1)
        hi = min(t - 1, i + 1)
        if lo != hi:
            mujoco.mj_differentiatePos(
                model,
                differentiated[i],
                (hi - lo) / clip.fps,
                qpos_track[lo],
                qpos_track[hi],
            )
    root_linvel[:] = differentiated[:, rv : rv + 3]
    root_angvel[:] = differentiated[:, rv + 3 : rv + 6]

    for i in range(t):
        data.qpos[:] = qpos_track[i]
        data.qvel[:] = differentiated[i]
        data.qvel[joint_dof] = clip.joint_vel[i]
        mujoco.mj_forward(model, data)
        body_pos[i] = data.xpos[body_ids]
        body_quat[i] = data.xquat[body_ids]
        body_angvel[i] = data.cvel[body_ids, :3]
        body_linvel[i] = data.cvel[body_ids, 3:]

    return {
        "joint_pos": clip.joint_pos,
        "joint_vel": clip.joint_vel,
        "root_pos": clip.root_pos,
        "root_quat": clip.root_quat,
        "root_linvel": root_linvel,
        "root_angvel": root_angvel,
        "body_pos": body_pos,
        "body_quat": body_quat,
        "body_linvel": body_linvel,
        "body_angvel": body_angvel,
        "body_names": np.asarray(body_names, dtype=object),
        "fps": np.asarray(clip.fps, dtype=np.float32),
        "motion_id": np.asarray(clip.motion_id, dtype=object),
        "caption": np.asarray(clip.caption, dtype=object),
    }


def save_preprocessed(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
