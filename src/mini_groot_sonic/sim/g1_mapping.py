from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class G1ModelMap:
    joint_names: list[str]
    joint_qpos_adr: np.ndarray
    joint_dof_adr: np.ndarray
    actuator_ids: np.ndarray
    root_qpos_adr: int
    root_dof_adr: int
    root_body_id: int
    body_ids: dict[str, int]
    default_joint_pos: np.ndarray
    ctrl_low: np.ndarray
    ctrl_high: np.ndarray
    joint_low: np.ndarray
    joint_high: np.ndarray
    actuator_is_position: np.ndarray
    actuator_is_motor: np.ndarray
    actuator_gear: np.ndarray

    @classmethod
    def from_mjmodel(cls, model, root_body_name: str, body_names: list[str] | tuple[str, ...]):
        import mujoco

        joint_names: list[str] = []
        qpos_adr: list[int] = []
        dof_adr: list[int] = []
        actuator_ids: list[int] = []

        # Use actuator order as the canonical 29-DOF action order.
        for aid in range(model.nu):
            jid = int(model.actuator_trnid[aid, 0])
            if jid < 0:
                continue
            jtype = int(model.jnt_type[jid])
            if jtype not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            joint_names.append(name)
            qpos_adr.append(int(model.jnt_qposadr[jid]))
            dof_adr.append(int(model.jnt_dofadr[jid]))
            actuator_ids.append(aid)

        if len(joint_names) != 29:
            raise ValueError(
                f"Expected 29 actuated 1-DOF G1 joints, found {len(joint_names)}. "
                "Use a standard 29-DOF Unitree G1 MJCF or adjust the mapping."
            )

        free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_joints) != 1:
            raise ValueError(f"Expected exactly one floating-base free joint, got {len(free_joints)}")
        root_jid = int(free_joints[0])
        root_qpos_adr = int(model.jnt_qposadr[root_jid])
        root_dof_adr = int(model.jnt_dofadr[root_jid])

        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        if root_body_id < 0:
            # Fallback to free joint's body.
            root_body_id = int(model.jnt_bodyid[root_jid])

        ids = {}
        for name in body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body {name!r} not found in MJCF")
            ids[name] = int(bid)

        qpos_adr_np = np.asarray(qpos_adr, dtype=np.int64)
        actuator_ids_np = np.asarray(actuator_ids, dtype=np.int64)
        default_joint_pos = np.asarray(model.qpos0[qpos_adr_np], dtype=np.float32)
        joint_range = np.asarray(model.jnt_range[[int(model.actuator_trnid[a, 0]) for a in actuator_ids]], dtype=np.float32)
        ctrlrange = np.asarray(model.actuator_ctrlrange[actuator_ids_np], dtype=np.float32)
        ctrl_limited = np.asarray(model.actuator_ctrllimited[actuator_ids_np], dtype=bool)
        ctrlrange[~ctrl_limited] = np.asarray([-np.inf, np.inf], dtype=np.float32)
        # MuJoCo's position shortcut compiles to an affine bias with
        # bias_prm[1] == -kp. A motor shortcut has zero affine bias.
        bias = np.asarray(model.actuator_biasprm[actuator_ids_np], dtype=np.float32)
        gain = np.asarray(model.actuator_gainprm[actuator_ids_np], dtype=np.float32)
        actuator_is_position = np.abs(bias[:, 1]) > 1e-8
        actuator_is_motor = (np.abs(bias).max(axis=1) < 1e-8) & (np.abs(gain[:, 0] - 1.0) < 1e-8)
        actuator_gear = np.asarray(model.actuator_gear[actuator_ids_np, 0], dtype=np.float32)

        return cls(
            joint_names=joint_names,
            joint_qpos_adr=qpos_adr_np,
            joint_dof_adr=np.asarray(dof_adr, dtype=np.int64),
            actuator_ids=actuator_ids_np,
            root_qpos_adr=root_qpos_adr,
            root_dof_adr=root_dof_adr,
            root_body_id=root_body_id,
            body_ids=ids,
            default_joint_pos=default_joint_pos,
            ctrl_low=ctrlrange[:, 0],
            ctrl_high=ctrlrange[:, 1],
            joint_low=joint_range[:, 0],
            joint_high=joint_range[:, 1],
            actuator_is_position=actuator_is_position,
            actuator_is_motor=actuator_is_motor,
            actuator_gear=actuator_gear,
        )
