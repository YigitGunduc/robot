from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gear_sonic_mjx.config import SonicConfig
from gear_sonic_mjx.data_process.bones import MotionClip, _finite_difference
from gear_sonic_mjx.envs.motion_library import open_motion_library
from gear_sonic_mjx.g1_parameters import (
    DEFAULT_ANGLES_MJ,
    EFFORT,
    G1_MUJOCO_JOINT_NAMES,
    SONIC_ANTI_SHAKE_BODY_NAMES,
    SONIC_EE_TERMINATION_BODY_NAMES,
    SONIC_REWARD_POINT_BODY_NAMES,
    SONIC_TRACKED_BODY_NAMES,
)


@dataclass
class PreflightReport:
    checks: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def validate_mjcf(path: str | Path, cfg: SonicConfig, report: PreflightReport):
    try:
        import mujoco
    except ImportError as exc:
        report.error(f"MuJoCo is not installed: {exc}")
        return None
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001 - report third-party parser failures verbatim
        report.error(f"failed to compile MJCF {path}: {exc}")
        return None

    free = [
        jid
        for jid in range(model.njnt)
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free) != 1:
        report.error(f"MJCF must contain exactly one free joint, found {len(free)}")
    if model.nu != 29:
        report.error(f"MJCF must expose exactly 29 actuators, found {model.nu}")

    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    actuator_ids: list[int] = []
    for name in G1_MUJOCO_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            report.error(f"missing canonical G1 joint {name!r}")
            continue
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            report.error(f"joint {name!r} is not a hinge")
        if not bool(model.jnt_limited[jid]):
            report.error(f"joint {name!r} has no position limit")
        qpos_indices.append(int(model.jnt_qposadr[jid]))
        qvel_indices.append(int(model.jnt_dofadr[jid]))
        matches = [
            aid for aid in range(model.nu) if int(model.actuator_trnid[aid, 0]) == jid
        ]
        if len(matches) != 1:
            report.error(
                f"joint {name!r} must have one direct actuator, found {len(matches)}"
            )
            continue
        aid = matches[0]
        actuator_ids.append(aid)
        direct_torque = (
            int(model.actuator_trntype[aid]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_dyntype[aid]) == int(mujoco.mjtDyn.mjDYN_NONE)
            and int(model.actuator_gaintype[aid]) == int(mujoco.mjtGain.mjGAIN_FIXED)
            and int(model.actuator_biastype[aid]) == int(mujoco.mjtBias.mjBIAS_NONE)
            and abs(float(model.actuator_gainprm[aid, 0]) - 1.0) <= 1e-6
            and abs(float(model.actuator_gear[aid, 0]) - 1.0) <= 1e-6
        )
        if not direct_torque:
            report.error(
                f"actuator {aid} for {name!r} is not a direct unit-gain torque motor"
            )

    required_bodies = sorted(
        set(
            SONIC_TRACKED_BODY_NAMES
            + SONIC_REWARD_POINT_BODY_NAMES
            + SONIC_EE_TERMINATION_BODY_NAMES
            + SONIC_ANTI_SHAKE_BODY_NAMES
            + cfg.contact.allowed_body_names
        )
    )
    for name in required_bodies:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            report.error(f"MJCF is missing required semantic body {name!r}")

    if len(qpos_indices) == 29:
        lower = np.asarray(
            [
                model.jnt_range[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name), 0
                ]
                for name in G1_MUJOCO_JOINT_NAMES
            ]
        )
        upper = np.asarray(
            [
                model.jnt_range[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name), 1
                ]
                for name in G1_MUJOCO_JOINT_NAMES
            ]
        )
        default = DEFAULT_ANGLES_MJ.numpy()
        bad = np.flatnonzero((default < lower) | (default > upper))
        if len(bad):
            report.error(
                "deployment default angles violate MJCF limits: "
                + ", ".join(G1_MUJOCO_JOINT_NAMES[i] for i in bad)
            )
        report.checks["joint_lower"] = lower.tolist()
        report.checks["joint_upper"] = upper.tolist()

    # Verify the sign and one-to-one meaning of every control channel dynamically.
    # Passive/gravity accelerations cancel in the centered +1/-1 torque response.
    if len(qvel_indices) == 29 and len(actuator_ids) == 29:
        data = mujoco.MjData(model)
        original_disable = int(model.opt.disableflags)
        model.opt.disableflags = original_disable | int(
            mujoco.mjtDisableBit.mjDSBL_CONTACT
        )
        responses = []
        try:
            for aid, dof in zip(actuator_ids, qvel_indices, strict=True):
                data.ctrl[:] = 0.0
                data.ctrl[aid] = 1.0
                mujoco.mj_forward(model, data)
                plus = float(data.qacc[dof])
                data.ctrl[aid] = -1.0
                mujoco.mj_forward(model, data)
                minus = float(data.qacc[dof])
                responses.append(0.5 * (plus - minus))
        finally:
            model.opt.disableflags = original_disable
        bad = [i for i, response in enumerate(responses) if response <= 0.0]
        if bad:
            report.error(
                "positive control does not produce positive canonical-joint acceleration: "
                + ", ".join(G1_MUJOCO_JOINT_NAMES[i] for i in bad)
            )
        report.checks["unit_torque_acceleration_response"] = responses

    report.checks["mjcf"] = {
        "path": str(Path(path).resolve()),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "timestep_xml": float(model.opt.timestep),
        "canonical_qpos_indices": qpos_indices,
        "canonical_qvel_indices": qvel_indices,
        "canonical_actuator_ids": actuator_ids,
        "effort_limits": EFFORT.tolist(),
    }
    return model


def _sample_indices(total: int, maximum: int) -> list[int]:
    if total <= maximum:
        return list(range(total))
    return sorted(set(np.linspace(0, total - 1, maximum).round().astype(int).tolist()))


def validate_motion_library(
    root: str | Path,
    mjcf_path: str | Path,
    cfg: SonicConfig,
    report: PreflightReport,
    max_clips: int = 100,
    max_frames_per_clip: int = 8,
) -> None:
    try:
        library = open_motion_library(root, cfg.motion.target_fps)
    except Exception as exc:  # noqa: BLE001 - report library-format failures verbatim
        report.error(f"failed to open motion library: {exc}")
        return
    try:
        import mujoco
    except ImportError as exc:
        report.error(f"MuJoCo is required for FK validation: {exc}")
        return

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    free = [
        jid
        for jid in range(model.njnt)
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free) != 1:
        return
    root_adr = int(model.jnt_qposadr[free[0]])
    qadr = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in G1_MUJOCO_JOINT_NAMES
    ]
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in SONIC_TRACKED_BODY_NAMES
    ]
    if any(jid < 0 for jid in qadr) or any(bid < 0 for bid in body_ids):
        report.error(
            "cannot validate motion FK because canonical joints/bodies are missing"
        )
        return
    joint_ids = qadr
    qadr = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    lower = np.asarray([model.jnt_range[jid, 0] for jid in joint_ids])
    upper = np.asarray([model.jnt_range[jid, 1] for jid in joint_ids])

    maxima = {
        "quaternion_norm_error": 0.0,
        "joint_limit_violation_rad": 0.0,
        "joint_velocity_consistency": 0.0,
        "max_joint_speed_rad_s": 0.0,
        "max_root_linear_speed_m_s": 0.0,
        "max_root_angular_speed_rad_s": 0.0,
        "fk_position_error_m": 0.0,
        "fk_orientation_error_rad": 0.0,
        "max_contact_penetration_m": 0.0,
        "minimum_tracked_body_origin_z_m": float("inf"),
    }
    sampled = _sample_indices(len(library), max_clips)
    for motion_id in sampled:
        clip: MotionClip = library._load(motion_id)
        arrays = [clip.root_pos, clip.root_quat_wxyz, clip.joint_pos, clip.joint_vel]
        if not all(np.isfinite(value).all() for value in arrays):
            report.error(f"{clip.name}: NaN/Inf in root or joint trajectory")
            continue
        qnorm = np.linalg.norm(clip.root_quat_wxyz, axis=-1)
        maxima["quaternion_norm_error"] = max(
            maxima["quaternion_norm_error"], float(np.max(np.abs(qnorm - 1.0)))
        )
        violation = np.maximum(lower - clip.joint_pos, 0) + np.maximum(
            clip.joint_pos - upper, 0
        )
        maxima["joint_limit_violation_rad"] = max(
            maxima["joint_limit_violation_rad"], float(np.max(violation))
        )
        fd = _finite_difference(clip.joint_pos, clip.fps)
        maxima["joint_velocity_consistency"] = max(
            maxima["joint_velocity_consistency"],
            float(np.max(np.abs(fd - clip.joint_vel))),
        )
        maxima["max_joint_speed_rad_s"] = max(
            maxima["max_joint_speed_rad_s"], float(np.max(np.abs(clip.joint_vel)))
        )
        root_linear_speed = np.linalg.norm(
            np.diff(clip.root_pos, axis=0) * clip.fps, axis=-1
        )
        maxima["max_root_linear_speed_m_s"] = max(
            maxima["max_root_linear_speed_m_s"], float(np.max(root_linear_speed))
        )
        root_dot = np.abs(
            np.sum(clip.root_quat_wxyz[:-1] * clip.root_quat_wxyz[1:], axis=-1)
        )
        root_angular_speed = 2.0 * np.arccos(np.clip(root_dot, 0.0, 1.0)) * clip.fps
        maxima["max_root_angular_speed_rad_s"] = max(
            maxima["max_root_angular_speed_rad_s"],
            float(np.max(root_angular_speed)),
        )
        if (
            clip.body_names is None
            or clip.body_pos is None
            or clip.body_quat_wxyz is None
        ):
            report.error(f"{clip.name}: missing FK cache")
            continue
        lookup = {name: i for i, name in enumerate(clip.body_names)}
        missing = [name for name in SONIC_TRACKED_BODY_NAMES if name not in lookup]
        if missing:
            report.error(f"{clip.name}: FK cache missing bodies {missing}")
            continue
        ref_idx = [lookup[name] for name in SONIC_TRACKED_BODY_NAMES]
        for frame in _sample_indices(clip.num_frames, max_frames_per_clip):
            data.qpos[root_adr : root_adr + 3] = clip.root_pos[frame]
            data.qpos[root_adr + 3 : root_adr + 7] = clip.root_quat_wxyz[frame]
            data.qpos[qadr] = clip.joint_pos[frame]
            mujoco.mj_forward(model, data)
            if data.ncon:
                penetration = max(
                    0.0,
                    -min(float(data.contact[index].dist) for index in range(data.ncon)),
                )
                maxima["max_contact_penetration_m"] = max(
                    maxima["max_contact_penetration_m"], penetration
                )
            pos_error = np.linalg.norm(
                data.xpos[body_ids] - clip.body_pos[frame, ref_idx], axis=-1
            )
            maxima["minimum_tracked_body_origin_z_m"] = min(
                maxima["minimum_tracked_body_origin_z_m"],
                float(np.min(data.xpos[body_ids, 2])),
            )
            maxima["fk_position_error_m"] = max(
                maxima["fk_position_error_m"], float(np.max(pos_error))
            )
            dot = np.abs(
                np.sum(
                    data.xquat[body_ids] * clip.body_quat_wxyz[frame, ref_idx], axis=-1
                )
            )
            angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
            maxima["fk_orientation_error_rad"] = max(
                maxima["fk_orientation_error_rad"], float(np.max(angle))
            )

    thresholds = {
        "quaternion_norm_error": 1e-3,
        "joint_limit_violation_rad": 1e-3,
        "joint_velocity_consistency": 1e-4,
        "fk_position_error_m": 1e-5,
        "fk_orientation_error_rad": 1e-4,
    }
    for name, threshold in thresholds.items():
        if maxima[name] > threshold:
            report.error(f"motion {name}={maxima[name]:.6g} exceeds {threshold:.6g}")
    sanity_limits = {
        "max_joint_speed_rad_s": 100.0,
        "max_root_linear_speed_m_s": 20.0,
        "max_root_angular_speed_rad_s": 30.0,
        "max_contact_penetration_m": 0.03,
    }
    for name, threshold in sanity_limits.items():
        if maxima[name] > threshold:
            report.error(
                f"motion {name}={maxima[name]:.6g} exceeds sanity limit {threshold:.6g}"
            )
    if maxima["minimum_tracked_body_origin_z_m"] < -0.05:
        report.error(
            "tracked body origin is below the ground reference: "
            f"z={maxima['minimum_tracked_body_origin_z_m']:.6g} m"
        )
    if not np.isfinite(maxima["minimum_tracked_body_origin_z_m"]):
        maxima["minimum_tracked_body_origin_z_m"] = None
    report.checks["motions"] = {
        "path": str(Path(root).resolve()),
        "clips": len(library),
        "sampled_clips": len(sampled),
        "target_fps": cfg.motion.target_fps,
        **maxima,
    }
