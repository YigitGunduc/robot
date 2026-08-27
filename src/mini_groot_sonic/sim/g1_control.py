from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_NATURAL_FREQUENCY = 10.0 * 2.0 * np.pi
_DAMPING_RATIO = 2.0
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425


@dataclass(frozen=True)
class G1ControlProfile:
    """Per-joint control values from the released SONIC G1 configuration."""

    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    armature: np.ndarray
    effort_limit: np.ndarray


def _drive(armature: float, effort: float, multiplier: float = 1.0) -> tuple[float, ...]:
    armature *= multiplier
    stiffness = armature * _NATURAL_FREQUENCY**2
    damping = 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQUENCY
    action_scale = 0.25 * effort / stiffness
    return stiffness, damping, armature, effort, action_scale


def _joint_drive(name: str) -> tuple[float, ...]:
    if any(part in name for part in ("hip_pitch", "hip_roll", "knee")):
        return _drive(_ARMATURE_7520_22, 139.0)
    if "hip_yaw" in name or name == "waist_yaw_joint":
        return _drive(_ARMATURE_7520_14, 88.0)
    if "ankle" in name or name in {"waist_roll_joint", "waist_pitch_joint"}:
        return _drive(_ARMATURE_5020, 50.0, multiplier=2.0)
    if "wrist_pitch" in name or "wrist_yaw" in name:
        return _drive(_ARMATURE_4010, 5.0)
    if any(part in name for part in ("shoulder", "elbow", "wrist_roll")):
        return _drive(_ARMATURE_5020, 25.0)
    raise ValueError(f"No released SONIC G1 drive calibration for joint {name!r}")


def _default_position(name: str) -> float:
    if "hip_pitch" in name:
        return -0.312
    if "knee" in name:
        return 0.669
    if "ankle_pitch" in name:
        return -0.363
    if "elbow" in name:
        return 0.6
    if name in {"left_shoulder_pitch_joint", "right_shoulder_pitch_joint"}:
        return 0.2
    if name == "left_shoulder_roll_joint":
        return 0.2
    if name == "right_shoulder_roll_joint":
        return -0.2
    return 0.0


def sonic_g1_control_profile(joint_names: list[str]) -> G1ControlProfile:
    values = [_joint_drive(name) for name in joint_names]
    return G1ControlProfile(
        default_joint_pos=np.asarray([_default_position(name) for name in joint_names], np.float32),
        stiffness=np.asarray([value[0] for value in values], np.float32),
        damping=np.asarray([value[1] for value in values], np.float32),
        armature=np.asarray([value[2] for value in values], np.float32),
        effort_limit=np.asarray([value[3] for value in values], np.float32),
        action_scale=np.asarray([value[4] for value in values], np.float32),
    )


def calibrate_position_actuators(model, model_map, profile: G1ControlProfile) -> None:
    """Apply SONIC's implicit-PD values to compiled MuJoCo position servos.

    MuJoCo position actuators compile to ``gain * target - kp*q - kd*qd``.
    Updating the compiled arrays keeps the Menagerie asset and MJWarp backend while
    matching SONIC's controller bandwidth, damping, armature, and effort limits.
    """

    actuator_ids = model_map.actuator_ids
    model.actuator_gainprm[actuator_ids, 0] = profile.stiffness
    model.actuator_biasprm[actuator_ids, 1] = -profile.stiffness
    model.actuator_biasprm[actuator_ids, 2] = -profile.damping
    model.actuator_forcelimited[actuator_ids] = 1
    model.actuator_forcerange[actuator_ids, 0] = -profile.effort_limit
    model.actuator_forcerange[actuator_ids, 1] = profile.effort_limit
    model.dof_armature[model_map.joint_dof_adr] = profile.armature
