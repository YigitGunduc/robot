from types import SimpleNamespace

import numpy as np

from mini_groot_sonic.sim.g1_control import (
    calibrate_position_actuators,
    sonic_g1_control_profile,
)

G1_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def test_sonic_g1_profile_has_small_joint_specific_residuals_and_crouch():
    profile = sonic_g1_control_profile(G1_JOINTS)
    assert profile.action_scale.shape == (29,)
    assert np.all(profile.action_scale > 0.05)
    assert np.all(profile.action_scale < 0.6)
    assert profile.default_joint_pos[G1_JOINTS.index("left_hip_pitch_joint")] == -0.312
    assert profile.default_joint_pos[G1_JOINTS.index("left_knee_joint")] == 0.669
    assert profile.default_joint_pos[G1_JOINTS.index("right_ankle_pitch_joint")] == -0.363
    assert profile.stiffness[G1_JOINTS.index("left_hip_pitch_joint")] > 90.0
    assert profile.damping[G1_JOINTS.index("left_hip_pitch_joint")] > 6.0


def test_position_actuator_calibration_updates_compiled_pd_and_effort_limits():
    profile = sonic_g1_control_profile(G1_JOINTS[:2])
    model = SimpleNamespace(
        actuator_gainprm=np.zeros((2, 10)),
        actuator_biasprm=np.zeros((2, 10)),
        actuator_forcelimited=np.zeros(2, dtype=np.int32),
        actuator_forcerange=np.zeros((2, 2)),
        dof_armature=np.zeros(8),
    )
    mapping = SimpleNamespace(
        actuator_ids=np.asarray([0, 1]),
        joint_dof_adr=np.asarray([6, 7]),
    )
    calibrate_position_actuators(model, mapping, profile)
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], profile.stiffness)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], -profile.stiffness)
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], -profile.damping)
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], profile.effort_limit)
    np.testing.assert_allclose(model.dof_armature[[6, 7]], profile.armature)
