import numpy as np
import pytest

from mini_groot_sonic.data.bones import BonesClip
from mini_groot_sonic.data.preprocess import precompute_mujoco_kinematics
from mini_groot_sonic.sim.g1_mapping import G1ModelMap

mujoco = pytest.importorskip("mujoco")


def _xml(actuator_tag: str):
    bodies = [
        f'<body name="b{i}" pos="0 0 0.05"><joint name="j{i}" axis="0 1 0" '
        'range="-1 1"/><geom type="sphere" size="0.01" mass="0.1"/>'
        for i in range(29)
    ]
    body_xml = "".join(bodies) + "</body>" * 29
    actuators = "".join(
        f'<{actuator_tag} name="a{i}" joint="j{i}" ctrlrange="-{i + 1} {i + 1}"/>'
        for i in range(29)
    )
    xml = (
        '<mujoco><compiler angle="radian"/><worldbody><body name="pelvis" pos="0 0 1">'
        '<freejoint/><geom type="sphere" size="0.02" mass="1"/>'
        f"{body_xml}</body></worldbody><actuator>{actuators}</actuator></mujoco>"
    )
    return xml


def _model(actuator_tag: str):
    return mujoco.MjModel.from_xml_string(_xml(actuator_tag))


def test_detects_motor_actuator_semantics():
    mapping = G1ModelMap.from_mjmodel(_model("motor"), "pelvis", [])
    assert len(mapping.joint_names) == 29
    assert mapping.actuator_is_motor.all()
    assert not mapping.actuator_is_position.any()


def test_detects_position_actuator_semantics():
    mapping = G1ModelMap.from_mjmodel(_model("position"), "pelvis", [])
    assert mapping.actuator_is_position.all()
    assert not mapping.actuator_is_motor.any()


def test_preprocessing_computes_root_linear_and_angular_velocity(tmp_path):
    xml_path = tmp_path / "g1.xml"
    xml_path.write_text(_xml("motor"))
    angles = np.asarray([0.0, 0.1, 0.2], np.float32)
    root_quat = np.stack(
        [np.cos(angles / 2), np.zeros(3), np.zeros(3), np.sin(angles / 2)],
        axis=-1,
    ).astype(np.float32)
    clip = BonesClip(
        motion_id="turn",
        caption="turn",
        fps=50.0,
        joint_pos=np.zeros((3, 29), np.float32),
        joint_vel=np.zeros((3, 29), np.float32),
        root_pos=np.asarray([[0, 0, 1], [0.01, 0, 1], [0.02, 0, 1]], np.float32),
        root_quat=root_quat,
    )
    arrays = precompute_mujoco_kinematics(clip, xml_path, [f"j{i}" for i in range(29)])
    assert np.all(arrays["root_linvel"][:, 0] > 0.4)
    assert np.all(arrays["root_angvel"][:, 2] > 4.0)
