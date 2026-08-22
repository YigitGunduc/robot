from pathlib import Path

import numpy as np
import pytest

from g1_stack.controllers.hold_position import HoldPositionController
from g1_stack.core.types import ActuatorCommand
from g1_stack.sim.mujoco_backend import MujocoBackend, MujocoConfig

mujoco = pytest.importorskip("mujoco")

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_robot.xml"


def test_reset_and_step_are_finite_and_ordered() -> None:
    with MujocoBackend(MujocoConfig(model_path=FIXTURE)) as backend:
        state = backend.reset(keyframe="stand")
        controller = HoldPositionController()
        controller.reset(state)
        command = controller.compute(state)

        for _ in range(10):
            state = backend.step(command)

        assert state.finite
        assert state.time_s == pytest.approx(0.02)
        assert state.actuator_names == ("joint_position",)
        assert backend.data.ctrl[0] == pytest.approx(0.25)
        assert abs(state.actuator_positions[0]) < 1.0


def test_step_rejects_wrong_actuator_order() -> None:
    with MujocoBackend(MujocoConfig(model_path=FIXTURE)) as backend:
        backend.reset(keyframe="stand")
        command = ActuatorCommand(names=("wrong",), values=np.array([0.0]))
        with pytest.raises(ValueError, match="order mismatch"):
            backend.step(command)


def test_step_rejects_out_of_range_control() -> None:
    with MujocoBackend(MujocoConfig(model_path=FIXTURE)) as backend:
        backend.reset(keyframe="stand")
        command = ActuatorCommand(names=backend.actuator_names, values=np.array([2.0]))
        with pytest.raises(ValueError, match="control range"):
            backend.step(command)
