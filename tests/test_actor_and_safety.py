from pathlib import Path

import numpy as np
import pytest

from g1_stack.actors.scripted_pose import PoseTarget, ScriptedPoseActor
from g1_stack.controllers.joint_position import JointPositionController
from g1_stack.core.types import ActuatorCommand, PhysicalIntent
from g1_stack.safety.joint_limits import JointLimitSafety
from g1_stack.sim.mujoco_backend import MujocoBackend, MujocoConfig

mujoco = pytest.importorskip("mujoco")

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_robot.xml"
INTENT = PhysicalIntent(objective="test the actor")


def test_scripted_actor_interpolates_and_accepts_manual_pose() -> None:
    program = (
        PoseTarget("neutral", 1.0, {}),
        PoseTarget("raised", 1.0, {"joint_position": 0.5}),
    )
    with MujocoBackend(MujocoConfig(model_path=FIXTURE)) as backend:
        state = backend.reset(keyframe="stand")
        actor = ScriptedPoseActor(program)
        actor.reset(state, INTENT)
        actor.request_pose("raised")

        start = actor.act(state, INTENT)
        controller = JointPositionController()
        controller.reset(state)
        later_state = backend.step(controller.compute(state, start), frame_skip=250)
        halfway = actor.act(later_state, INTENT)

    assert start.joint_position_targets[0] == pytest.approx(0.25)
    assert 0.25 < halfway.joint_position_targets[0] < 0.75


def test_joint_position_controller_maps_reference_to_command() -> None:
    program = (PoseTarget("neutral", 1.0, {}),)
    with MujocoBackend(MujocoConfig(model_path=FIXTURE)) as backend:
        state = backend.reset(keyframe="stand")
        actor = ScriptedPoseActor(program)
        actor.reset(state, INTENT)
        reference = actor.act(state, INTENT)
        controller = JointPositionController()
        controller.reset(state)

        command = controller.compute(state, reference)

    assert command.names == state.actuator_names
    assert command.values.tolist() == reference.joint_position_targets.tolist()


def test_safety_clips_position_and_target_rate() -> None:
    with MujocoBackend(MujocoConfig(model_path=FIXTURE)) as backend:
        state = backend.reset(keyframe="stand")
        lower, upper = backend.actuator_control_bounds
        safety = JointLimitSafety(
            backend.actuator_names,
            lower,
            upper,
            max_target_rate_rad_s=1.0,
        )
        safety.reset(state)
        requested = ActuatorCommand(backend.actuator_names, np.array([2.0]))

        decision = safety.filter(requested, state, dt_s=0.1)

    assert decision.command.values[0] == pytest.approx(0.35)
    assert decision.limited_actuators == ("joint_position",)
    assert decision.reasons == ("actuator position limit", "actuator target rate limit")


def test_safety_emergency_stop_holds_observed_position() -> None:
    with MujocoBackend(MujocoConfig(model_path=FIXTURE)) as backend:
        state = backend.reset(keyframe="stand")
        lower, upper = backend.actuator_control_bounds
        safety = JointLimitSafety(backend.actuator_names, lower, upper)
        safety.reset(state)
        safety.engage_emergency_stop("test stop")
        requested = ActuatorCommand(backend.actuator_names, np.array([0.8]))

        decision = safety.filter(requested, state, dt_s=0.01)

    assert decision.stopped
    assert decision.reasons == ("test stop",)
    assert decision.command.values.tolist() == state.actuator_positions.tolist()
