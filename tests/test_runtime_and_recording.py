import json
from pathlib import Path

import pytest

from g1_stack.actors.scripted_pose import PoseTarget, ScriptedPoseActor
from g1_stack.controllers.joint_position import JointPositionController
from g1_stack.core.types import MissionRequest
from g1_stack.data.episode import EpisodeRecorder, load_episode
from g1_stack.reasoning.noop import NoOpReasoner
from g1_stack.runtime.robot_runtime import RobotRuntime, RunConfig
from g1_stack.safety.joint_limits import JointLimitSafety
from g1_stack.sim.mujoco_backend import MujocoBackend, MujocoConfig

mujoco = pytest.importorskip("mujoco")

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_robot.xml"


def test_runtime_executes_safe_loop_and_writes_episode(tmp_path: Path) -> None:
    program = (
        PoseTarget("neutral", 0.1, {}),
        PoseTarget("raised", 0.1, {"joint_position": 0.2}),
    )
    with MujocoBackend(MujocoConfig(model_path=FIXTURE)) as backend:
        lower, upper = backend.actuator_control_bounds
        runtime = RobotRuntime(
            backend,
            NoOpReasoner(),
            ScriptedPoseActor(program),
            JointPositionController(),
            JointLimitSafety(backend.actuator_names, lower, upper),
            EpisodeRecorder(tmp_path),
        )
        summary = runtime.run(
            RunConfig(max_steps=20, frame_skip=2, keyframe="stand"),
            request=MissionRequest("exercise the fixture joint"),
        )

    assert summary.success
    assert summary.steps == 20
    assert summary.stop_reason == "step limit reached"
    episode = load_episode(summary.episode_path)
    assert episode.manifest["steps"] == 20
    assert episode.manifest["intent"]["objective"] == "exercise the fixture joint"
    assert episode.manifest["mission_request"]["text"] == "exercise the fixture joint"
    assert episode.arrays["time_s"].shape == (20,)
    assert episode.arrays["requested_command"].shape == (20, 1)
    assert episode.arrays["applied_command"].shape == (20, 1)
    assert episode.arrays["reference_joint_positions"].shape == (20, 1)
    assert json.loads((summary.episode_path / "manifest.json").read_text())["success"]


def test_episode_directories_are_unique(tmp_path: Path) -> None:
    paths = []
    for _ in range(2):
        with MujocoBackend(MujocoConfig(model_path=FIXTURE)) as backend:
            lower, upper = backend.actuator_control_bounds
            runtime = RobotRuntime(
                backend,
                NoOpReasoner(),
                ScriptedPoseActor((PoseTarget("neutral", 1.0, {}),)),
                JointPositionController(),
                JointLimitSafety(backend.actuator_names, lower, upper),
                EpisodeRecorder(tmp_path),
            )
            paths.append(runtime.run(RunConfig(max_steps=1, keyframe="stand")).episode_path)

    assert paths[0] != paths[1]
    assert all(path.is_dir() for path in paths)
