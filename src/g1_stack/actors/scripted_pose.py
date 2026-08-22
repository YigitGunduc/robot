from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from g1_stack.core.types import PhysicalIntent, RobotState, WholeBodyReference


@dataclass(frozen=True, slots=True)
class PoseTarget:
    name: str
    duration_s: float
    offsets_rad: dict[str, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PoseTarget.name cannot be empty")
        if self.duration_s <= 0:
            raise ValueError("PoseTarget.duration_s must be positive")


DEFAULT_POSE_PROGRAM = (
    PoseTarget("neutral", 1.0, {}),
    PoseTarget(
        "arms_out",
        1.5,
        {
            "left_shoulder_roll_joint": 0.45,
            "right_shoulder_roll_joint": -0.45,
            "left_elbow_joint": 0.30,
            "right_elbow_joint": 0.30,
        },
    ),
    PoseTarget(
        "left_wave_a",
        0.75,
        {
            "left_shoulder_pitch_joint": -0.25,
            "left_shoulder_roll_joint": 0.65,
            "left_elbow_joint": 0.75,
            "left_wrist_roll_joint": -0.35,
        },
    ),
    PoseTarget(
        "left_wave_b",
        0.75,
        {
            "left_shoulder_pitch_joint": -0.25,
            "left_shoulder_roll_joint": 0.65,
            "left_elbow_joint": 0.75,
            "left_wrist_roll_joint": 0.35,
        },
    ),
    PoseTarget(
        "small_crouch",
        1.5,
        {
            "left_hip_pitch_joint": -0.10,
            "right_hip_pitch_joint": -0.10,
            "left_knee_joint": 0.18,
            "right_knee_joint": 0.18,
            "left_ankle_pitch_joint": -0.08,
            "right_ankle_pitch_joint": -0.08,
        },
    ),
    PoseTarget("neutral", 1.5, {}),
)


class ScriptedPoseActor:
    """Smooth deterministic pose sequence for runtime and safety validation."""

    def __init__(self, program: tuple[PoseTarget, ...] = DEFAULT_POSE_PROGRAM) -> None:
        if not program:
            raise ValueError("A pose program must contain at least one target")
        self.program = program
        self._base: np.ndarray | None = None
        self._start_time_s = 0.0
        self._index_by_name: dict[str, int] = {}
        self._pending_pose: str | None = None
        self._manual_goal: np.ndarray | None = None
        self._manual_start: np.ndarray | None = None
        self._manual_start_time_s = 0.0
        self._manual_duration_s = 1.0
        self._auto = True
        self._lock = Lock()

    @property
    def available_poses(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.program)

    def reset(self, state: RobotState, intent: PhysicalIntent) -> None:
        del intent
        self._base = np.asarray(state.actuator_positions, dtype=np.float64).copy()
        self._start_time_s = state.time_s
        self._index_by_name = {name: i for i, name in enumerate(state.actuator_names)}
        self._manual_goal = None
        self._manual_start = None
        self._auto = True
        with self._lock:
            self._pending_pose = None
        self._validate_program_names()

    def request_pose(self, name: str) -> None:
        names = set(self.available_poses)
        if name not in names:
            raise ValueError(f"Unknown pose '{name}'; available poses: {sorted(names)}")
        with self._lock:
            self._pending_pose = name

    def request_auto(self) -> None:
        with self._lock:
            self._pending_pose = "__auto__"

    def act(self, state: RobotState, intent: PhysicalIntent) -> WholeBodyReference:
        del intent
        if self._base is None:
            raise RuntimeError("ScriptedPoseActor.reset() must be called first")

        pending = self._take_pending_pose()
        current_target = self._target_at(state.time_s)
        if pending == "__auto__":
            self._auto = True
            self._start_time_s = state.time_s
            self._manual_goal = None
            self._manual_start = None
        elif pending is not None:
            self._auto = False
            self._manual_start = current_target.copy()
            self._manual_goal = self._target_for_name(pending)
            self._manual_start_time_s = state.time_s

        values = self._target_at(state.time_s)
        return WholeBodyReference(
            joint_names=state.actuator_names,
            joint_position_targets=values,
        )

    def _take_pending_pose(self) -> str | None:
        with self._lock:
            pending = self._pending_pose
            self._pending_pose = None
        return pending

    def _target_at(self, time_s: float) -> np.ndarray:
        if not self._auto and self._manual_goal is not None and self._manual_start is not None:
            progress = (time_s - self._manual_start_time_s) / self._manual_duration_s
            alpha = self._smoothstep(progress)
            return self._manual_start + alpha * (self._manual_goal - self._manual_start)
        return self._auto_target_at(time_s)

    def _auto_target_at(self, time_s: float) -> np.ndarray:
        durations = np.asarray([pose.duration_s for pose in self.program], dtype=np.float64)
        total_duration = float(np.sum(durations))
        total_elapsed = max(0.0, time_s - self._start_time_s)
        cycle_index = int(total_elapsed // total_duration)
        elapsed = total_elapsed % total_duration
        cumulative = np.cumsum(durations)
        segment_index = int(np.searchsorted(cumulative, elapsed, side="right"))
        segment_index = min(segment_index, len(self.program) - 1)
        segment_start = 0.0 if segment_index == 0 else float(cumulative[segment_index - 1])
        progress = (elapsed - segment_start) / durations[segment_index]
        if segment_index == 0 and cycle_index == 0:
            assert self._base is not None
            start = self._base
        else:
            previous_index = (segment_index - 1) % len(self.program)
            start = self._target_for_pose(self.program[previous_index])
        goal = self._target_for_pose(self.program[segment_index])
        alpha = self._smoothstep(progress)
        return start + alpha * (goal - start)

    def _target_for_name(self, name: str) -> np.ndarray:
        for pose in self.program:
            if pose.name == name:
                return self._target_for_pose(pose)
        raise ValueError(f"Unknown pose '{name}'")

    def _target_for_pose(self, pose: PoseTarget) -> np.ndarray:
        assert self._base is not None
        target = self._base.copy()
        for joint_name, offset in pose.offsets_rad.items():
            target[self._index_by_name[joint_name]] += offset
        return target

    def _validate_program_names(self) -> None:
        missing = sorted(
            {
                joint_name
                for pose in self.program
                for joint_name in pose.offsets_rad
                if joint_name not in self._index_by_name
            }
        )
        if missing:
            raise ValueError(f"Pose program references missing actuators: {missing}")

    @staticmethod
    def _smoothstep(value: float) -> float:
        clipped = float(np.clip(value, 0.0, 1.0))
        return clipped * clipped * (3.0 - 2.0 * clipped)
