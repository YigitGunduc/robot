from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from g1_stack.core.types import ActuatorCommand, RobotState

try:
    import mujoco
except ImportError as error:  # pragma: no cover - exercised in environments without sim extra
    mujoco = None
    _MUJOCO_IMPORT_ERROR = error
else:
    _MUJOCO_IMPORT_ERROR = None


@dataclass(frozen=True, slots=True)
class MujocoConfig:
    model_path: Path
    timestep_s: float = 0.002
    base_body: str = "pelvis"
    clamp_controls: bool = False

    def __post_init__(self) -> None:
        if self.timestep_s <= 0:
            raise ValueError("timestep_s must be positive")


class MujocoBackend:
    """Small deterministic MuJoCo backend with explicit actuator ordering."""

    def __init__(self, config: MujocoConfig) -> None:
        if mujoco is None:
            raise RuntimeError(
                "MuJoCo is not installed; install the project with the 'sim' extra"
            ) from _MUJOCO_IMPORT_ERROR

        model_path = config.model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")

        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.timestep_s
        self._renderer: Any | None = None
        self._rng = np.random.default_rng(0)

        self._actuator_names = tuple(self._name_of_actuator(i) for i in range(self.model.nu))
        if len(set(self._actuator_names)) != len(self._actuator_names):
            raise ValueError("The MuJoCo model contains duplicate actuator names")

        self._joint_ids = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        if np.any(self._joint_ids < 0):
            raise ValueError("Every actuator must target a joint")

        joint_types = self.model.jnt_type[self._joint_ids]
        scalar_types = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        if not np.all(np.isin(joint_types, scalar_types)):
            raise ValueError("Only scalar hinge/slide joint actuators are supported")

        self._qpos_addresses = self.model.jnt_qposadr[self._joint_ids]
        self._dof_addresses = self.model.jnt_dofadr[self._joint_ids]
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, config.base_body
        )
        if self._base_body_id < 0:
            raise ValueError(f"Base body '{config.base_body}' was not found in the model")

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return self._actuator_names

    @property
    def timestep_s(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def actuator_control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return control limits in actuator order, using infinities when unlimited."""
        limited = self.model.actuator_ctrllimited.astype(bool)
        lower = np.where(limited, self.model.actuator_ctrlrange[:, 0], -np.inf)
        upper = np.where(limited, self.model.actuator_ctrlrange[:, 1], np.inf)
        return lower.copy(), upper.copy()

    def _name_of_actuator(self, actuator_id: int) -> str:
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        return name or f"actuator_{actuator_id}"

    def reset(self, *, seed: int = 0, keyframe: str | None = None) -> RobotState:
        self._rng = np.random.default_rng(seed)
        if keyframe is None:
            mujoco.mj_resetData(self.model, self.data)
        else:
            keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if keyframe_id < 0:
                raise ValueError(f"Keyframe '{keyframe}' was not found in the model")
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def step(self, command: ActuatorCommand, *, frame_skip: int = 1) -> RobotState:
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        if command.names != self.actuator_names:
            raise ValueError(
                "Actuator order mismatch; command names must exactly match the loaded model"
            )

        values = np.asarray(command.values, dtype=np.float64)
        if values.shape != (self.model.nu,):
            raise ValueError(f"Expected {self.model.nu} actuator values, received {values.size}")

        if np.any(self.model.actuator_ctrllimited):
            limited = self.model.actuator_ctrllimited.astype(bool)
            lower = self.model.actuator_ctrlrange[:, 0]
            upper = self.model.actuator_ctrlrange[:, 1]
            violation = limited & ((values < lower) | (values > upper))
            if np.any(violation):
                if not self.config.clamp_controls:
                    names = [self.actuator_names[i] for i in np.flatnonzero(violation)]
                    raise ValueError(f"Actuator command exceeds control range: {names}")
                values = np.where(limited, np.clip(values, lower, upper), values)

        self.data.ctrl[:] = values
        for _ in range(frame_skip):
            mujoco.mj_step(self.model, self.data)
        state = self.state()
        if not state.finite:
            raise FloatingPointError("MuJoCo produced non-finite robot state")
        return state

    def state(self) -> RobotState:
        return RobotState(
            time_s=float(self.data.time),
            qpos=self.data.qpos,
            qvel=self.data.qvel,
            actuator_names=self.actuator_names,
            actuator_positions=self.data.qpos[self._qpos_addresses],
            actuator_velocities=self.data.qvel[self._dof_addresses],
            actuator_forces=self.data.actuator_force,
            base_position=self.data.xpos[self._base_body_id],
            base_quaternion_wxyz=self.data.xquat[self._base_body_id],
            contact_count=int(self.data.ncon),
        )

    def render(self, *, width: int = 640, height: int = 480, camera: str | int | None = None):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        if camera is None:
            self._renderer.update_scene(self.data)
        else:
            self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> MujocoBackend:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
