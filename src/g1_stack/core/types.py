from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _frozen_float_array(value: NDArray[np.floating] | list[float]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    """A complete ordered command for all actuators in a backend."""

    names: tuple[str, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        values = _frozen_float_array(self.values)
        if values.ndim != 1:
            raise ValueError("ActuatorCommand.values must be one-dimensional")
        if len(self.names) != values.size:
            raise ValueError("Actuator names and values must have equal length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Actuator names must be unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("Actuator commands must be finite")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PhysicalIntent:
    """Structured high-level directive consumed by an embodied actor."""

    objective: str
    constraints: tuple[str, ...] = ()
    strategy_hint: str | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("PhysicalIntent.objective cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("PhysicalIntent.confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """Result of applying independent command safety."""

    command: ActuatorCommand
    limited_actuators: tuple[str, ...] = ()
    stopped: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RobotState:
    """Backend-neutral snapshot copied out of a simulator or robot driver."""

    time_s: float
    qpos: FloatArray
    qvel: FloatArray
    actuator_names: tuple[str, ...]
    actuator_positions: FloatArray
    actuator_velocities: FloatArray
    actuator_forces: FloatArray
    base_position: FloatArray
    base_quaternion_wxyz: FloatArray
    contact_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "qpos",
            "qvel",
            "actuator_positions",
            "actuator_velocities",
            "actuator_forces",
            "base_position",
            "base_quaternion_wxyz",
        ):
            object.__setattr__(self, field_name, _frozen_float_array(getattr(self, field_name)))

        count = len(self.actuator_names)
        if self.actuator_positions.size != count:
            raise ValueError("actuator_positions does not match actuator_names")
        if self.actuator_velocities.size != count:
            raise ValueError("actuator_velocities does not match actuator_names")
        if self.actuator_forces.size != count:
            raise ValueError("actuator_forces does not match actuator_names")
        if self.base_position.shape != (3,):
            raise ValueError("base_position must have shape (3,)")
        if self.base_quaternion_wxyz.shape != (4,):
            raise ValueError("base_quaternion_wxyz must have shape (4,)")
        if not np.isfinite(self.time_s):
            raise ValueError("time_s must be finite")

    @property
    def finite(self) -> bool:
        arrays = (
            self.qpos,
            self.qvel,
            self.actuator_positions,
            self.actuator_velocities,
            self.actuator_forces,
            self.base_position,
            self.base_quaternion_wxyz,
        )
        return all(np.all(np.isfinite(array)) for array in arrays)
