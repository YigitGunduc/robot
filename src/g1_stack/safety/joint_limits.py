from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from g1_stack.core.types import ActuatorCommand, RobotState, SafetyDecision


class JointLimitSafety:
    """Limits target position, target rate, excessive velocity, and fallen state."""

    def __init__(
        self,
        names: tuple[str, ...],
        lower: NDArray[np.floating],
        upper: NDArray[np.floating],
        *,
        max_target_rate_rad_s: float = 2.0,
        max_observed_velocity_rad_s: float = 35.0,
        minimum_base_height_m: float = 0.35,
    ) -> None:
        self.names = names
        self.lower = np.asarray(lower, dtype=np.float64).copy()
        self.upper = np.asarray(upper, dtype=np.float64).copy()
        if self.lower.shape != (len(names),) or self.upper.shape != (len(names),):
            raise ValueError("Safety bounds must match actuator names")
        if np.any(self.lower > self.upper):
            raise ValueError("Safety lower bounds cannot exceed upper bounds")
        if max_target_rate_rad_s <= 0:
            raise ValueError("max_target_rate_rad_s must be positive")
        self.max_target_rate_rad_s = max_target_rate_rad_s
        self.max_observed_velocity_rad_s = max_observed_velocity_rad_s
        self.minimum_base_height_m = minimum_base_height_m
        self._last_command: np.ndarray | None = None
        self._emergency_stop = False
        self._emergency_reason = "operator emergency stop"

    def reset(self, state: RobotState) -> None:
        self._validate_names(state.actuator_names)
        self._last_command = np.asarray(state.actuator_positions, dtype=np.float64).copy()
        self._emergency_stop = False

    def engage_emergency_stop(self, reason: str = "operator emergency stop") -> None:
        self._emergency_stop = True
        self._emergency_reason = reason

    def clear_emergency_stop(self, state: RobotState) -> None:
        self._last_command = np.asarray(state.actuator_positions, dtype=np.float64).copy()
        self._emergency_stop = False

    def filter(
        self, command: ActuatorCommand, state: RobotState, *, dt_s: float
    ) -> SafetyDecision:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        self._validate_names(command.names)
        self._validate_names(state.actuator_names)
        if self._last_command is None:
            raise RuntimeError("JointLimitSafety.reset() must be called first")

        reasons: list[str] = []
        if state.base_position[2] < self.minimum_base_height_m:
            self.engage_emergency_stop("base height below safety threshold")
        maximum_velocity = np.max(np.abs(state.actuator_velocities), initial=0.0)
        if maximum_velocity > self.max_observed_velocity_rad_s:
            self.engage_emergency_stop("observed joint velocity exceeded safety threshold")

        if self._emergency_stop:
            hold = ActuatorCommand(names=self.names, values=state.actuator_positions)
            self._last_command = np.asarray(hold.values).copy()
            return SafetyDecision(
                command=hold,
                stopped=True,
                reasons=(self._emergency_reason,),
            )

        requested = np.asarray(command.values, dtype=np.float64)
        bounded = np.clip(requested, self.lower, self.upper)
        position_limited = ~np.isclose(requested, bounded, rtol=0.0, atol=1e-12)
        if np.any(position_limited):
            reasons.append("actuator position limit")

        max_delta = self.max_target_rate_rad_s * dt_s
        rate_bounded = np.clip(
            bounded,
            self._last_command - max_delta,
            self._last_command + max_delta,
        )
        rate_limited = ~np.isclose(bounded, rate_bounded, rtol=0.0, atol=1e-12)
        if np.any(rate_limited):
            reasons.append("actuator target rate limit")

        limited_mask = position_limited | rate_limited
        limited_names = tuple(
            name
            for name, limited in zip(self.names, limited_mask, strict=True)
            if limited
        )
        safe_command = ActuatorCommand(names=self.names, values=rate_bounded)
        self._last_command = np.asarray(safe_command.values).copy()
        return SafetyDecision(
            command=safe_command,
            limited_actuators=limited_names,
            reasons=tuple(reasons),
        )

    def _validate_names(self, names: tuple[str, ...]) -> None:
        if names != self.names:
            raise ValueError("Safety actuator order does not match the configured model")
