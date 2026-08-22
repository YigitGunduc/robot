from __future__ import annotations

from g1_stack.core.types import ActuatorCommand, RobotState


class HoldPositionController:
    """Simple position-actuator baseline used to validate a simulator."""

    def __init__(self) -> None:
        self._command: ActuatorCommand | None = None

    def reset(self, state: RobotState) -> None:
        self._command = ActuatorCommand(
            names=state.actuator_names,
            values=state.actuator_positions,
        )

    def compute(self, state: RobotState) -> ActuatorCommand:
        del state
        if self._command is None:
            raise RuntimeError("HoldPositionController.reset() must be called first")
        return self._command

