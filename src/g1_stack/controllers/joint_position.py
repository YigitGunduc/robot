from __future__ import annotations

from g1_stack.core.types import ActuatorCommand, RobotState, WholeBodyReference


class JointPositionController:
    """V1 low-level controller that forwards a complete position reference.

    This preserves the existing MuJoCo behavior while giving SONIC a distinct
    replaceable boundary. It is not a learned balance controller.
    """

    def __init__(self) -> None:
        self._names: tuple[str, ...] | None = None

    def reset(self, state: RobotState) -> None:
        self._names = state.actuator_names

    def compute(
        self, state: RobotState, reference: WholeBodyReference
    ) -> ActuatorCommand:
        if self._names is None:
            raise RuntimeError("JointPositionController.reset() must be called first")
        if state.actuator_names != self._names:
            raise ValueError("Robot state actuator order changed after controller reset")
        if reference.joint_names != self._names:
            raise ValueError("Whole-body reference does not match controller actuator order")
        return ActuatorCommand(
            names=self._names,
            values=reference.joint_position_targets,
        )
