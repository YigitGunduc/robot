from __future__ import annotations

from g1_stack.core.types import MissionRequest, PhysicalIntent, RobotState


class NoOpReasoner:
    """Pass a mission through as intent before a language model is connected."""

    def deliberate(self, request: MissionRequest, state: RobotState) -> PhysicalIntent:
        del state
        constraints = tuple(
            dict.fromkeys((*request.constraints, "respect_joint_limits", "preserve_balance"))
        )
        return PhysicalIntent(
            objective=request.text,
            constraints=constraints,
            strategy_hint="follow the selected pose program",
            confidence=1.0,
        )
