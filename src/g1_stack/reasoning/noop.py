from __future__ import annotations

from g1_stack.core.types import PhysicalIntent, RobotState


class NoOpReasoner:
    """Deterministic reasoner used before an LLM backend is connected."""

    def __init__(self, objective: str = "execute scripted pose demonstration") -> None:
        self.objective = objective

    def deliberate(self, state: RobotState) -> PhysicalIntent:
        del state
        return PhysicalIntent(
            objective=self.objective,
            constraints=("respect_joint_limits", "preserve_balance"),
            strategy_hint="follow the selected pose program",
            confidence=1.0,
        )

