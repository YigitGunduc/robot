from __future__ import annotations

from typing import Protocol, runtime_checkable

from g1_stack.core.types import (
    ActuatorCommand,
    MissionRequest,
    PhysicalIntent,
    RobotState,
    SafetyDecision,
    WholeBodyReference,
)


@runtime_checkable
class SimulatorBackend(Protocol):
    @property
    def actuator_names(self) -> tuple[str, ...]: ...

    @property
    def timestep_s(self) -> float: ...

    def reset(self, *, seed: int = 0, keyframe: str | None = None) -> RobotState: ...

    def step(self, command: ActuatorCommand, *, frame_skip: int = 1) -> RobotState: ...

    def close(self) -> None: ...


@runtime_checkable
class LowLevelController(Protocol):
    def reset(self, state: RobotState) -> None: ...

    def compute(
        self, state: RobotState, reference: WholeBodyReference
    ) -> ActuatorCommand: ...


@runtime_checkable
class EmbodiedActor(Protocol):
    def reset(self, state: RobotState, intent: PhysicalIntent) -> None: ...

    def act(self, state: RobotState, intent: PhysicalIntent) -> WholeBodyReference: ...


@runtime_checkable
class ReasoningProvider(Protocol):
    def deliberate(self, request: MissionRequest, state: RobotState) -> PhysicalIntent: ...


@runtime_checkable
class SafetySupervisor(Protocol):
    def reset(self, state: RobotState) -> None: ...

    def filter(
        self, command: ActuatorCommand, state: RobotState, *, dt_s: float
    ) -> SafetyDecision: ...
