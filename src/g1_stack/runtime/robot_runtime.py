from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from g1_stack.core.interfaces import (
    EmbodiedActor,
    LowLevelController,
    ReasoningProvider,
    SafetySupervisor,
    SimulatorBackend,
)
from g1_stack.core.types import MissionRequest, RobotState, SafetyDecision
from g1_stack.data.episode import EpisodeRecorder


class RuntimeViewer(Protocol):
    def is_running(self) -> bool: ...

    def sync(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RunConfig:
    max_steps: int | None = 3000
    frame_skip: int = 5
    seed: int = 0
    keyframe: str | None = "stand"
    realtime: bool = False

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or None")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")


@dataclass(frozen=True, slots=True)
class RunSummary:
    steps: int
    simulated_time_s: float
    safety_interventions: int
    success: bool
    stop_reason: str
    episode_path: Path


class RuntimeControl:
    """Thread-safe controls shared with MuJoCo's viewer callback."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._paused = False
        self._stop = False
        self._reset = False

    def toggle_pause(self) -> bool:
        with self._lock:
            self._paused = not self._paused
            return self._paused

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True

    def request_reset(self) -> None:
        with self._lock:
            self._reset = True

    def snapshot(self) -> tuple[bool, bool, bool]:
        with self._lock:
            reset = self._reset
            self._reset = False
            return self._paused, self._stop, reset


class RobotRuntime:
    """Own the observe -> reason -> act -> safety -> simulate -> record loop."""

    def __init__(
        self,
        simulator: SimulatorBackend,
        reasoner: ReasoningProvider,
        actor: EmbodiedActor,
        controller: LowLevelController,
        safety: SafetySupervisor,
        recorder: EpisodeRecorder,
        *,
        control: RuntimeControl | None = None,
    ) -> None:
        self.simulator = simulator
        self.reasoner = reasoner
        self.actor = actor
        self.controller = controller
        self.safety = safety
        self.recorder = recorder
        self.control = control or RuntimeControl()

    def run(
        self,
        config: RunConfig,
        *,
        request: MissionRequest | None = None,
        viewer: RuntimeViewer | None = None,
        on_step: Callable[[RobotState, SafetyDecision, int], None] | None = None,
    ) -> RunSummary:
        mission = request or MissionRequest("execute scripted pose demonstration")
        state = self.simulator.reset(seed=config.seed, keyframe=config.keyframe)
        intent = self.reasoner.deliberate(mission, state)
        self.actor.reset(state, intent)
        self.controller.reset(state)
        self.safety.reset(state)
        episode_path = self.recorder.start(
            state,
            mission,
            intent,
            configuration=asdict(config),
        )
        steps = 0
        safety_interventions = 0
        success = False
        stop_reason = "runtime error"
        next_tick = time.perf_counter()
        control_dt_s = config.frame_skip * self.simulator.timestep_s

        try:
            while config.max_steps is None or steps < config.max_steps:
                if viewer is not None and not viewer.is_running():
                    stop_reason = "viewer closed"
                    success = True
                    break

                paused, stop_requested, reset_requested = self.control.snapshot()
                if stop_requested:
                    stop_reason = "operator stop"
                    success = True
                    break
                if reset_requested:
                    state = self.simulator.reset(seed=config.seed, keyframe=config.keyframe)
                    intent = self.reasoner.deliberate(mission, state)
                    self.actor.reset(state, intent)
                    self.controller.reset(state)
                    self.safety.reset(state)
                    self.recorder.event("reset", step=steps)
                    next_tick = time.perf_counter()

                if paused:
                    if viewer is not None:
                        viewer.sync()
                    time.sleep(0.01)
                    next_tick = time.perf_counter()
                    continue

                reference = self.actor.act(state, intent)
                requested = self.controller.compute(state, reference)
                decision = self.safety.filter(requested, state, dt_s=control_dt_s)
                if decision.limited_actuators or decision.stopped:
                    safety_interventions += 1
                if decision.stopped:
                    self.recorder.event("safety_stop", step=steps, reasons=list(decision.reasons))
                    stop_reason = "; ".join(decision.reasons) or "safety stop"
                    break

                state = self.simulator.step(decision.command, frame_skip=config.frame_skip)
                self.recorder.record(state, reference, requested, decision)
                steps += 1
                if on_step is not None:
                    on_step(state, decision, steps)

                if viewer is not None:
                    viewer.sync()
                if config.realtime:
                    next_tick += control_dt_s
                    remaining = next_tick - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    elif remaining < -control_dt_s:
                        next_tick = time.perf_counter()
            else:
                success = True
                stop_reason = "step limit reached"
        except BaseException as error:
            stop_reason = f"{type(error).__name__}: {error}"
            self.recorder.event("runtime_error", step=steps, error=stop_reason)
            raise
        finally:
            self.recorder.close(success=success, stop_reason=stop_reason, steps=steps)

        return RunSummary(
            steps=steps,
            simulated_time_s=state.time_s,
            safety_interventions=safety_interventions,
            success=success,
            stop_reason=stop_reason,
            episode_path=episode_path,
        )
