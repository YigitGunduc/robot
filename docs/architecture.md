# Initial architecture and gates

## Runtime contracts

```text
ReasoningProvider.deliberate(RobotState) -> PhysicalIntent
EmbodiedActor.act(RobotState, PhysicalIntent) -> ActuatorCommand
SafetySupervisor.filter(ActuatorCommand, RobotState) -> SafetyDecision
SimulatorBackend.step(ActuatorCommand) -> RobotState
```

`ActuatorCommand` contains ordered actuator names and values. The simulator checks
the names against the loaded model before stepping, which prevents silent joint-order
corruption.

`RobotRuntime` owns the loop and is the only component that composes these contracts:

```text
observe -> deliberate -> act -> safety -> simulate -> record
```

The initial `NoOpReasoner` and `ScriptedPoseActor` make this executable without a
GPU. They are test doubles for orchestration, not substitutes for SONIC. The safety
supervisor remains outside the actor so a future LLM, GR00T, or SONIC adapter cannot
bypass actuator limits, rate limits, fall detection, or emergency stop.

Episodes are append-only directories. `manifest.json` captures intent, configuration,
events, and termination; `episode.npz` captures state plus requested and applied
commands. Requested and applied commands are both retained so safety interventions
remain auditable.

SONIC is process-oriented rather than a Python `compute()` implementation. The
official deployment owns its 50 Hz loop, TensorRT engines, observation layout, and
Unitree transport. `SonicProcessAdapter` validates that external compatibility unit
and provides deterministic launch commands. A future in-process SONIC adapter must
pass parity tests against this reference before becoming selectable.

## Gate 1: simulator

- G1 MJCF loads from a pinned asset revision.
- Reset is deterministic.
- Actuator and joint ordering are explicit.
- One thousand headless steps produce finite state.
- Out-of-range or mismatched commands fail loudly.
- The full Python runtime holds the standing G1 stable through a scripted sequence.
- Every run produces a loadable episode with the command/state ordering preserved.

## Gate 2: official SONIC

- Container preflight reports TensorRT 10.13.
- Container source commit equals `SONIC_REF`.
- Official deployment builds inside the image.
- G1 initializes without a pose snap.
- Repeated reference motions are stable in official sim-to-sim.

## Gate 3: integration

- Record identical initial conditions in our simulator and the official simulator.
- Compare joint order, timestep, control mode, gains, base state, and contacts.
- Do not connect GR00T until unexplained divergence is removed or documented.
