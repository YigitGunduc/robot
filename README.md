# G1 robot stack

This repository is a small, runnable scaffold for the G1 roadmap. It deliberately
separates our application from NVIDIA SONIC and from any particular simulator.

The first runnable vertical slice is:

```text
MissionRequest -> NoOpReasoner -> PhysicalIntent -> ScriptedPoseActor
                                                        |
                                              WholeBodyReference
                                                        |
                                             JointPositionController
                                                        |
                                                ActuatorCommand
                                                        |
                                               JointLimitSafety
                                                        |
EpisodeRecorder <- RobotState <- MujocoBackend <--------+
```

This is deliberately a simple actor, not SONIC pretending to run on a CPU. It lets
us exercise the complete Python orchestration and data path locally before swapping
in a GPU-backed actor/controller.

The next integration is:

```text
GR00T actor -> SONIC process adapter -> official SONIC deploy -> G1/simulator
```

SONIC remains an external, pinned backend because its checkpoint, observation
configuration, C++ deployment code, CUDA, and TensorRT version are one compatibility
unit.

## Local setup

Python 3.11 or 3.12 is required. With `uv`:

```bash
uv sync --extra sim --extra dev
./scripts/fetch_mujoco_assets.sh
uv run g1-stack sim-info
uv run g1-stack sim-smoke --steps 1000
uv run g1-stack sim-run --duration 12
uv run pytest
```

The asset script fetches only `unitree_g1` from MuJoCo Menagerie and checks out the
commit pinned in `upstream.env`.

Override the model when needed:

```bash
uv run g1-stack sim-smoke --model /absolute/path/to/scene.xml --keyframe stand
```

## Interactive runtime

The runtime accepts a human/application `MissionRequest`, creates structured intent,
asks the actor for a whole-body reference, converts that through a low-level
controller, applies independent position/rate/fall safety, steps MuJoCo, and records
the episode. Run the desktop viewer on macOS through MuJoCo's `mjpython` launcher:

```bash
uv run mjpython -m g1_stack.cli sim-run --viewer --duration 0
```

On Linux, the ordinary entry point can launch the viewer:

```bash
uv run g1-stack sim-run --viewer --duration 0
```

Viewer keys are `0` for the automatic program, `1` neutral, `2` arms out, `3`/`4`
wave positions, `5` crouch, space to pause, `r` to reset, `e` for emergency stop,
and `q` to quit. A positive `--duration` stops automatically; zero runs until the
viewer closes or `q` is pressed.

Every `sim-run` creates a new directory under `artifacts/episodes` containing
`manifest.json` and compressed `episode.npz`. Inspect one with:

```bash
uv run g1-stack episode-info artifacts/episodes/<episode-directory>
```

The Python seams are intentionally explicit:

```text
ReasoningProvider.deliberate(MissionRequest, RobotState) -> PhysicalIntent
EmbodiedActor.act(RobotState, PhysicalIntent) -> WholeBodyReference
LowLevelController.compute(RobotState, WholeBodyReference) -> ActuatorCommand
```

`MissionRequest` is the human/application side of the boundary. The current
`NoOpReasoner` only passes its text through; it does not understand natural language.
A future Vesta/LLM adapter replaces that component. The current
`JointPositionController` forwards the actor's complete position reference; a future
SONIC adapter replaces that component. `RobotRuntime` composes all layers with the
simulator, safety supervisor, and recorder without allowing a reasoner or actor to
bypass safety.

## CPU simulation container

This image runs the same headless smoke test on macOS Docker, Linux, or a hosted
notebook environment that supports Docker:

```bash
docker compose build sim
docker compose run --rm sim
```

No NVIDIA GPU is required for this path. The image has been kept separate so a
developer machine never pulls CUDA or TensorRT merely to test simulation.

For a Colab runtime where Docker is unavailable, upload or clone this repository
and run the native setup instead:

```bash
./scripts/colab_sim_setup.sh
```

## SONIC inference container

The SONIC image is intentionally separate from the portable simulation image. It is
for a remote x86_64 Ubuntu host with an NVIDIA GPU and NVIDIA Container Toolkit. It
pins:

- NVIDIA CUDA/TensorRT image `25.08-py3` (TensorRT 10.13.2)
- `GR00T-WholeBodyControl` to the commit in `upstream.env`
- the official SONIC deployment build produced by that commit

It also downloads the matching public deployment checkpoint during the image build
and records a SHA-256 manifest at `/opt/sonic-model.sha256`.

Build and verify it on the NVIDIA host:

```bash
docker compose --profile sonic build sonic
docker compose --profile sonic run --rm sonic preflight
```

Open a prepared shell:

```bash
docker compose --profile sonic run --rm sonic shell
```

Run the official MuJoCo simulator and SONIC deploy in separate terminals. Host
networking is required for the Unitree DDS simulation transport:

```bash
# Terminal 1
docker compose --profile sonic run --rm sonic official-sim

# Terminal 2
docker compose --profile sonic run --rm sonic sonic-deploy --input-type keyboard sim
```

These commands are Linux/NVIDIA targets. They are not expected to run on this Mac.
The local `MujocoBackend` remains the portable development and contract-test backend.
Colab is suitable for the pure Python/headless simulation path, but ordinary Colab
runtimes do not reliably provide Docker daemon access, host networking, or NVIDIA
Container Toolkit. The SONIC container therefore targets a GPU VM or hosted GPU
service with Docker support rather than assuming Colab can run it.

## Boundaries

- `SimulatorBackend` owns simulation lifecycle and state extraction.
- `MissionRequest` carries human/application language and explicit constraints.
- `ReasoningProvider` turns the request and observations into `PhysicalIntent`.
- `EmbodiedActor` turns state plus intent into `WholeBodyReference` (GR00T's slot).
- `LowLevelController` turns that reference into `ActuatorCommand` (SONIC's slot).
- `SafetySupervisor` independently limits or vetoes every command.
- `EpisodeRecorder` writes replayable arrays and a human-readable manifest.
- `SonicProcessAdapter` validates and describes the pinned official SONIC runtime.
- No application code imports `gear_sonic` directly.
- Raw joint control remains available for tests; SONIC remains the production
  whole-body controller.

See `docs/architecture.md` for the contracts and acceptance gates.
