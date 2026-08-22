from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from g1_stack.actors.scripted_pose import ScriptedPoseActor
from g1_stack.controllers.hold_position import HoldPositionController
from g1_stack.data.episode import EpisodeRecorder, load_episode
from g1_stack.reasoning.noop import NoOpReasoner
from g1_stack.runtime.robot_runtime import RobotRuntime, RunConfig, RuntimeControl
from g1_stack.safety.joint_limits import JointLimitSafety
from g1_stack.sim.mujoco_backend import MujocoBackend, MujocoConfig
from g1_stack.sim.viewer import PassiveMujocoViewer

DEFAULT_MODEL = Path("third_party") / "mujoco_menagerie" / "unitree_g1" / "scene.xml"


def _model_path(value: str | None) -> Path:
    if value:
        return Path(value)
    environment_path = os.environ.get("G1_MJCF_PATH")
    return Path(environment_path) if environment_path else Path.cwd() / DEFAULT_MODEL


def _sim_info(args: argparse.Namespace) -> int:
    model_path = _model_path(args.model)
    with MujocoBackend(MujocoConfig(model_path=model_path)) as backend:
        state = backend.reset(keyframe=args.keyframe)
        print(f"model={model_path.resolve()}")
        print(f"nq={backend.model.nq} nv={backend.model.nv} nu={backend.model.nu}")
        print(f"timestep_s={backend.model.opt.timestep}")
        print(f"base_position={state.base_position.tolist()}")
        print("actuators:")
        for index, name in enumerate(backend.actuator_names):
            print(f"  {index:02d} {name}")
    return 0


def _sim_smoke(args: argparse.Namespace) -> int:
    model_path = _model_path(args.model)
    config = MujocoConfig(
        model_path=model_path,
        timestep_s=args.timestep,
        clamp_controls=args.clamp_controls,
    )
    with MujocoBackend(config) as backend:
        state = backend.reset(seed=args.seed, keyframe=args.keyframe)
        controller = HoldPositionController()
        controller.reset(state)
        for _ in range(args.steps):
            state = backend.step(controller.compute(state), frame_skip=args.frame_skip)

        expected_time = args.steps * args.frame_skip * args.timestep
        tolerance = max(1e-9, expected_time * 1e-9)
        if abs(state.time_s - expected_time) > tolerance:
            raise RuntimeError(
                f"Unexpected simulation time: expected {expected_time}, got {state.time_s}"
            )
        print(
            "simulation_ok "
            f"steps={args.steps} frame_skip={args.frame_skip} time_s={state.time_s:.6f} "
            f"actuators={len(state.actuator_names)} contacts={state.contact_count} "
            f"base_z={state.base_position[2]:.6f}"
        )
    return 0


def _sim_run(args: argparse.Namespace) -> int:
    model_path = _model_path(args.model)
    if args.steps is None:
        if args.duration <= 0 and not args.viewer:
            raise ValueError("An unlimited run requires --viewer")
        max_steps = (
            None
            if args.duration <= 0
            else math.ceil(args.duration / (args.timestep * args.frame_skip))
        )
    else:
        max_steps = args.steps

    config = MujocoConfig(model_path=model_path, timestep_s=args.timestep)
    with MujocoBackend(config) as backend:
        actor = ScriptedPoseActor()
        reasoner = NoOpReasoner(objective=args.objective)
        lower, upper = backend.actuator_control_bounds
        safety = JointLimitSafety(
            backend.actuator_names,
            lower,
            upper,
            max_target_rate_rad_s=args.max_target_rate,
            minimum_base_height_m=args.minimum_base_height,
        )
        recorder = EpisodeRecorder(Path(args.record_dir), label="g1-scripted")
        control = RuntimeControl()
        runtime = RobotRuntime(
            backend,
            reasoner,
            actor,
            safety,
            recorder,
            control=control,
        )
        run_config = RunConfig(
            max_steps=max_steps,
            frame_skip=args.frame_skip,
            seed=args.seed,
            keyframe=args.keyframe,
            realtime=args.realtime or args.viewer,
        )

        def on_key(keycode: int) -> None:
            key = chr(keycode).lower() if 0 <= keycode <= 0x10FFFF else ""
            pose_keys = {
                "1": "neutral",
                "2": "arms_out",
                "3": "left_wave_a",
                "4": "left_wave_b",
                "5": "small_crouch",
            }
            if key in pose_keys:
                actor.request_pose(pose_keys[key])
                print(f"pose={pose_keys[key]}")
            elif key == "0":
                actor.request_auto()
                print("pose=auto")
            elif key == " ":
                print(f"paused={control.toggle_pause()}")
            elif key == "r":
                control.request_reset()
                print("reset=requested")
            elif key == "e":
                safety.engage_emergency_stop()
                print("emergency_stop=engaged")
            elif key == "q":
                control.request_stop()

        control_dt_s = run_config.frame_skip * backend.timestep_s
        status_every = max(1, round(1.0 / control_dt_s))

        def show_status(state, decision, step: int) -> None:
            if step % status_every == 0:
                print(
                    f"state step={step} time_s={state.time_s:.2f} "
                    f"base_z={state.base_position[2]:.3f} contacts={state.contact_count} "
                    f"limited={len(decision.limited_actuators)}"
                )

        viewer_context = (
            PassiveMujocoViewer(backend, key_callback=on_key) if args.viewer else nullcontext()
        )
        if args.viewer:
            print(
                "controls: 0 auto | 1 neutral | 2 arms out | 3/4 wave | "
                "5 crouch | space pause | r reset | e emergency stop | q quit"
            )
        with viewer_context as viewer:
            summary = runtime.run(run_config, viewer=viewer, on_step=show_status)

    print(
        f"run_complete success={summary.success} steps={summary.steps} "
        f"time_s={summary.simulated_time_s:.3f} "
        f"safety_interventions={summary.safety_interventions} "
        f"reason={summary.stop_reason!r} episode={summary.episode_path}"
    )
    return 0 if summary.success else 3


def _episode_info(args: argparse.Namespace) -> int:
    episode = load_episode(Path(args.path))
    manifest = episode.manifest
    print(f"episode={episode.path}")
    print(
        f"success={manifest['success']} steps={manifest['steps']} "
        f"reason={manifest['stop_reason']!r}"
    )
    print(f"objective={manifest['intent']['objective']!r}")
    if episode.arrays["time_s"].size:
        print(
            f"time_s={episode.arrays['time_s'][-1]:.3f} "
            f"samples={episode.arrays['time_s'].size}"
        )
    return 0


def _add_sim_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="MJCF scene path; defaults to pinned Menagerie G1")
    parser.add_argument("--keyframe", default="stand", help="Reset keyframe name")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g1-stack")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("sim-info", help="Inspect the configured MuJoCo model")
    _add_sim_arguments(info)
    info.set_defaults(handler=_sim_info)

    smoke = subparsers.add_parser("sim-smoke", help="Run a deterministic headless smoke test")
    _add_sim_arguments(smoke)
    smoke.add_argument("--steps", type=int, default=1000)
    smoke.add_argument("--frame-skip", type=int, default=1)
    smoke.add_argument("--timestep", type=float, default=0.002)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--clamp-controls", action="store_true")
    smoke.set_defaults(handler=_sim_smoke)

    run = subparsers.add_parser("sim-run", help="Run the scripted safe G1 runtime")
    _add_sim_arguments(run)
    run.add_argument(
        "--duration", type=float, default=12.0, help="Simulated seconds; <=0 is unlimited"
    )
    run.add_argument("--steps", type=int, help="Control steps; overrides --duration")
    run.add_argument("--frame-skip", type=int, default=5)
    run.add_argument("--timestep", type=float, default=0.002)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer")
    run.add_argument("--realtime", action="store_true", help="Pace a headless run in wall time")
    run.add_argument("--record-dir", default="artifacts/episodes")
    run.add_argument("--objective", default="execute scripted pose demonstration")
    run.add_argument("--max-target-rate", type=float, default=2.0)
    run.add_argument("--minimum-base-height", type=float, default=0.35)
    run.set_defaults(handler=_sim_run)

    episode_info = subparsers.add_parser("episode-info", help="Inspect a recorded episode")
    episode_info.add_argument("path")
    episode_info.set_defaults(handler=_episode_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError, FloatingPointError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
