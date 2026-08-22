#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from g1_stack.actors.scripted_pose import ScriptedPoseActor
from g1_stack.controllers.joint_position import JointPositionController
from g1_stack.core.types import MissionRequest
from g1_stack.data.episode import EpisodeRecorder
from g1_stack.reasoning.noop import NoOpReasoner
from g1_stack.runtime.robot_runtime import RobotRuntime, RunConfig
from g1_stack.safety.joint_limits import JointLimitSafety
from g1_stack.sim.mujoco_backend import MujocoBackend, MujocoConfig


class FFmpegWriter:
    def __init__(self, output: Path, *, width: int, height: int, fps: int) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is required to encode the video")
        self.output = output.expanduser().resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = 0
        self._process = subprocess.Popen(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.output),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg input is closed")
        self._process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = self._process.stderr.read() if self._process.stderr is not None else b""
        return_code = self._process.wait()
        if return_code:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed with status {return_code}: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the scripted G1 demo to MP4")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("third_party/mujoco_menagerie/unitree_g1/scene.xml"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/videos/g1_demo.mp4"))
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("duration, fps, width, and height must be positive")

    timestep_s = 0.002
    frame_skip = 5
    max_steps = round(args.duration / (timestep_s * frame_skip))
    with MujocoBackend(MujocoConfig(model_path=args.model, timestep_s=timestep_s)) as backend:
        backend.model.vis.global_.offwidth = max(backend.model.vis.global_.offwidth, args.width)
        backend.model.vis.global_.offheight = max(backend.model.vis.global_.offheight, args.height)
        lower, upper = backend.actuator_control_bounds
        runtime = RobotRuntime(
            backend,
            NoOpReasoner(),
            ScriptedPoseActor(),
            JointPositionController(),
            JointLimitSafety(backend.actuator_names, lower, upper),
            EpisodeRecorder(Path("artifacts/episodes"), label="video-demo"),
        )
        writer = FFmpegWriter(args.output, width=args.width, height=args.height, fps=args.fps)
        frame_period_s = 1.0 / args.fps
        next_frame_time_s = frame_period_s

        def capture(state, decision, step: int) -> None:
            nonlocal next_frame_time_s
            del decision, step
            while state.time_s + 1e-12 >= next_frame_time_s:
                writer.write(backend.render(width=args.width, height=args.height))
                next_frame_time_s += frame_period_s

        try:
            summary = runtime.run(
                RunConfig(max_steps=max_steps, frame_skip=frame_skip, keyframe="stand"),
                request=MissionRequest("render scripted G1 pose demonstration"),
                on_step=capture,
            )
        finally:
            writer.close()

    print(
        f"video_complete path={writer.output} frames={writer.frames} "
        f"steps={summary.steps} episode={summary.episode_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
