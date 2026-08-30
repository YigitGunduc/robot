#!/usr/bin/env python3
"""Render one trained checkpoint without rerunning training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import sonic_lite_g1  # noqa: F401  # Register the local task.


TASK = "Mjlab-SonicLite-Tracking-Flat-Unitree-G1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-length", type=int, default=600)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    motion_file = args.motion_file.expanduser().resolve()
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    if not motion_file.exists():
        raise SystemExit(f"Motion file does not exist: {motion_file}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = "egl"
    configure_torch_backends()
    device = "cpu"

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    motion_cmd = env_cfg.commands["motion"]
    motion_cmd.motion_file = str(motion_file)
    agent_cfg = load_rl_cfg(TASK)

    # This is deliberately headless: it never starts mjlab's interactive
    # Native/Viser viewer, which is the component that can segfault in Colab.
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK)
    runner = (runner_cls or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    frames: list[np.ndarray] = []
    obs = env.get_observations()
    with torch.inference_mode():
        for index in range(args.video_length):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            frame = env.unwrapped.render()
            if frame is None:
                raise RuntimeError("The environment did not return an RGB frame")
            frame = frame[0] if frame.ndim == 4 else frame
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            frames.append(frame)
            if (index + 1) % 100 == 0:
                print(f"rendered {index + 1}/{args.video_length} frames", flush=True)
    env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(args.output), frames, fps=50)
    print(f"wrote rendered MP4: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
