from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_groot_sonic.config import load_project_config
from mini_groot_sonic.data.motion_bank import MotionBank
from mini_groot_sonic.models.frozen_backbones import FrozenSiglip2
from mini_groot_sonic.models.runtime import (
    RecedingHorizonTokenController,
    load_body_checkpoint,
    load_flow_checkpoint,
)
from mini_groot_sonic.sim.math_utils import quat_rotate_inverse
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv


def _tilt_angle(obs) -> torch.Tensor:
    gravity_world = torch.zeros_like(obs.root_angvel)
    gravity_world[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(obs.root_quat, gravity_world)
    return torch.acos((-projected_gravity[:, 2]).clamp(-1.0, 1.0))


def _require_upright(obs, *, min_root_height: float, max_tilt: float, phase: str) -> None:
    height = float(obs.root_pos[0, 2])
    tilt = float(_tilt_angle(obs)[0])
    if height < min_root_height or tilt > max_tilt:
        raise RuntimeError(
            f"Robot fell during {phase}: root_height={height:.3f}, tilt={tilt:.3f} rad"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a natural-language body command through flow tokens -> tiny SONIC controller")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--body", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument(
        "--initial-motion",
        required=True,
        help="Preprocessed standing/balance NPZ used to initialize and warm up the body policy",
    )
    ap.add_argument("--command", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--min-root-height", type=float, default=0.5)
    ap.add_argument("--max-tilt", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hf-model", default="google/siglip2-base-patch16-224")
    args = ap.parse_args()

    cfg = load_project_config(args.config)
    device = args.device or cfg.sim.device
    body, sonic_cfg, sim_cfg = load_body_checkpoint(args.body, device)
    sim_cfg.mjcf = Path(args.mjcf)
    sim_cfg.device = device
    sim_cfg.enable_randomization = False
    sim_cfg.enable_observation_noise = False
    env = MJWarpG1VecEnv(sim_cfg, sonic_cfg, 1)
    flow, _ = load_flow_checkpoint(
        args.flow,
        device,
        expected_body_fingerprint=body.checkpoint_fingerprint,
    )
    backbone = FrozenSiglip2(args.hf_model, device)
    runner = RecedingHorizonTokenController(flow, body, backbone)

    bank = MotionBank([Path(args.initial_motion)], sonic_cfg, device)
    if bank.body_names != env.body_names:
        raise ValueError("Initial-motion body_names do not match the runtime MJCF")
    motion_ids = torch.zeros(1, dtype=torch.long, device=device)
    frame_ids = torch.zeros(1, dtype=torch.long, device=device)
    ref = bank.current_reference(motion_ids, frame_ids)
    obs = env.reset(
        ref["root_pos"],
        ref["root_quat"],
        ref["joint_pos"],
        ref["root_linvel"],
        ref["root_angvel"],
        ref["joint_vel"],
    )

    max_start = (
        bank.lengths[motion_ids]
        - (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride
        - 1
    ).clamp_min(0)
    for _ in range(max(0, args.warmup_steps)):
        future_ids = torch.minimum(frame_ids, max_start)
        future = bank.future_reference(motion_ids, future_ids, obs.root_quat)
        action = body.act_deterministic(env.proprio_history(), future).action_mean
        obs = env.step(action)
        frame_ids += 1
        _require_upright(
            obs,
            min_root_height=args.min_root_height,
            max_tilt=args.max_tilt,
            phase="body-policy warmup",
        )

    for i in range(args.steps):
        action, token = runner.action(args.command, obs, env.proprio_history())
        obs = env.step(action)
        _require_upright(
            obs,
            min_root_height=args.min_root_height,
            max_tilt=args.max_tilt,
            phase=f"command step {i}",
        )
        if i % 50 == 0:
            print(
                f"step={i} root={obs.root_pos[0].tolist()} "
                f"token_norm={float(token.norm()):.3f} action_norm={float(action.norm()):.3f}"
            )


if __name__ == "__main__":
    main()
