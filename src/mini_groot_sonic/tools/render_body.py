from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from mini_groot_sonic.config import load_project_config
from mini_groot_sonic.data.motion_bank import MotionBank
from mini_groot_sonic.models.runtime import load_body_checkpoint
from mini_groot_sonic.sim.math_utils import quat_distance_angle
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv


def _camera(mujoco, root_position: np.ndarray, distance: float):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = root_position
    camera.distance = distance
    camera.azimuth = 140.0
    camera.elevation = -15.0
    return camera


def _label_frame(
    policy_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    caption: str,
    step: int,
    errors: dict[str, float],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    combined = np.concatenate([policy_rgb, reference_rgb], axis=1)
    image = Image.fromarray(combined)
    draw = ImageDraw.Draw(image)
    width = image.width
    draw.rectangle((0, 0, width, 52), fill=(0, 0, 0))
    draw.text((12, 8), "LEARNED POLICY", fill=(80, 255, 120))
    draw.text((width // 2 + 12, 8), "BONES REFERENCE", fill=(100, 190, 255))
    safe_caption = caption.encode("ascii", errors="replace").decode("ascii")
    status = (
        f"{safe_caption[:70]} | step {step} | root {errors['root_position_error']:.3f} m | "
        f"MPJPE {errors['mpjpe']:.3f} m | joint {errors['joint_position_error']:.3f} rad"
    )
    draw.text((12, 30), status, fill=(255, 255, 255))
    return np.asarray(image)


@torch.no_grad()
def render_body_rollout(
    motion_path: Path,
    mjcf: Path,
    checkpoint: Path,
    config: Path,
    output: Path,
    *,
    device: str,
    width: int,
    height: int,
    fps: int,
    render_stride: int,
    max_steps: int,
    camera_distance: float,
) -> dict[str, float | str]:
    # Must be selected before MuJoCo initializes OpenGL in this fresh process.
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("Install video extras: pip install -e '.[video]'") from exc
    import mujoco

    cfg = load_project_config(config)
    cfg.sim.mjcf = mjcf
    cfg.sim.device = device
    cfg.sim.enable_randomization = False
    policy, sonic_cfg = load_body_checkpoint(checkpoint, device)
    bank = MotionBank([motion_path], sonic_cfg, device)
    env = MJWarpG1VecEnv(cfg.sim, sonic_cfg, 1)
    if bank.body_names != env.body_names:
        raise ValueError("Preprocessed body_names do not match the rendering MJCF")

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

    policy_renderer = mujoco.Renderer(env.mjm, height=height, width=width)
    reference_renderer = mujoco.Renderer(env.mjm, height=height, width=width)
    reference_data = mujoco.MjData(env.mjm)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    metric_sums = {
        "root_position_error": 0.0,
        "root_orientation_error": 0.0,
        "mpjpe": 0.0,
        "joint_position_error": 0.0,
    }
    samples = 0
    rendered_frames = 0
    future_span = (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride + 1
    available_steps = max(1, int(bank.lengths[0]) - future_span)
    rollout_steps = min(available_steps, max_steps) if max_steps > 0 else available_steps

    try:
        for step in range(rollout_steps):
            safe_frame = torch.minimum(frame_ids, bank.lengths[motion_ids] - 1)
            ref_now = bank.current_reference(motion_ids, safe_frame)
            errors = {
                "root_position_error": float((obs.root_pos - ref_now["root_pos"]).norm(dim=-1)[0]),
                "root_orientation_error": float(
                    quat_distance_angle(obs.root_quat, ref_now["root_quat"])[0]
                ),
                "mpjpe": float((obs.body_pos - ref_now["body_pos"]).norm(dim=-1).mean(-1)[0]),
                "joint_position_error": float(
                    (obs.joint_pos - ref_now["joint_pos"]).abs().mean(-1)[0]
                ),
            }
            for name, value in errors.items():
                metric_sums[name] += value
            samples += 1

            if step % render_stride == 0:
                actual_data = env.sync_world0_to_cpu()
                reference_data.qpos[:] = env.mjm.qpos0
                rp = env.map.root_qpos_adr
                rv = env.map.root_dof_adr
                reference_data.qpos[rp : rp + 3] = ref_now["root_pos"][0].cpu().numpy()
                reference_data.qpos[rp + 3 : rp + 7] = ref_now["root_quat"][0].cpu().numpy()
                reference_data.qpos[env.map.joint_qpos_adr] = ref_now["joint_pos"][0].cpu().numpy()
                reference_data.qvel[:] = 0
                reference_data.qvel[rv : rv + 3] = ref_now["root_linvel"][0].cpu().numpy()
                reference_data.qvel[rv + 3 : rv + 6] = ref_now["root_angvel"][0].cpu().numpy()
                reference_data.qvel[env.map.joint_dof_adr] = ref_now["joint_vel"][0].cpu().numpy()
                mujoco.mj_forward(env.mjm, reference_data)

                lookat = 0.5 * (actual_data.qpos[rp : rp + 3] + reference_data.qpos[rp : rp + 3])
                camera = _camera(mujoco, lookat, camera_distance)
                policy_renderer.update_scene(actual_data, camera=camera)
                reference_renderer.update_scene(reference_data, camera=camera)
                frame = _label_frame(
                    policy_renderer.render().copy(),
                    reference_renderer.render().copy(),
                    bank.captions[0],
                    step,
                    errors,
                )
                writer.append_data(frame)
                rendered_frames += 1

            future = bank.future_reference(motion_ids, frame_ids)
            action = policy.act_deterministic(env.proprio_history(), future).action_mean
            obs = env.step(action)
            frame_ids += 1
    finally:
        writer.close()
        policy_renderer.close()
        reference_renderer.close()

    metrics: dict[str, float | str] = {
        name: value / max(samples, 1) for name, value in metric_sums.items()
    }
    metrics.update(
        {
            "motion_id": bank.motion_names[0],
            "caption": bank.captions[0],
            "rollout_steps": float(samples),
            "rendered_frames": float(rendered_frames),
            "video": str(output),
        }
    )
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Render learned policy vs BONES reference to MP4")
    ap.add_argument("--motions", required=True, help="Directory containing held-out preprocessed clips")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--motion-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=480, help="Width of each side-by-side panel")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--render-stride", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=750, help="0 renders the entire usable clip")
    ap.add_argument("--camera-distance", type=float, default=3.0)
    args = ap.parse_args()

    paths = sorted(Path(args.motions).glob("*.npz"))
    if not paths:
        raise SystemExit("No held-out preprocessed motion NPZ files found")
    if not 0 <= args.motion_index < len(paths):
        raise SystemExit(f"motion-index must be in [0, {len(paths) - 1}]")
    metrics = render_body_rollout(
        paths[args.motion_index],
        Path(args.mjcf),
        Path(args.body),
        Path(args.config),
        Path(args.out),
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render_stride=max(1, args.render_stride),
        max_steps=args.max_steps,
        camera_distance=args.camera_distance,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
