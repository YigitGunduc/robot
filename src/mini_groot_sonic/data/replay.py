from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from mini_groot_sonic.config import ReplayConfig, SimConfig, SonicTinyConfig
from mini_groot_sonic.data.episode_writer import EpisodeWriter
from mini_groot_sonic.models.sonic_tiny import TinySonicPolicy
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv


class ReplayHook(Protocol):
    def reset(self) -> None: ...
    def observe(self, step: int, env: MJWarpG1VecEnv) -> dict[str, np.ndarray]: ...


class NullReplayHook:
    def reset(self) -> None:
        pass

    def observe(self, step: int, env: MJWarpG1VecEnv) -> dict[str, np.ndarray]:
        return {}


class SingleWorldRGBHook:
    """Optional debug RGB capture by copying world 0 from MJWarp to CPU MuJoCo.

    Physics remains GPU/MJWarp. Rendering is intentionally single-world and low-rate;
    replace this hook with MJWarp's batch renderer for large visual datasets.
    """

    def __init__(self, env: MJWarpG1VecEnv, cfg: ReplayConfig):
        if env.num_envs != 1:
            raise ValueError("SingleWorldRGBHook requires num_envs=1")
        self.env = env
        self.cfg = cfg
        self.renderer = env.mujoco.Renderer(env.mjm, height=cfg.height, width=cfg.width)
        self.stride = max(1, round(cfg.control_hz / cfg.camera_hz))

    def reset(self) -> None:
        pass

    def observe(self, step: int, env: MJWarpG1VecEnv) -> dict[str, np.ndarray]:
        if step % self.stride:
            return {}
        mjd = env.sync_world0_to_cpu()
        self.renderer.update_scene(mjd, camera=self.cfg.camera_name)
        rgb = self.renderer.render().copy()
        return {"rgb": rgb, "rgb_step": np.asarray(step, dtype=np.int32)}


def load_policy_checkpoint(path: str | Path, cfg: SonicTinyConfig, device: str) -> TinySonicPolicy:
    ckpt = torch.load(path, map_location=device)
    policy = TinySonicPolicy(cfg).to(device)
    policy.load_state_dict(ckpt["policy"] if "policy" in ckpt else ckpt)
    policy.eval()
    return policy


def _future_ref(q: torch.Tensor, qd: torch.Tensor, i: int, cfg: SonicTinyConfig) -> torch.Tensor:
    ids = torch.arange(cfg.future_frames, device=q.device) * cfg.future_stride + i
    ids = ids.clamp_max(q.shape[0] - 1)
    return torch.cat([q[ids], qd[ids]], dim=-1)[None]


def collect_preprocessed_episode(
    clip_path: str | Path,
    sim_cfg: SimConfig,
    sonic_cfg: SonicTinyConfig,
    replay_cfg: ReplayConfig,
    writer: EpisodeWriter,
    *,
    policy: TinySonicPolicy | None = None,
    mode: str = "policy",
    hook: ReplayHook | None = None,
) -> Path:
    data = np.load(clip_path, allow_pickle=True)
    device = torch.device(sim_cfg.device)
    q = torch.from_numpy(np.asarray(data["joint_pos"], np.float32)).to(device)
    qd = torch.from_numpy(np.asarray(data["joint_vel"], np.float32)).to(device)
    root_pos = torch.from_numpy(np.asarray(data["root_pos"], np.float32)).to(device)
    root_quat = torch.from_numpy(np.asarray(data["root_quat"], np.float32)).to(device)
    caption = str(data["caption"].item())
    captions = [str(x) for x in data["captions"].tolist()] if "captions" in data.files else [caption]
    motion_id = str(data["motion_id"].item())

    env = MJWarpG1VecEnv(sim_cfg, sonic_cfg, num_envs=1)
    env.reset(root_pos[:1], root_quat[:1], q[:1], joint_vel=qd[:1])
    if hook is None:
        hook = SingleWorldRGBHook(env, replay_cfg) if replay_cfg.save_rgb else NullReplayHook()
    hook.reset()

    rows: dict[str, list[np.ndarray]] = {
        k: [] for k in (
            "q", "qdot", "root_pos", "root_quat", "root_linvel", "root_angvel",
            "token", "action", "reference_q", "reference_qdot", "body_pos", "body_quat",
            "goal_slots",
        )
    }
    rgb, rgb_steps = [], []
    target_body_names = [sim_cfg.root_body_name, *sim_cfg.keypoint_body_names]
    root_body_idx = env.body_names.index(sim_cfg.root_body_name) if sim_cfg.root_body_name in env.body_names else 0
    target_body_idx = [env.body_names.index(n) for n in sim_cfg.keypoint_body_names]

    max_i = max(1, len(q) - (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride)
    for i in range(max_i):
        future = _future_ref(q, qd, i, sonic_cfg)
        prop = env.proprio_history()
        with torch.no_grad():
            if policy is not None:
                out = policy(prop, future)
                token = out.token
            else:
                token = torch.zeros(1, sonic_cfg.token_dim, device=device)

            if mode == "policy":
                if policy is None:
                    raise ValueError("mode='policy' requires a trained TinySonicPolicy")
                action = out.action_mean.clamp(-1.0, 1.0)
            elif mode == "reference_pd":
                # Bootstrap replay: transform desired q into the project's normalized position-offset action space.
                action = ((q[i : i + 1] - env._default_joint_pos) / sim_cfg.action_scale).clamp(-1.0, 1.0)
            else:
                raise ValueError("mode must be 'policy' or 'reference_pd'")

        obs = env.step(action)
        # root + head/wrists/ankles as sparse goal slots, all in world SE(3) for storage.
        slot_pos = torch.cat([obs.root_pos[:, None], obs.body_pos[:, target_body_idx]], dim=1)
        slot_quat = torch.cat([obs.root_quat[:, None], obs.body_quat[:, target_body_idx]], dim=1)
        goal_slots = torch.cat([slot_pos, slot_quat], dim=-1)

        rows["q"].append(obs.joint_pos[0].detach().cpu().numpy())
        rows["qdot"].append(obs.joint_vel[0].detach().cpu().numpy())
        rows["root_pos"].append(obs.root_pos[0].detach().cpu().numpy())
        rows["root_quat"].append(obs.root_quat[0].detach().cpu().numpy())
        rows["root_linvel"].append(obs.root_linvel[0].detach().cpu().numpy())
        rows["root_angvel"].append(obs.root_angvel[0].detach().cpu().numpy())
        rows["token"].append(token[0].detach().cpu().numpy())
        rows["action"].append(action[0].detach().cpu().numpy())
        rows["reference_q"].append(q[i].detach().cpu().numpy())
        rows["reference_qdot"].append(qd[i].detach().cpu().numpy())
        rows["body_pos"].append(obs.body_pos[0].detach().cpu().numpy())
        rows["body_quat"].append(obs.body_quat[0].detach().cpu().numpy())
        rows["goal_slots"].append(goal_slots[0].detach().cpu().numpy())

        extra = hook.observe(i, env)
        if "rgb" in extra:
            rgb.append(extra["rgb"])
            rgb_steps.append(int(extra.get("rgb_step", i)))

    arrays = {k: np.stack(v).astype(np.float32) for k, v in rows.items()}
    if rgb:
        arrays["rgb"] = np.stack(rgb).astype(np.uint8)
        arrays["rgb_step"] = np.asarray(rgb_steps, dtype=np.int32)
    return writer.write(
        motion_id,
        arrays,
        {
            "caption": caption,
            "captions": captions,
            "source_clip": str(clip_path),
            "mode": mode,
            "control_hz": replay_cfg.control_hz,
            "body_names": env.body_names,
            "goal_slot_names": target_body_names,
        },
    )
