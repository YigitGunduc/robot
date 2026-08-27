from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from mini_groot_sonic.checkpoint import BODY_CONTROL_STACK_VERSION
from mini_groot_sonic.config import ReplayConfig, SimConfig, SonicTinyConfig
from mini_groot_sonic.data.episode_writer import EpisodeWriter
from mini_groot_sonic.data.reference import make_reference_features
from mini_groot_sonic.models.runtime import load_body_checkpoint
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


def load_policy_checkpoint(
    path: str | Path,
    device: str,
    cfg: SonicTinyConfig | None = None,
) -> tuple[TinySonicPolicy, SonicTinyConfig, SimConfig]:
    policy, checkpoint_cfg, sim_cfg = load_body_checkpoint(path, device)
    if cfg is not None and cfg != checkpoint_cfg:
        raise ValueError("Explicit SonicTinyConfig does not match the body checkpoint")
    return policy, checkpoint_cfg, sim_cfg


def _future_ref(
    q: torch.Tensor,
    qd: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_linvel: torch.Tensor,
    root_angvel: torch.Tensor,
    i: int,
    cfg: SonicTinyConfig,
    robot_root_quat: torch.Tensor | None = None,
) -> torch.Tensor:
    ids = torch.arange(cfg.future_frames, device=q.device) * cfg.future_stride + i
    ids = ids.clamp_max(q.shape[0] - 1)
    return make_reference_features(
        q[ids][None],
        qd[ids][None],
        root_pos[ids][None],
        root_quat[ids][None],
        root_linvel[ids][None],
        root_angvel[ids][None],
        robot_root_quat,
    )


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
    root_linvel = torch.from_numpy(
        np.asarray(data["root_linvel"] if "root_linvel" in data.files else data["body_linvel"][:, 0], np.float32)
    ).to(device)
    root_angvel = torch.from_numpy(
        np.asarray(data["root_angvel"] if "root_angvel" in data.files else data["body_angvel"][:, 0], np.float32)
    ).to(device)
    caption = str(data["caption"].item())
    captions = [str(x) for x in data["captions"].tolist()] if "captions" in data.files else [caption]
    motion_id = str(data["motion_id"].item())
    actor_uid = str(data["actor_uid"].item()) if "actor_uid" in data.files else None
    source_motion_id = (
        str(data["source_motion_id"].item()) if "source_motion_id" in data.files else motion_id
    )

    env = MJWarpG1VecEnv(sim_cfg, sonic_cfg, num_envs=1)
    obs = env.reset(
        root_pos[:1],
        root_quat[:1],
        q[:1],
        root_linvel=root_linvel[:1],
        root_angvel=root_angvel[:1],
        joint_vel=qd[:1],
    )
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
    target_body_idx = [env.body_names.index(n) for n in sim_cfg.keypoint_body_names]

    max_i = max(1, len(q) - (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride)
    for i in range(max_i):
        future = _future_ref(
            q,
            qd,
            root_pos,
            root_quat,
            root_linvel,
            root_angvel,
            i,
            sonic_cfg,
            obs.root_quat,
        )
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
                action = out.action_mean.clamp(
                    -env.action_clip_value,
                    env.action_clip_value,
                )
            elif mode == "reference_pd":
                # Bootstrap replay: transform desired q into SONIC residual actions.
                action = env.target_to_action(q[i : i + 1])
            else:
                raise ValueError("mode must be 'policy' or 'reference_pd'")

        # Store the state that produced this token/action. This causal alignment is
        # required by the flow-policy training target.
        # Root + head/wrists/ankles are stored in world SE(3); the dataset later
        # converts a future goal into the current root frame.
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
        obs = env.step(action)

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
            "actor_uid": actor_uid,
            "source_motion_id": source_motion_id,
            "body_control_stack_version": BODY_CONTROL_STACK_VERSION,
            "body_policy_fingerprint": getattr(
                policy, "checkpoint_fingerprint", None
            ),
        },
    )
