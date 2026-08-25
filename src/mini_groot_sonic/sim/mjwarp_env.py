from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mini_groot_sonic.config import SimConfig, SonicTinyConfig
from mini_groot_sonic.sim.g1_mapping import G1ModelMap
from mini_groot_sonic.sim.math_utils import quat_rotate_inverse


@dataclass
class EnvObservation:
    proprio_frame: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_linvel: torch.Tensor
    root_angvel: torch.Tensor
    body_pos: torch.Tensor
    body_quat: torch.Tensor
    body_angvel: torch.Tensor
    body_linvel: torch.Tensor


class MJWarpG1VecEnv:
    """Thin PyTorch-friendly wrapper around MuJoCo Warp batched simulation.

    Physics remains in MJWarp/Warp on NVIDIA GPU. PyTorch obtains zero-copy views
    through Warp's torch interop where possible. The class intentionally does not
    depend on Isaac Lab or JAX.
    """

    def __init__(self, sim_cfg: SimConfig, sonic_cfg: SonicTinyConfig, num_envs: int):
        try:
            import mujoco
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as exc:
            raise ImportError("Install simulator extras: pip install -e '.[sim]'") from exc

        self.mujoco = mujoco
        self.mjw = mjw
        self.wp = wp
        self.sim_cfg = sim_cfg
        self.sonic_cfg = sonic_cfg
        self.num_envs = num_envs

        wp.init()
        wp.set_device(sim_cfg.device)
        self.mjm = mujoco.MjModel.from_xml_path(str(sim_cfg.mjcf))
        self.mjm.opt.timestep = sim_cfg.physics_dt
        self.map = G1ModelMap.from_mjmodel(
            self.mjm,
            sim_cfg.root_body_name,
            tuple(sim_cfg.keypoint_body_names),
        )
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.make_data(
            self.mjm,
            nworld=num_envs,
            nconmax=sim_cfg.nconmax,
            njmax=sim_cfg.njmax,
        )

        self.device = torch.device(sim_cfg.device)
        self._qpos = wp.to_torch(self.d.qpos)
        self._qvel = wp.to_torch(self.d.qvel)
        self._ctrl = wp.to_torch(self.d.ctrl)
        self._xpos = wp.to_torch(self.d.xpos)
        self._xquat = wp.to_torch(self.d.xquat)
        self._cvel = wp.to_torch(self.d.cvel)
        self._last_action = torch.zeros(num_envs, sonic_cfg.dof, device=self.device)
        self._history = torch.zeros(
            num_envs,
            sonic_cfg.prop_history,
            sonic_cfg.proprio_dim_per_frame,
            device=self.device,
        )

        self._joint_qpos_adr = torch.as_tensor(self.map.joint_qpos_adr, device=self.device)
        self._joint_dof_adr = torch.as_tensor(self.map.joint_dof_adr, device=self.device)
        self._actuator_ids = torch.as_tensor(self.map.actuator_ids, device=self.device)
        self._default_joint_pos = torch.as_tensor(self.map.default_joint_pos, device=self.device)
        self._ctrl_low = torch.as_tensor(self.map.ctrl_low, device=self.device)
        self._ctrl_high = torch.as_tensor(self.map.ctrl_high, device=self.device)
        self.joint_low = torch.as_tensor(self.map.joint_low, device=self.device)
        self.joint_high = torch.as_tensor(self.map.joint_high, device=self.device)
        # Track every non-world body for SONIC-style relative body rewards.
        self.body_names = [
            self.mujoco.mj_id2name(self.mjm, self.mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(1, self.mjm.nbody)
        ]
        self._all_body_ids = torch.arange(1, self.mjm.nbody, device=self.device)
        self.keypoint_indices = torch.as_tensor(
            [self.body_names.index(n) for n in sim_cfg.keypoint_body_names],
            device=self.device,
        )

        self._graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                mjw.step(self.m, self.d)
            self._graph = capture.graph

    @property
    def control_dt(self) -> float:
        return self.sim_cfg.physics_dt * self.sim_cfg.decimation

    @property
    def joint_names(self) -> list[str]:
        return self.map.joint_names

    def reset(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_linvel: torch.Tensor | None = None,
        root_angvel: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
    ) -> EnvObservation:
        if root_pos.shape[0] != self.num_envs:
            raise ValueError("reset batch must equal num_envs")
        rp = self.map.root_qpos_adr
        rv = self.map.root_dof_adr
        self._qpos[:, rp : rp + 3] = root_pos
        self._qpos[:, rp + 3 : rp + 7] = root_quat
        self._qpos[:, self._joint_qpos_adr] = joint_pos
        self._qvel.zero_()
        if root_linvel is not None:
            self._qvel[:, rv : rv + 3] = root_linvel
        if root_angvel is not None:
            self._qvel[:, rv + 3 : rv + 6] = root_angvel
        if joint_vel is not None:
            self._qvel[:, self._joint_dof_adr] = joint_vel
        self._last_action.zero_()
        self._history.zero_()
        self.mjw.forward(self.m, self.d)
        obs = self.observe()
        # Fill history with the initial state rather than zeros.
        self._history[:] = obs.proprio_frame[:, None, :]
        return self.observe()


    def reset_idx(
        self,
        indices: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_linvel: torch.Tensor | None = None,
        root_angvel: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
    ) -> EnvObservation:
        indices = indices.to(self.device, dtype=torch.long)
        rp = self.map.root_qpos_adr
        rv = self.map.root_dof_adr
        self._qpos[indices, rp : rp + 3] = root_pos
        self._qpos[indices, rp + 3 : rp + 7] = root_quat
        self._qpos[indices[:, None], self._joint_qpos_adr[None, :]] = joint_pos
        self._qvel[indices] = 0
        if root_linvel is not None:
            self._qvel[indices, rv : rv + 3] = root_linvel
        if root_angvel is not None:
            self._qvel[indices, rv + 3 : rv + 6] = root_angvel
        if joint_vel is not None:
            self._qvel[indices[:, None], self._joint_dof_adr[None, :]] = joint_vel
        self._last_action[indices] = 0
        self._history[indices] = 0
        self.mjw.forward(self.m, self.d)
        obs = self.observe()
        self._history[indices] = obs.proprio_frame[indices, None, :]
        return self.observe()

    def _step_once(self) -> None:
        if self._graph is None:
            self.mjw.step(self.m, self.d)
        else:
            self.wp.capture_launch(self._graph)

    def step(self, normalized_action: torch.Tensor) -> EnvObservation:
        # Preserve the user's existing action semantics: normalized position offsets.
        normalized_action = normalized_action.clamp(-1.0, 1.0)
        target = self._default_joint_pos + self.sim_cfg.action_scale * normalized_action
        target = torch.maximum(torch.minimum(target, self._ctrl_high), self._ctrl_low)
        self._ctrl[:, self._actuator_ids] = target
        for _ in range(self.sim_cfg.decimation):
            self._step_once()
        self._last_action.copy_(normalized_action)
        obs = self.observe()
        self._history = torch.roll(self._history, shifts=-1, dims=1)
        self._history[:, -1] = obs.proprio_frame
        return obs

    def observe(self) -> EnvObservation:
        rp = self.map.root_qpos_adr
        rv = self.map.root_dof_adr
        q = self._qpos[:, self._joint_qpos_adr]
        qd = self._qvel[:, self._joint_dof_adr]
        root_pos = self._qpos[:, rp : rp + 3]
        root_quat = self._qpos[:, rp + 3 : rp + 7]
        root_linvel = self._qvel[:, rv : rv + 3]
        root_angvel = self._qvel[:, rv + 3 : rv + 6]
        gravity_world = torch.zeros_like(root_angvel)
        gravity_world[:, 2] = -1.0
        projected_gravity = quat_rotate_inverse(root_quat, gravity_world)
        prop = torch.cat([q, qd, root_angvel, projected_gravity, self._last_action], dim=-1)

        body_pos = self._xpos[:, self._all_body_ids]
        body_quat = self._xquat[:, self._all_body_ids]
        # MuJoCo cvel ordering is angular then linear spatial velocity.
        cvel = self._cvel[:, self._all_body_ids]
        body_angvel = cvel[..., :3]
        body_linvel = cvel[..., 3:]
        return EnvObservation(
            proprio_frame=prop,
            joint_pos=q,
            joint_vel=qd,
            root_pos=root_pos,
            root_quat=root_quat,
            root_linvel=root_linvel,
            root_angvel=root_angvel,
            body_pos=body_pos,
            body_quat=body_quat,
            body_angvel=body_angvel,
            body_linvel=body_linvel,
        )

    def proprio_history(self) -> torch.Tensor:
        return self._history.flatten(1)

    def sync_world0_to_cpu(self):
        """Return an mjData copy for optional single-world rendering/debugging."""
        if self.num_envs != 1:
            raise RuntimeError("CPU rendering sync is intentionally limited to num_envs=1")
        mjd = self.mujoco.MjData(self.mjm)
        self.mjw.get_data_into(mjd, self.mjm, self.d)
        return mjd
