from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mini_groot_sonic.config import SimConfig, SonicTinyConfig
from mini_groot_sonic.sim.g1_control import (
    calibrate_position_actuators,
    sonic_g1_control_profile,
)
from mini_groot_sonic.sim.g1_mapping import G1ModelMap
from mini_groot_sonic.sim.math_utils import (
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
    quat_to_rotation_6d,
)


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
        self._control_profile = None
        if sim_cfg.sonic_g1_control:
            self._control_profile = sonic_g1_control_profile(self.map.joint_names)
            if self.map.actuator_is_position.all():
                calibrate_position_actuators(self.mjm, self.map, self._control_profile)
                # Refresh actuator metadata after modifying the compiled model.
                self.map = G1ModelMap.from_mjmodel(
                    self.mjm,
                    sim_cfg.root_body_name,
                    tuple(sim_cfg.keypoint_body_names),
                )
        self._batched_dynamics = False
        if sim_cfg.enable_randomization:
            try:
                self.m = mjw.put_model(
                    self.mjm,
                    batch_sizes={
                        "geom_friction": num_envs,
                        "body_mass": num_envs,
                        "body_inertia": num_envs,
                        "body_ipos": num_envs,
                    },
                )
                self._batched_dynamics = True
            except TypeError:
                # Older MJWarp releases do not expose per-field model batching.
                self.m = mjw.put_model(self.mjm)
        else:
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
        default_joint_pos = (
            self._control_profile.default_joint_pos
            if self._control_profile is not None
            else self.map.default_joint_pos
        )
        default_joint_pos = np.clip(default_joint_pos, self.map.joint_low, self.map.joint_high)
        self._default_joint_pos = torch.as_tensor(default_joint_pos, device=self.device)
        self._ctrl_low = torch.as_tensor(self.map.ctrl_low, device=self.device)
        self._ctrl_high = torch.as_tensor(self.map.ctrl_high, device=self.device)
        self.joint_low = torch.as_tensor(self.map.joint_low, device=self.device)
        self.joint_high = torch.as_tensor(self.map.joint_high, device=self.device)
        joint_mid = 0.5 * (self.joint_low + self.joint_high)
        joint_half_range = 0.5 * (self.joint_high - self.joint_low)
        self.soft_joint_low = joint_mid - sim_cfg.soft_joint_limit_factor * joint_half_range
        self.soft_joint_high = joint_mid + sim_cfg.soft_joint_limit_factor * joint_half_range
        if sim_cfg.action_scale is not None:
            self._action_scale = torch.full(
                (sonic_cfg.dof,), float(sim_cfg.action_scale), device=self.device
            )
        elif self._control_profile is not None:
            self._action_scale = torch.as_tensor(
                self._control_profile.action_scale, device=self.device
            )
        else:
            self._action_scale = None
        position_mask = self.map.actuator_is_position
        if sim_cfg.actuator_mode == "auto":
            if position_mask.all():
                self.actuator_mode = "position"
            elif (~position_mask).all():
                self.actuator_mode = "pd_torque"
            else:
                raise ValueError("Mixed position/motor actuators require an explicit unified MJCF")
        elif sim_cfg.actuator_mode in {"position", "pd_torque"}:
            self.actuator_mode = sim_cfg.actuator_mode
        else:
            raise ValueError("actuator_mode must be 'auto', 'position', or 'pd_torque'")
        if self.actuator_mode == "position" and not position_mask.all():
            raise ValueError("actuator_mode='position' requires MuJoCo position actuators")
        if self.actuator_mode == "pd_torque":
            if not self.map.actuator_is_motor.all():
                raise ValueError("actuator_mode='pd_torque' requires MuJoCo motor actuators")
            if not np.isfinite(self.map.ctrl_low).all() or not np.isfinite(self.map.ctrl_high).all():
                raise ValueError("PD torque control requires finite actuator ctrlrange torque limits")
            if np.any(np.abs(self.map.actuator_gear) < 1e-8):
                raise ValueError("PD torque control requires non-zero actuator gear ratios")
        self._actuator_gear = torch.as_tensor(self.map.actuator_gear, device=self.device)

        base_kp = (
            torch.as_tensor(self._control_profile.stiffness, device=self.device)
            if self._control_profile is not None
            else torch.full((sonic_cfg.dof,), sim_cfg.joint_stiffness, device=self.device)
        )
        base_kd = (
            torch.as_tensor(self._control_profile.damping, device=self.device)
            if self._control_profile is not None
            else torch.full((sonic_cfg.dof,), sim_cfg.joint_damping, device=self.device)
        )
        self._base_kp = base_kp
        self._base_kd = base_kd
        self._kp = base_kp[None].repeat(num_envs, 1)
        self._kd = base_kd[None].repeat(num_envs, 1)
        self._motor_strength = torch.ones(num_envs, sonic_cfg.dof, device=self.device)
        self._delayed_action = torch.zeros_like(self._last_action)
        self._delay_mask = torch.zeros(num_envs, 1, dtype=torch.bool, device=self.device)
        self._actuator_force = wp.to_torch(self.d.actuator_force)
        self._geom_friction = wp.to_torch(self.m.geom_friction)
        self._body_mass = wp.to_torch(self.m.body_mass)
        self._body_inertia = wp.to_torch(self.m.body_inertia)
        self._body_ipos = wp.to_torch(self.m.body_ipos)
        self._base_geom_friction = self._geom_friction[0].clone()
        self._base_body_mass = self._body_mass[0].clone()
        self._base_body_inertia = self._body_inertia[0].clone()
        self._base_body_ipos = self._body_ipos[0].clone()

        # Contact arrays are flat across worlds in MJWarp. Keeping them as Torch
        # views lets rewards stay on the GPU without copying an mjData per world.
        self._contact_geom = wp.to_torch(self.d.contact.geom)
        self._contact_world = wp.to_torch(self.d.contact.worldid)
        self._contact_dist = wp.to_torch(self.d.contact.dist)
        self._geom_bodyid = torch.as_tensor(self.mjm.geom_bodyid, device=self.device, dtype=torch.long)
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
        allowed_body_ids = []
        for name in sim_cfg.allowed_contact_body_names:
            bid = self.mujoco.mj_name2id(self.mjm, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                allowed_body_ids.append(int(bid))
        self._allowed_contact_body_ids = torch.as_tensor(allowed_body_ids, device=self.device)

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

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self._default_joint_pos

    @property
    def action_scale(self) -> torch.Tensor | None:
        return self._action_scale

    @property
    def action_clip_value(self) -> float:
        return self.sim_cfg.action_clip_value

    @property
    def last_action(self) -> torch.Tensor:
        return self._last_action

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
        joint_pos = self._randomized_joint_reset(joint_pos)
        self._qpos[:, self._joint_qpos_adr] = joint_pos
        self._qvel.zero_()
        if root_linvel is not None:
            self._qvel[:, rv : rv + 3] = root_linvel
        if root_angvel is not None:
            self._qvel[:, rv + 3 : rv + 6] = root_angvel
        if joint_vel is not None:
            self._qvel[:, self._joint_dof_adr] = joint_vel
        if self.sim_cfg.enable_randomization:
            self._qvel += self.sim_cfg.reset_velocity_noise * torch.randn_like(self._qvel)
        self._randomize_actuators(torch.arange(self.num_envs, device=self.device))
        self._last_action.zero_()
        self._delayed_action.zero_()
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
        joint_pos = self._randomized_joint_reset(joint_pos)
        self._qpos[indices[:, None], self._joint_qpos_adr[None, :]] = joint_pos
        self._qvel[indices] = 0
        if root_linvel is not None:
            self._qvel[indices, rv : rv + 3] = root_linvel
        if root_angvel is not None:
            self._qvel[indices, rv + 3 : rv + 6] = root_angvel
        if joint_vel is not None:
            self._qvel[indices[:, None], self._joint_dof_adr[None, :]] = joint_vel
        if self.sim_cfg.enable_randomization:
            self._qvel[indices] += self.sim_cfg.reset_velocity_noise * torch.randn_like(self._qvel[indices])
        self._randomize_actuators(indices)
        self._last_action[indices] = 0
        self._delayed_action[indices] = 0
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

    def _randomized_joint_reset(self, joint_pos: torch.Tensor) -> torch.Tensor:
        if not self.sim_cfg.enable_randomization:
            return joint_pos
        noisy = joint_pos + self.sim_cfg.reset_joint_noise * torch.randn_like(joint_pos)
        return torch.maximum(torch.minimum(noisy, self.joint_high), self.joint_low)

    def _randomize_actuators(self, indices: torch.Tensor) -> None:
        if not self.sim_cfg.enable_randomization:
            self._kp[indices] = self._base_kp
            self._kd[indices] = self._base_kd
            self._motor_strength[indices] = 1.0
            self._delay_mask[indices] = False
            return

        def uniform(bounds: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
            low, high = bounds
            return low + (high - low) * torch.rand(shape, device=self.device)

        shape = (len(indices), self.sonic_cfg.dof)
        self._kp[indices] = self._base_kp[None] * uniform(self.sim_cfg.stiffness_range, shape)
        self._kd[indices] = self._base_kd[None] * uniform(self.sim_cfg.damping_range, shape)
        self._motor_strength[indices] = uniform(self.sim_cfg.motor_strength_range, shape)
        self._delay_mask[indices] = (
            torch.rand(len(indices), 1, device=self.device) < self.sim_cfg.action_delay_probability
        )
        if self._batched_dynamics:
            friction_scale = uniform(
                self.sim_cfg.friction_scale_range,
                (len(indices), 1, 1),
            )
            self._geom_friction[indices] = self._base_geom_friction[None] * friction_scale
            mass_scale = uniform(self.sim_cfg.mass_scale_range, (len(indices), self.mjm.nbody, 1))
            self._body_mass[indices] = self._base_body_mass[None] * mass_scale[..., 0]
            self._body_inertia[indices] = self._base_body_inertia[None] * mass_scale
            com_noise = self.sim_cfg.center_of_mass_noise * torch.randn(
                len(indices), self.mjm.nbody, 3, device=self.device
            )
            com_noise[:, 0] = 0
            self._body_ipos[indices] = self._base_body_ipos[None] + com_noise

    def action_to_target(self, action: torch.Tensor) -> torch.Tensor:
        if self._action_scale is not None:
            action = action.clamp(
                -self.sim_cfg.action_clip_value,
                self.sim_cfg.action_clip_value,
            )
            target = self._default_joint_pos + self._action_scale * action
        else:
            # The non-SONIC fallback maps a normalized action over each side of
            # the joint range and therefore remains a [-1, 1] interface.
            action = action.clamp(-1.0, 1.0)
            low = self.joint_low + self.sim_cfg.joint_limit_margin
            high = self.joint_high - self.sim_cfg.joint_limit_margin
            positive = action * (high - self._default_joint_pos)
            negative = action * (self._default_joint_pos - low)
            target = self._default_joint_pos + torch.where(action >= 0, positive, negative)
        return torch.maximum(torch.minimum(target, self.joint_high), self.joint_low)

    def target_to_action(self, target: torch.Tensor) -> torch.Tensor:
        if self._action_scale is not None:
            return ((target - self._default_joint_pos) / self._action_scale).clamp(
                -self.sim_cfg.action_clip_value,
                self.sim_cfg.action_clip_value,
            )
        low = self.joint_low + self.sim_cfg.joint_limit_margin
        high = self.joint_high - self.sim_cfg.joint_limit_margin
        delta = target - self._default_joint_pos
        scale = torch.where(delta >= 0, high - self._default_joint_pos, self._default_joint_pos - low)
        return (delta / scale.clamp_min(1e-6)).clamp(-1.0, 1.0)

    def _maybe_push(self) -> None:
        if not self.sim_cfg.enable_randomization or self.sim_cfg.push_probability_per_step <= 0:
            return
        pushed = torch.rand(self.num_envs, device=self.device) < self.sim_cfg.push_probability_per_step
        direction = torch.randn(self.num_envs, 2, device=self.device)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rv = self.map.root_dof_adr
        self._qvel[:, rv : rv + 2] += (
            pushed[:, None] * self.sim_cfg.push_velocity * direction
        )

    def step(self, action: torch.Tensor) -> EnvObservation:
        action = action.clamp(
            -self.sim_cfg.action_clip_value,
            self.sim_cfg.action_clip_value,
        )
        applied_action = torch.where(self._delay_mask, self._delayed_action, action)
        self._delayed_action.copy_(action)
        target = self.action_to_target(applied_action)
        self._maybe_push()
        for _ in range(self.sim_cfg.decimation):
            if self.actuator_mode == "position":
                ctrl = torch.maximum(torch.minimum(target, self._ctrl_high), self._ctrl_low)
            else:
                q = self._qpos[:, self._joint_qpos_adr]
                qd = self._qvel[:, self._joint_dof_adr]
                joint_torque = self._motor_strength * (self._kp * (target - q) - self._kd * qd)
                ctrl = joint_torque / self._actuator_gear
                ctrl = torch.maximum(torch.minimum(ctrl, self._ctrl_high), self._ctrl_low)
            self._ctrl[:, self._actuator_ids] = ctrl
            self._step_once()
        self._last_action.copy_(applied_action)
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
        prop_q, prop_qd = q - self._default_joint_pos, qd
        prop_angvel, prop_gravity = root_angvel, projected_gravity
        if self.sim_cfg.enable_observation_noise:
            def uniform_noise(value: torch.Tensor, magnitude: float) -> torch.Tensor:
                return value + magnitude * (2.0 * torch.rand_like(value) - 1.0)

            prop_q = uniform_noise(prop_q, self.sim_cfg.observation_joint_pos_noise)
            prop_qd = uniform_noise(prop_qd, self.sim_cfg.observation_joint_vel_noise)
            prop_angvel = uniform_noise(
                prop_angvel, self.sim_cfg.observation_angular_vel_noise
            )
            prop_gravity = uniform_noise(prop_gravity, self.sim_cfg.gravity_noise)
        prop = torch.cat(
            [prop_gravity, prop_angvel, prop_q, prop_qd, self._last_action],
            dim=-1,
        )

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

    def privileged_observation(
        self,
        obs: EnvObservation | None = None,
        reference_root_pos: torch.Tensor | None = None,
        reference_root_quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs = obs or self.observe()
        root_quat = obs.root_quat[:, None].expand_as(obs.body_quat)
        body_pos_local = quat_rotate_inverse(
            root_quat, obs.body_pos - obs.root_pos[:, None]
        )
        body_quat_local = self._relative_body_quat(obs.root_quat, obs.body_quat)
        body_ori_6d = quat_to_rotation_6d(body_quat_local)
        if reference_root_pos is None:
            reference_root_pos = obs.root_pos
        if reference_root_quat is None:
            reference_root_quat = obs.root_quat
        anchor_pos_local = quat_rotate_inverse(
            obs.root_quat, reference_root_pos - obs.root_pos
        )
        anchor_ori_local = quat_to_rotation_6d(
            quat_mul(quat_conjugate(obs.root_quat), reference_root_quat)
        )
        base_linvel_local = quat_rotate_inverse(obs.root_quat, obs.root_linvel)
        return torch.cat(
            [
                obs.joint_pos,
                obs.joint_vel,
                anchor_pos_local,
                anchor_ori_local,
                base_linvel_local,
                obs.root_angvel,
                body_pos_local.flatten(1),
                body_ori_6d.flatten(1),
            ],
            dim=-1,
        )

    @staticmethod
    def _relative_body_quat(root_quat: torch.Tensor, body_quat: torch.Tensor) -> torch.Tensor:
        root_inv = quat_conjugate(root_quat)[:, None].expand_as(body_quat)
        return quat_mul(root_inv, body_quat)

    @property
    def privileged_dim(self) -> int:
        return 2 * self.sonic_cfg.dof + 15 + 9 * len(self.body_names)

    def undesired_contact_count(self) -> torch.Tensor:
        geom = self._contact_geom.long()
        world = self._contact_world.long()
        valid = (world >= 0) & (world < self.num_envs) & (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
        valid &= self._contact_dist <= -self.sim_cfg.undesired_contact_penetration
        safe_geom = geom.clamp(0, max(len(self._geom_bodyid) - 1, 0))
        body0 = self._geom_bodyid[safe_geom[:, 0]]
        body1 = self._geom_bodyid[safe_geom[:, 1]]
        robot_body = torch.where(body0 > 0, body0, body1)
        ground_contact = (body0 == 0) ^ (body1 == 0)
        self_contact = (body0 > 0) & (body1 > 0)
        if self._allowed_contact_body_ids.numel():
            allowed = (robot_body[:, None] == self._allowed_contact_body_ids[None]).any(-1)
        else:
            allowed = torch.zeros_like(valid)
        undesired = valid & (self_contact | (ground_contact & ~allowed))
        # Count affected bodies rather than raw contact points; a multi-point foot
        # collision should not be penalized several times merely due to geometry.
        pairs = world[undesired] * self.mjm.nbody + robot_body[undesired]
        unique_pairs = torch.unique(pairs)
        unique_worlds = torch.div(unique_pairs, self.mjm.nbody, rounding_mode="floor")
        counts = torch.zeros(self.num_envs, device=self.device)
        counts.scatter_add_(0, unique_worlds, torch.ones_like(unique_worlds, dtype=counts.dtype))
        return counts

    def mean_abs_actuator_force(self) -> torch.Tensor:
        return self._actuator_force[:, self._actuator_ids].abs().mean(-1)

    def sync_world0_to_cpu(self):
        """Return an mjData copy for optional single-world rendering/debugging."""
        if self.num_envs != 1:
            raise RuntimeError("CPU rendering sync is intentionally limited to num_envs=1")
        mjd = self.mujoco.MjData(self.mjm)
        self.mjw.get_data_into(mjd, self.mjm, self.d)
        return mjd
