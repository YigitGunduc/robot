from __future__ import annotations

from dataclasses import dataclass

import torch

from gear_sonic_mjx.config import SonicConfig
from gear_sonic_mjx.envs.adaptive_sampling import AdaptiveMotionSampler, AdaptiveSamplerConfig
from gear_sonic_mjx.envs.motion_library import BonesMotionLibrary
from gear_sonic_mjx.envs.mdp.actions import joint_position_target, pd_torque
from gear_sonic_mjx.envs.mdp.observations import ProprioHistory, PrivilegedHistory, g1_tokenizer_observation
from gear_sonic_mjx.envs.mdp.events import sample_root_velocity_push
from gear_sonic_mjx.envs.mdp.rewards import SonicReward, TrackingReference, TrackingState
from gear_sonic_mjx.envs.mdp.terminations import TerminationMetrics, termination_mask
from gear_sonic_mjx.g1_parameters import (
    DEFAULT_ANGLES_MJ,
    EFFORT,
    KP_MJ,
    KD_MJ,
    SONIC_TRACKED_BODY_NAMES,
    SONIC_REWARD_POINT_BODY_NAMES,
    SONIC_REWARD_POINT_OFFSETS,
    SONIC_FOOT_BODY_NAMES,
    SONIC_ANTI_SHAKE_BODY_NAMES,
)
from gear_sonic_mjx.math_utils import (
    projected_gravity,
    quat_angle_error,
    rotate_inverse_wxyz,
    quat_apply_wxyz,
    quat_mul_wxyz,
    quat_conjugate_wxyz,
    euler_xyz_to_quat_wxyz,
    heading_quat_wxyz,
    relative_rotation_6d,
)
from gear_sonic_mjx.sim.mjwarp_backend import MjWarpBatchSim


@dataclass
class StepOutput:
    encoder_obs: torch.Tensor
    proprio_obs: torch.Tensor
    critic_obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    info: dict[str, torch.Tensor]


class G1SonicTrackingTask:
    """Batched SONIC-style G1 motion tracking task on MuJoCo-Warp.

    This class deliberately mirrors NVIDIA's *interfaces* rather than depending on Isaac Lab:
      - 29-D joint-position action through the released G1 PD scaling;
      - 10-frame actor history with observation corruption;
      - 10 future G1 reference frames at 0.1 s spacing;
      - clean privileged asymmetric critic;
      - BONES-SEED adaptive failure sampling;
      - upper-body recombination and freeze-frame augmentation;
      - heading/translation aligned body references for body tracking terms.

    Exact body tracking requires FK-augmented BONES caches. Run ``scripts/augment_bones_fk.py``
    against the same MJCF before training.
    """

    def __init__(
        self,
        sim: MjWarpBatchSim,
        motions: BonesMotionLibrary,
        cfg: SonicConfig,
        reward_body_names: list[str] | None = None,
        reward_point_names: list[str] | None = None,
        foot_names: list[str] | None = None,
        anti_shake_names: list[str] | None = None,
        require_fk_cache: bool = True,
    ):
        self.sim, self.motions, self.cfg = sim, motions, cfg
        self.device, self.n = sim.device, sim.nworld
        ac = cfg.motion.adaptive_sampling
        self.sampler = AdaptiveMotionSampler(
            motions.lengths,
            AdaptiveSamplerConfig(
                ac.bin_size,
                ac.init_num_failures,
                ac.uniform_sampling_rate,
                ac.pre_failure_sample_window,
                ac.max_failure_over_mean,
            ),
            self.device,
        )
        self.reward_fn = SonicReward(cfg.reward)
        self.history = ProprioHistory(self.n, cfg.model.dof, cfg.motion.actor_prop_history_length, self.device)
        self.critic_history = PrivilegedHistory(self.n, cfg.model.dof, cfg.motion.actor_prop_history_length, self.device)
        self.motion_id = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.frame = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.prev_action = torch.zeros(self.n, cfg.model.dof, device=self.device)
        self.freeze_enabled = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.freeze_frame = torch.full((self.n,), 2**30, dtype=torch.long, device=self.device)
        self.joint_lower, self.joint_upper = sim.joint_limits()
        self.default_q = DEFAULT_ANGLES_MJ.to(self.device)

        probe = motions._load(0)
        self.reward_body_names = list(reward_body_names or SONIC_TRACKED_BODY_NAMES)
        self.reward_point_names = list(reward_point_names or SONIC_REWARD_POINT_BODY_NAMES)
        self.foot_names = list(foot_names or SONIC_FOOT_BODY_NAMES)
        self.anti_shake_names = list(anti_shake_names or SONIC_ANTI_SHAKE_BODY_NAMES)
        self.reward_point_offsets = SONIC_REWARD_POINT_OFFSETS.to(self.device)

        if require_fk_cache and (probe.body_names is None or probe.body_pos is None or probe.body_quat_wxyz is None):
            raise ValueError("BONES cache lacks body FK. Run scripts/augment_bones_fk.py before SONIC training.")
        self.require_fk_cache = require_fk_cache

        if probe.body_names:
            ref_lookup = {name: i for i, name in enumerate(probe.body_names)}
            missing = [name for name in self.reward_body_names if name not in ref_lookup]
            if missing and require_fk_cache:
                raise KeyError(f"FK cache is missing SONIC tracked bodies: {missing}")
            self.ref_body_idx = torch.tensor(
                [ref_lookup[name] for name in self.reward_body_names if name in ref_lookup],
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.ref_body_idx = None

        # Simulation bodies are queried directly in the exact canonical SONIC ordering.
        self.body_ids = sim.body_ids(self.reward_body_names) if self.reward_body_names else None
        body_lookup = {name: i for i, name in enumerate(self.reward_body_names)}
        self.reward_point_idx = torch.tensor([body_lookup[n] for n in self.reward_point_names], device=self.device)
        self.foot_idx = torch.tensor([body_lookup[n] for n in self.foot_names], device=self.device)
        self.anti_shake_idx = torch.tensor([body_lookup[n] for n in self.anti_shake_names], device=self.device)

        self.episode_step = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.next_push_step = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.prev_foot_linvel = torch.zeros(self.n, len(self.foot_names), 3, device=self.device)
        self._resample_push_schedule(torch.arange(self.n, device=self.device))

        # Released critic composition approximation:
        # future q/qd (580) + anchor pos/ori (9) + 14 body pos/orientation (126)
        # + 10 clean history frames (930) = 1645 dimensions.
        f = cfg.motion.num_future_frames
        self.critic_dim = f * cfg.model.dof * 2 + 3 + 6 + len(self.reward_body_names) * (3 + 6) + self.critic_history.length * self.critic_history.frame_dim

    def _uniform(self, shape: tuple[int, ...], limits, *, dtype=torch.float32) -> torch.Tensor:
        lo, hi = float(limits[0]), float(limits[1])
        return torch.empty(shape, device=self.device, dtype=dtype).uniform_(lo, hi)

    def _resample_push_schedule(self, env_ids: torch.Tensor) -> None:
        interval = self.cfg.domain_randomization.get("push_interval_s", [4.0, 6.0])
        lo = max(1, int(float(interval[0]) / self.cfg.sim.policy_dt))
        hi = max(lo + 1, int(float(interval[1]) / self.cfg.sim.policy_dt) + 1)
        self.next_push_step[env_ids] = self.episode_step[env_ids] + torch.randint(lo, hi, (env_ids.numel(),), device=self.device)

    @torch.no_grad()
    def _maybe_push(self) -> None:
        ids = (self.episode_step >= self.next_push_step).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            # Defaults match the released command velocity perturbation ranges.
            self.sim.add_root_velocity(ids, sample_root_velocity_push(ids.numel(), self.device))
            self._resample_push_schedule(ids)

    def _effective_frame(self) -> torch.Tensor:
        return torch.where(self.freeze_enabled, torch.minimum(self.frame, self.freeze_frame), self.frame)

    def _reference(self):
        ref = self.motions.batch_current(self.motion_id, self._effective_frame(), self.device)
        frozen_now = self.freeze_enabled & (self.frame >= self.freeze_frame)
        if frozen_now.any():
            ref["joint_vel"] = ref["joint_vel"].clone(); ref["joint_vel"][frozen_now] = 0.0
            for key in ("body_linvel", "body_angvel"):
                if ref.get(key) is not None:
                    ref[key] = ref[key].clone(); ref[key][frozen_now] = 0.0
        return ref

    def _select_ref_bodies(self, ref: dict) -> dict[str, torch.Tensor | None]:
        out = {}
        for key in ("body_pos", "body_quat_wxyz", "body_linvel", "body_angvel"):
            value = ref.get(key)
            if value is None or self.ref_body_idx is None:
                out[key] = None
            else:
                out[key] = value.index_select(1, self.ref_body_idx)
        return out

    def _aligned_reference_bodies(
        self,
        robot_root_pos: torch.Tensor,
        robot_root_q: torch.Tensor,
        ref_root_pos: torch.Tensor,
        ref_root_q: torch.Tensor,
        ref_body_pos: torch.Tensor | None,
        ref_body_q: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Reproduce SONIC command-manager heading alignment for body pose tracking."""
        if ref_body_pos is None or ref_body_q is None:
            return None, None
        # NVIDIA anchors reference XY to robot XY but retains the reference root height.
        delta_pos = robot_root_pos.clone()
        delta_pos[:, 2] = ref_root_pos[:, 2]
        full_delta = quat_mul_wxyz(robot_root_q, quat_conjugate_wxyz(ref_root_q))
        delta_q = heading_quat_wxyz(full_delta)
        dq = delta_q[:, None, :].expand(-1, ref_body_pos.shape[1], -1)
        centered = ref_body_pos - ref_root_pos[:, None, :]
        aligned_pos = delta_pos[:, None, :] + quat_apply_wxyz(dq, centered)
        aligned_q = quat_mul_wxyz(dq, ref_body_q)
        return aligned_pos, aligned_q

    def _reward_points(
        self,
        body_pos: torch.Tensor | None,
        body_q: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if body_pos is None or body_q is None:
            return None
        p = body_pos.index_select(1, self.reward_point_idx)
        q = body_q.index_select(1, self.reward_point_idx)
        offsets = self.reward_point_offsets.to(p)[None].expand(p.shape[0], -1, -1)
        return p + quat_apply_wxyz(q, offsets)

    def _clean_current(self):
        root_pos, root_q = self.sim.root_pose()
        root_v, root_w = self.sim.root_velocity()
        q, qd = self.sim.joint_state()
        base_lin_local = rotate_inverse_wxyz(root_q, root_v)
        base_ang_local = rotate_inverse_wxyz(root_q, root_w)
        q_rel = q - self.default_q
        grav = projected_gravity(root_q)
        return root_pos, root_q, base_lin_local, base_ang_local, q, q_rel, qd, grav

    def _actor_frame(self):
        _, _, _, base_ang, _, q_rel, qd, grav = self._clean_current()
        if self.cfg.observation_noise.enabled:
            n = self.cfg.observation_noise
            base_ang = base_ang + torch.empty_like(base_ang).uniform_(-n.base_ang_vel, n.base_ang_vel)
            q_rel = q_rel + torch.empty_like(q_rel).uniform_(-n.joint_pos, n.joint_pos)
            qd = qd + torch.empty_like(qd).uniform_(-n.joint_vel, n.joint_vel)
            grav = grav + torch.empty_like(grav).uniform_(-n.gravity, n.gravity)
        return base_ang, q_rel, qd, grav

    def _build_critic(
        self,
        future: dict[str, torch.Tensor],
        root_pos: torch.Tensor,
        root_q: torch.Tensor,
        body: dict[str, torch.Tensor | None],
        advance_history: bool,
    ) -> torch.Tensor:
        _, _, base_lin, base_ang, _, q_rel, qd, _ = self._clean_current()
        if advance_history:
            hist = self.critic_history.push(base_lin, base_ang, q_rel, qd, self.prev_action)
        else:
            hist = self.critic_history.flat()

        command = torch.cat([future["joint_pos"], future["joint_vel"]], dim=-1).reshape(self.n, -1)
        ref_anchor_pos = future["root_pos"][:, 0]
        ref_anchor_q = future["root_quat_wxyz"][:, 0]
        anchor_pos_b = rotate_inverse_wxyz(root_q, ref_anchor_pos - root_pos)
        anchor_ori_b = relative_rotation_6d(root_q, ref_anchor_q)

        if body["body_pos"] is None or body["body_quat_wxyz"] is None:
            body_features = torch.zeros(self.n, len(self.reward_body_names) * 9, device=self.device)
        else:
            bpos = body["body_pos"]
            bq = body["body_quat_wxyz"]
            rq = root_q[:, None, :].expand(-1, bpos.shape[1], -1)
            bpos_local = rotate_inverse_wxyz(rq, bpos - root_pos[:, None, :])
            bori_local = relative_rotation_6d(rq, bq)
            body_features = torch.cat([bpos_local.reshape(self.n, -1), bori_local.reshape(self.n, -1)], dim=-1)
        critic = torch.cat([command, anchor_pos_b, anchor_ori_b, body_features, hist], dim=-1)
        if critic.shape[-1] != self.critic_dim:
            raise RuntimeError(f"critic observation contract mismatch: expected {self.critic_dim}, got {critic.shape[-1]}")
        return critic

    def _obs(self, *, advance_history: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_pos, root_q, _, _, _, _, _, _ = self._clean_current()
        base_ang, q_rel, qd, grav = self._actor_frame()
        prop = self.history.push(base_ang, q_rel, qd, self.prev_action, grav) if advance_history else self.history.flat()

        future = self.motions.batch_future(
            self.motion_id,
            self.frame,
            self.cfg.motion.num_future_frames,
            self.cfg.motion.dt_future_ref_frames,
            self.device,
            frame_cap=self.freeze_frame if self.cfg.motion.freeze_frame_aug else None,
        )
        if self.cfg.motion.freeze_frame_aug:
            stride = max(1, int(round(self.cfg.motion.dt_future_ref_frames * self.cfg.motion.target_fps)))
            absolute = self.frame[:, None] + torch.arange(self.cfg.motion.num_future_frames, device=self.device)[None] * stride
            frozen_future = self.freeze_enabled[:, None] & (absolute >= self.freeze_frame[:, None])
            if frozen_future.any():
                future["joint_vel"] = future["joint_vel"].clone()
                future["joint_vel"][frozen_future] = 0.0
        enc = g1_tokenizer_observation(
            future["joint_pos"],
            future["joint_vel"],
            root_q,
            future["root_quat_wxyz"],
            orientation_noise=self.cfg.observation_noise.tokenizer_orientation if self.cfg.observation_noise.enabled else 0.0,
        )
        body = self.sim.body_state(self.body_ids) if self.body_ids is not None else {
            k: None for k in ["body_pos", "body_quat_wxyz", "body_linvel", "body_angvel"]
        }
        critic = self._build_critic(future, root_pos, root_q, body, advance_history)
        return enc, prop, critic

    def _seed_histories(self, env_ids: torch.Tensor) -> None:
        _, _, base_lin, base_ang, _, q_rel, qd, grav = self._clean_current()
        # Actor seed is clean current state; subsequent real frames receive NVIDIA-style corruption.
        self.history.seed(env_ids, base_ang, q_rel, qd, self.prev_action, grav)
        self.critic_history.seed(env_ids, base_lin, base_ang, q_rel, qd, self.prev_action)

    @torch.no_grad()
    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        mids, frames = self.sampler.sample(env_ids.numel())
        self.motion_id[env_ids] = mids
        self.frame[env_ids] = frames
        # NVIDIA freezes ~10% of loaded sequences from a random frame onward. In this streaming
        # port the same augmentation is sampled per episode, with all future/current references
        # clamped consistently so reward/body targets and tokenizer commands cannot disagree.
        if self.cfg.motion.freeze_frame_aug:
            enabled = torch.rand(env_ids.numel(), device=self.device) < self.cfg.motion.freeze_frame_aug_prob
            lengths = self.motions.lengths.to(self.device)[mids]
            cap = torch.floor(torch.rand(env_ids.numel(), device=self.device) * lengths.float()).long()
            self.freeze_enabled[env_ids] = enabled
            self.freeze_frame[env_ids] = torch.where(enabled, cap, torch.full_like(cap, 2**30))
        else:
            self.freeze_enabled[env_ids] = False
            self.freeze_frame[env_ids] = 2**30
        effective = torch.where(self.freeze_enabled[env_ids], torch.minimum(frames, self.freeze_frame[env_ids]), frames)
        ref = self.motions.batch_current(mids, effective, self.device)
        frozen_at_reset = self.freeze_enabled[env_ids] & (frames >= self.freeze_frame[env_ids])
        if frozen_at_reset.any():
            ref["joint_vel"] = ref["joint_vel"].clone(); ref["joint_vel"][frozen_at_reset] = 0.0

        root_pos = ref["root_pos"].clone()
        root_q = ref["root_quat_wxyz"].clone()
        q = ref["joint_pos"].clone()
        qd = ref["joint_vel"].clone()
        dr = self.cfg.domain_randomization

        pose = dr.get("initial_pose", {})
        if pose:
            root_pos[:, 0] += self._uniform((env_ids.numel(),), pose.get("x", [0, 0]))
            root_pos[:, 1] += self._uniform((env_ids.numel(),), pose.get("y", [0, 0]))
            root_pos[:, 2] += self._uniform((env_ids.numel(),), pose.get("z", [0, 0]))
            euler = torch.stack(
                [
                    self._uniform((env_ids.numel(),), pose.get("roll", [0, 0])),
                    self._uniform((env_ids.numel(),), pose.get("pitch", [0, 0])),
                    self._uniform((env_ids.numel(),), pose.get("yaw", [0, 0])),
                ],
                dim=-1,
            )
            root_q = quat_mul_wxyz(root_q, euler_xyz_to_quat_wxyz(euler))

        qrange = dr.get("initial_joint_position", [0.0, 0.0])
        q += self._uniform(q.shape, qrange)
        q = torch.maximum(torch.minimum(q, self.joint_upper), self.joint_lower)
        qdrange = dr.get("initial_joint_velocity", [0.0, 0.0])
        qd += self._uniform(qd.shape, qdrange)

        vel = dr.get("initial_velocity", {})
        root_velocity6 = None
        if vel:
            root_velocity6 = torch.stack(
                [
                    self._uniform((env_ids.numel(),), vel.get("x", [0, 0])),
                    self._uniform((env_ids.numel(),), vel.get("y", [0, 0])),
                    self._uniform((env_ids.numel(),), vel.get("z", [0, 0])),
                    self._uniform((env_ids.numel(),), vel.get("roll", [0, 0])),
                    self._uniform((env_ids.numel(),), vel.get("pitch", [0, 0])),
                    self._uniform((env_ids.numel(),), vel.get("yaw", [0, 0])),
                ],
                dim=-1,
            )

        self.sim.set_state(env_ids, root_pos, root_q, q, qd, root_velocity6=root_velocity6)
        self.prev_action[env_ids] = 0.0
        self.episode_step[env_ids] = 0
        self._resample_push_schedule(env_ids)
        self._seed_histories(env_ids)
        body = self.sim.body_state(self.body_ids)
        if body["body_linvel"] is not None:
            self.prev_foot_linvel[env_ids] = body["body_linvel"].index_select(1, self.foot_idx)[env_ids]
        else:
            self.prev_foot_linvel[env_ids] = 0.0

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.n, device=self.device)
        self._reset_envs(env_ids)
        # Histories were explicitly seeded; initial observation must not advance them.
        return self._obs(advance_history=False)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> StepOutput:
        if action.shape != (self.n, self.cfg.model.dof):
            raise ValueError(f"Expected action [{self.n},{self.cfg.model.dof}], got {tuple(action.shape)}")
        self._maybe_push()
        target_q = joint_position_target(action)
        for _ in range(self.cfg.sim.decimation):
            q, qd = self.sim.joint_state()
            tau = pd_torque(target_q, q, qd, KP_MJ.to(self.device), KD_MJ.to(self.device))
            tau = tau.clamp(-EFFORT.to(self.device), EFFORT.to(self.device))
            self.sim.write_torque(tau)
            self.sim.step(1)

        self.frame += 1
        self.episode_step += 1
        ref = self._reference()
        ref_b = self._select_ref_bodies(ref)
        root_pos, root_q = self.sim.root_pose()
        body = self.sim.body_state(self.body_ids) if self.body_ids is not None else {
            k: None for k in ["body_pos", "body_quat_wxyz", "body_linvel", "body_angvel"]
        }
        aligned_ref_pos, aligned_ref_q = self._aligned_reference_bodies(
            root_pos,
            root_q,
            ref["root_pos"],
            ref["root_quat_wxyz"],
            ref_b["body_pos"],
            ref_b["body_quat_wxyz"],
        )

        reward_points = self._reward_points(body["body_pos"], body["body_quat_wxyz"])
        # The local point reward uses the unaligned source trajectory and each trajectory's own root.
        ref_reward_points = self._reward_points(ref_b["body_pos"], ref_b["body_quat_wxyz"])

        q, _ = self.sim.joint_state()
        anti_shake = None if body["body_angvel"] is None else body["body_angvel"].index_select(1, self.anti_shake_idx)
        feet_acc = None
        if body["body_linvel"] is not None:
            foot_vel = body["body_linvel"].index_select(1, self.foot_idx)
            feet_acc = (foot_vel - self.prev_foot_linvel) / self.cfg.sim.policy_dt
            self.prev_foot_linvel.copy_(foot_vel)

        state = TrackingState(
            root_pos=root_pos,
            root_quat=root_q,
            body_pos=body["body_pos"],
            body_quat=body["body_quat_wxyz"],
            body_linvel=body["body_linvel"],
            body_angvel=body["body_angvel"],
            reward_points=reward_points,
            action=action,
            prev_action=self.prev_action,
            joint_pos=q,
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
            undesired_contact=None,  # Add a project-specific contact selector once MJCF collision groups are known.
            anti_shake_angvel=anti_shake,
            feet_acc=feet_acc,
        )
        reference = TrackingReference(
            root_pos=ref["root_pos"],
            root_quat=ref["root_quat_wxyz"],
            body_pos=aligned_ref_pos,
            body_quat=aligned_ref_q,
            body_linvel=ref_b["body_linvel"],
            body_angvel=ref_b["body_angvel"],
            reward_points=ref_reward_points,
        )
        reward, terms = self.reward_fn(state, reference)

        root_pos_error = torch.linalg.vector_norm(root_pos - ref["root_pos"], dim=-1)
        root_ori_error = quat_angle_error(root_q, ref["root_quat_wxyz"])
        ee_error = foot_error = None
        if body["body_pos"] is not None and aligned_ref_pos is not None:
            ee_error = torch.linalg.vector_norm(
                body["body_pos"].index_select(1, self.reward_point_idx)
                - aligned_ref_pos.index_select(1, self.reward_point_idx),
                dim=-1,
            ).amax(dim=-1)
            foot_error = torch.linalg.vector_norm(
                body["body_pos"].index_select(1, self.foot_idx)
                - aligned_ref_pos.index_select(1, self.foot_idx),
                dim=-1,
            ).amax(dim=-1)
        finished = self.frame >= self.motions.lengths.to(self.device)[self.motion_id]
        done, reasons = termination_mask(
            TerminationMetrics(root_pos_error, root_ori_error, ee_error, foot_error, finished), self.cfg.termination
        )

        failed = done & ~finished
        if failed.any():
            self.sampler.record_failure_with_window(self.motion_id[failed], self.frame[failed])
        alive_or_completed = ~failed
        if alive_or_completed.any():
            self.sampler.record(
                self.motion_id[alive_or_completed],
                self.frame[alive_or_completed],
                torch.zeros_like(self.frame[alive_or_completed], dtype=torch.bool),
            )

        self.prev_action.copy_(action)
        if done.any():
            self._reset_envs(done.nonzero(as_tuple=False).squeeze(-1))
        # Every physical policy step advances all surviving/newly-seeded histories exactly once.
        enc, prop, critic = self._obs(advance_history=True)
        info = {
            **terms,
            **{f"termination/{k}": v.float() for k, v in reasons.items()},
            "failed": failed.float(),
        }
        return StepOutput(enc, prop, critic, reward, done, info)
