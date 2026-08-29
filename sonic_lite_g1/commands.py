from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg


class PackedFutureMotionCommand(MotionCommand):
    """MotionCommand supporting many clips packed into one mjlab-compatible NPZ.

    The normal mjlab MotionCommand tracks one continuous sequence. Our packed
    file concatenates many easy BONES clips and stores clip boundaries. This
    subclass prevents references/future windows from crossing those boundaries.

    The command sent to the actor is five future frames, each containing:
      joint_pos[29], joint_vel[29], anchor_lin_vel[3], anchor_ang_vel[3]
    = 64 values/frame. With five frames the encoder sees 320 values and emits a
    64-D quantized motor token.
    """

    cfg: "PackedFutureMotionCommandCfg"

    def __init__(self, cfg: "PackedFutureMotionCommandCfg", env) -> None:
        super().__init__(cfg, env)
        with np.load(cfg.motion_file, allow_pickle=False) as data:
            if "clip_starts" not in data or "clip_lengths" not in data:
                starts = np.array([0], dtype=np.int64)
                lengths = np.array([self.motion.time_step_total], dtype=np.int64)
            else:
                starts = np.asarray(data["clip_starts"], dtype=np.int64)
                lengths = np.asarray(data["clip_lengths"], dtype=np.int64)

        if starts.ndim != 1 or lengths.ndim != 1 or len(starts) != len(lengths):
            raise ValueError("clip_starts and clip_lengths must be matching 1-D arrays")
        if len(starts) == 0 or np.any(lengths <= 0):
            raise ValueError("packed motion must contain at least one non-empty clip")
        if int(starts[-1] + lengths[-1]) > self.motion.time_step_total:
            raise ValueError("clip metadata extends past concatenated motion arrays")

        self.clip_starts = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        self.clip_lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        self.clip_ends = self.clip_starts + self.clip_lengths  # exclusive
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._future_offsets = torch.as_tensor(
            cfg.future_frame_offsets, dtype=torch.long, device=self.device
        )
        self._sync_clip_ids()

    def _sync_clip_ids(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        # right=True: a frame exactly equal to an exclusive end belongs to next clip.
        clip_ids = torch.bucketize(self.time_steps[ids], self.clip_ends, right=True)
        self.clip_ids[ids] = torch.clamp(clip_ids, max=len(self.clip_ends) - 1)

    def _future_indices(self) -> torch.Tensor:
        idx = self.time_steps[:, None] + self._future_offsets[None, :]
        end = self.clip_ends[self.clip_ids][:, None] - 1
        return torch.minimum(idx, end)

    @property
    def command(self) -> torch.Tensor:
        idx = self._future_indices()
        q = self.motion.joint_pos[idx]
        qd = self.motion.joint_vel[idx]
        lin = self.motion.body_lin_vel_w[idx, self.motion_anchor_body_index]
        ang = self.motion.body_ang_vel_w[idx, self.motion_anchor_body_index]
        ref = torch.cat([q, qd, lin, ang], dim=-1)
        return ref.flatten(start_dim=1)

    def _uniform_sampling(self, env_ids: torch.Tensor):
        super()._uniform_sampling(env_ids)
        self._sync_clip_ids(env_ids)

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        super()._adaptive_sampling(env_ids)
        self._sync_clip_ids(env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        super()._resample_command(env_ids)
        self._sync_clip_ids(env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.time_steps += 1
        else:
            self.time_steps[env_ids] += 1

        # End each environment at its own clip boundary instead of walking into
        # the next concatenated clip.
        ends = self.clip_ends[self.clip_ids]
        wrap_ids = torch.where(self.time_steps >= ends)[0]
        if wrap_ids.numel() > 0:
            self._resample_command(wrap_ids)

        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
            self.update_relative_body_poses()

        # Same EMA update used by mjlab's adaptive hard-frame sampler.
        if env_ids is None and self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class PackedFutureMotionCommandCfg(MotionCommandCfg):
    # At 50 Hz: 0, 0.1, 0.2, 0.3, 0.4 seconds.
    future_frame_offsets: tuple[int, ...] = (0, 5, 10, 15, 20)

    def build(self, env) -> PackedFutureMotionCommand:
        return PackedFutureMotionCommand(self, env)
