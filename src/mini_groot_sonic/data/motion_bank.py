from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mini_groot_sonic.config import SonicTinyConfig
from mini_groot_sonic.data.reference import make_reference_features


@dataclass
class MotionBankBatch:
    motion_ids: torch.Tensor
    frame_ids: torch.Tensor


class MotionBank:
    """In-memory motion bank for the first tractable training stage.

    Load a few hundred/thousand preprocessed BONES clips first. Scale to sharded
    streaming only after the controller is proven; this keeps the implementation
    compact and debuggable.
    """

    def __init__(
        self,
        paths: list[str | Path],
        sonic_cfg: SonicTinyConfig,
        device: str,
        max_memory_gb: float = 8.0,
        failure_sampling_alpha: float = 0.5,
        failure_sampling_cap: float = 4.0,
    ):
        if not paths:
            raise ValueError("MotionBank requires at least one preprocessed clip")
        self.cfg = sonic_cfg
        self.device = torch.device(device)
        self.failure_sampling_alpha = min(1.0, max(0.0, failure_sampling_alpha))
        self.failure_sampling_cap = max(1.0, failure_sampling_cap)
        clips = [np.load(p, allow_pickle=True) for p in paths]
        self.lengths = torch.tensor([len(c["joint_pos"]) for c in clips], device=self.device, dtype=torch.long)
        self.captions = [str(c["caption"].item()) for c in clips]
        self.motion_names = [str(c["motion_id"].item()) for c in clips]
        self.body_names = [str(x) for x in clips[0]["body_names"].tolist()]
        minimum_padded_length = (sonic_cfg.future_frames - 1) * sonic_cfg.future_stride + 2
        max_t = max(int(max(self.lengths).item()), minimum_padded_length)
        n = len(clips)
        b = len(self.body_names)
        estimated_bytes = n * max_t * (sonic_cfg.dof * 2 + 13 + 13 * b) * 4
        if estimated_bytes > max_memory_gb * 1024**3:
            for clip in clips:
                clip.close()
            raise MemoryError(
                f"MotionBank would allocate about {estimated_bytes / 1024**3:.1f} GiB. "
                "Use a smaller subset now or implement sharded streaming before scaling."
            )

        def alloc(shape):
            return torch.zeros(shape, device=self.device, dtype=torch.float32)

        self.joint_pos = alloc((n, max_t, sonic_cfg.dof))
        self.joint_vel = alloc((n, max_t, sonic_cfg.dof))
        self.root_pos = alloc((n, max_t, 3))
        self.root_quat = alloc((n, max_t, 4))
        self.root_linvel = alloc((n, max_t, 3))
        self.root_angvel = alloc((n, max_t, 3))
        self.body_pos = alloc((n, max_t, b, 3))
        self.body_quat = alloc((n, max_t, b, 4))
        self.body_linvel = alloc((n, max_t, b, 3))
        self.body_angvel = alloc((n, max_t, b, 3))
        for i, c in enumerate(clips):
            t = len(c["joint_pos"])
            root_body_index = self.body_names.index("pelvis") if "pelvis" in self.body_names else 0
            for name in (
                "joint_pos", "joint_vel", "root_pos", "root_quat",
                "body_pos", "body_quat", "body_linvel", "body_angvel",
            ):
                getattr(self, name)[i, :t] = torch.from_numpy(np.asarray(c[name], dtype=np.float32)).to(self.device)
                if t < max_t:
                    getattr(self, name)[i, t:] = getattr(self, name)[i, t - 1]
            for name, fallback in (
                ("root_linvel", "body_linvel"),
                ("root_angvel", "body_angvel"),
            ):
                values = c[name] if name in c.files else c[fallback][:, root_body_index]
                getattr(self, name)[i, :t] = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(self.device)
                if t < max_t:
                    getattr(self, name)[i, t:] = getattr(self, name)[i, t - 1]
        valid_starts = (self.lengths - ((self.cfg.future_frames - 1) * self.cfg.future_stride + 2)).clamp_min(1)
        self.base_sampling_weights = valid_starts.float()
        self.failure_ema = torch.zeros(n, device=self.device)
        for clip in clips:
            clip.close()

    def sample_start(self, batch: int) -> MotionBankBatch:
        weights = self.sampling_weights()
        motion_ids = torch.multinomial(weights, batch, replacement=True)
        max_offset = (self.cfg.future_frames - 1) * self.cfg.future_stride + 2
        max_start = (self.lengths[motion_ids] - max_offset).clamp_min(1)
        frame_ids = (torch.rand(batch, device=self.device) * max_start.float()).long()
        return MotionBankBatch(motion_ids, frame_ids)

    @torch.no_grad()
    def sampling_weights(self) -> torch.Tensor:
        """Blend uniform-frame coverage with capped failure-targeted sampling."""

        uniform = self.base_sampling_weights / self.base_sampling_weights.sum()
        if self.failure_sampling_alpha <= 0.0:
            return uniform
        mean_failure = self.failure_ema.mean()
        capped_failure = self.failure_ema.clamp_max(
            self.failure_sampling_cap * mean_failure
        )
        targeted = self.base_sampling_weights * capped_failure
        targeted_sum = targeted.sum()
        targeted = targeted / targeted_sum.clamp_min(torch.finfo(targeted.dtype).eps)
        has_target = (targeted_sum > 0).to(targeted.dtype)
        targeted = has_target * targeted + (1.0 - has_target) * uniform
        return (
            (1.0 - self.failure_sampling_alpha) * uniform
            + self.failure_sampling_alpha * targeted
        )

    @torch.no_grad()
    def update_failures(
        self,
        motion_ids: torch.Tensor,
        failed: torch.Tensor,
    ) -> None:
        unique_ids, inverse = motion_ids.unique(return_inverse=True)
        totals = torch.zeros(len(unique_ids), device=self.device)
        counts = torch.zeros_like(totals)
        totals.scatter_add_(0, inverse, failed.float())
        counts.scatter_add_(0, inverse, torch.ones_like(failed, dtype=torch.float32))
        rates = totals / counts.clamp_min(1.0)
        self.failure_ema[unique_ids] = (
            0.95 * self.failure_ema[unique_ids] + 0.05 * rates
        )
        self.failure_ema.clamp_(0.0, 1.0)

    def future_reference(self, motion_ids: torch.Tensor, frame_ids: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.cfg.future_frames, device=self.device) * self.cfg.future_stride
        ids = frame_ids[:, None] + offsets[None, :]
        q = self.joint_pos[motion_ids[:, None], ids]
        qd = self.joint_vel[motion_ids[:, None], ids]
        root_pos = self.root_pos[motion_ids[:, None], ids]
        root_quat = self.root_quat[motion_ids[:, None], ids]
        root_linvel = self.root_linvel[motion_ids[:, None], ids]
        root_angvel = self.root_angvel[motion_ids[:, None], ids]
        return make_reference_features(q, qd, root_pos, root_quat, root_linvel, root_angvel)

    @torch.no_grad()
    def reference_stats(self, max_samples: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
        batch = min(max_samples, max(256, len(self.lengths) * 32))
        starts = self.sample_start(batch)
        refs = self.future_reference(starts.motion_ids, starts.frame_ids).flatten(1)
        return refs.mean(0), refs.std(0).clamp_min(1e-4)

    def current_reference(self, motion_ids: torch.Tensor, frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        ix = (motion_ids, frame_ids)
        return {
            "joint_pos": self.joint_pos[ix],
            "joint_vel": self.joint_vel[ix],
            "root_pos": self.root_pos[ix],
            "root_quat": self.root_quat[ix],
            "root_linvel": self.root_linvel[ix],
            "root_angvel": self.root_angvel[ix],
            "body_pos": self.body_pos[ix],
            "body_quat": self.body_quat[ix],
            "body_linvel": self.body_linvel[ix],
            "body_angvel": self.body_angvel[ix],
        }
