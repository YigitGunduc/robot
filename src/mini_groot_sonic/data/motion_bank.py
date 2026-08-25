from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mini_groot_sonic.config import SonicTinyConfig


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

    def __init__(self, paths: list[str | Path], sonic_cfg: SonicTinyConfig, device: str):
        if not paths:
            raise ValueError("MotionBank requires at least one preprocessed clip")
        self.cfg = sonic_cfg
        self.device = torch.device(device)
        clips = [np.load(p, allow_pickle=True) for p in paths]
        self.lengths = torch.tensor([len(c["joint_pos"]) for c in clips], device=self.device, dtype=torch.long)
        self.captions = [str(c["caption"].item()) for c in clips]
        self.motion_names = [str(c["motion_id"].item()) for c in clips]
        self.body_names = [str(x) for x in clips[0]["body_names"].tolist()]
        max_t = int(max(self.lengths).item())
        n = len(clips)
        b = len(self.body_names)

        def alloc(shape):
            return torch.zeros(shape, device=self.device, dtype=torch.float32)

        self.joint_pos = alloc((n, max_t, sonic_cfg.dof))
        self.joint_vel = alloc((n, max_t, sonic_cfg.dof))
        self.root_pos = alloc((n, max_t, 3))
        self.root_quat = alloc((n, max_t, 4))
        self.body_pos = alloc((n, max_t, b, 3))
        self.body_quat = alloc((n, max_t, b, 4))
        self.body_linvel = alloc((n, max_t, b, 3))
        self.body_angvel = alloc((n, max_t, b, 3))
        for i, c in enumerate(clips):
            t = len(c["joint_pos"])
            for name in (
                "joint_pos", "joint_vel", "root_pos", "root_quat",
                "body_pos", "body_quat", "body_linvel", "body_angvel",
            ):
                getattr(self, name)[i, :t] = torch.from_numpy(np.asarray(c[name], dtype=np.float32)).to(self.device)
                if t < max_t:
                    getattr(self, name)[i, t:] = getattr(self, name)[i, t - 1]

    def sample_start(self, batch: int) -> MotionBankBatch:
        motion_ids = torch.randint(0, len(self.lengths), (batch,), device=self.device)
        max_offset = (self.cfg.future_frames - 1) * self.cfg.future_stride + 2
        max_start = (self.lengths[motion_ids] - max_offset).clamp_min(1)
        frame_ids = (torch.rand(batch, device=self.device) * max_start.float()).long()
        return MotionBankBatch(motion_ids, frame_ids)

    def future_reference(self, motion_ids: torch.Tensor, frame_ids: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.cfg.future_frames, device=self.device) * self.cfg.future_stride
        ids = frame_ids[:, None] + offsets[None, :]
        q = self.joint_pos[motion_ids[:, None], ids]
        qd = self.joint_vel[motion_ids[:, None], ids]
        return torch.cat([q, qd], dim=-1)

    def current_reference(self, motion_ids: torch.Tensor, frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        ix = (motion_ids, frame_ids)
        return {
            "joint_pos": self.joint_pos[ix],
            "joint_vel": self.joint_vel[ix],
            "root_pos": self.root_pos[ix],
            "root_quat": self.root_quat[ix],
            "body_pos": self.body_pos[ix],
            "body_quat": self.body_quat[ix],
            "body_linvel": self.body_linvel[ix],
            "body_angvel": self.body_angvel[ix],
        }
