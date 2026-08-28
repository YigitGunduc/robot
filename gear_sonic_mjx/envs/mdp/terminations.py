from __future__ import annotations

from dataclasses import dataclass

import torch

from gear_sonic_mjx.config import TerminationConfig
from gear_sonic_mjx.math_utils import quat_angle_error


@dataclass
class TerminationMetrics:
    root_pos_error: torch.Tensor
    root_ori_error: torch.Tensor
    ee_pos_error: torch.Tensor | None = None
    foot_xyz_error: torch.Tensor | None = None
    motion_finished: torch.Tensor | None = None


def termination_mask(m: TerminationMetrics, cfg: TerminationConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reasons = {
        "anchor_pos": m.root_pos_error > cfg.anchor_pos,
        "anchor_ori": m.root_ori_error > cfg.anchor_ori,
    }
    if m.ee_pos_error is not None:
        reasons["ee_body_pos"] = m.ee_pos_error > cfg.ee_body_pos
    if m.foot_xyz_error is not None:
        reasons["foot_pos_xyz"] = m.foot_xyz_error > cfg.foot_pos_xyz
    if m.motion_finished is not None:
        reasons["motion_time_out"] = m.motion_finished.bool()
    done = torch.zeros_like(m.root_pos_error, dtype=torch.bool)
    for v in reasons.values():
        done |= v
    return done, reasons


class AdaptiveTerminationCurriculum:
    """Optional loose-to-strict thresholds.

    NVIDIA's release uses adaptive termination terms but does not expose one universal public
    schedule in the top-level docs. This helper lets you warm up from `start_scale * strict` down
    to the strict released thresholds without pretending the schedule is an exact NVIDIA constant.
    """

    def __init__(self, strict: TerminationConfig, start_scale: float = 2.0, end_step: int = 5000):
        self.strict = strict
        self.start_scale = float(start_scale)
        self.end_step = int(end_step)

    def at(self, update: int) -> TerminationConfig:
        alpha = min(max(update / max(self.end_step, 1), 0.0), 1.0)
        scale = self.start_scale + alpha * (1.0 - self.start_scale)
        return TerminationConfig(
            anchor_pos=self.strict.anchor_pos * scale,
            anchor_ori=self.strict.anchor_ori * scale,
            ee_body_pos=self.strict.ee_body_pos * scale,
            foot_pos_xyz=self.strict.foot_pos_xyz * scale,
        )
