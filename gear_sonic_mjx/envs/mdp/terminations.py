from __future__ import annotations

from dataclasses import dataclass

import torch

from gear_sonic_mjx.config import TerminationConfig


@dataclass
class TerminationMetrics:
    root_height_error: torch.Tensor
    root_ori_error: torch.Tensor
    ee_height_error: torch.Tensor | None = None
    foot_xyz_error: torch.Tensor | None = None
    reference_root_height: torch.Tensor | None = None
    motion_finished: torch.Tensor | None = None
    episode_time_out: torch.Tensor | None = None


def termination_mask(
    m: TerminationMetrics, cfg: TerminationConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reasons = {
        "anchor_pos": m.root_height_error > cfg.anchor_pos,
        # NVIDIA's exceeded_anchor_ori compares squared angular error to the configured threshold.
        "anchor_ori": m.root_ori_error.square() > cfg.anchor_ori,
    }
    if m.ee_height_error is not None:
        threshold = torch.full_like(m.ee_height_error, cfg.ee_body_pos)
        if m.reference_root_height is not None:
            low = m.reference_root_height < cfg.root_height_threshold
            threshold = torch.where(
                low, torch.full_like(threshold, cfg.down_threshold), threshold
            )
        reasons["ee_body_pos"] = m.ee_height_error > threshold
    if m.foot_xyz_error is not None:
        reasons["foot_pos_xyz"] = m.foot_xyz_error > cfg.foot_pos_xyz
    if m.motion_finished is not None:
        reasons["motion_time_out"] = m.motion_finished.bool()
    if m.episode_time_out is not None:
        reasons["episode_time_out"] = m.episode_time_out.bool()
    done = torch.zeros_like(m.root_height_error, dtype=torch.bool)
    for v in reasons.values():
        done |= v
    return done, reasons


class AdaptiveTerminationCurriculum:
    """Optional loose-to-strict thresholds.

    NVIDIA's release uses adaptive termination terms but does not expose one universal public
    schedule in the top-level docs. This helper lets you warm up from `start_scale * strict` down
    to the strict released thresholds without pretending the schedule is an exact NVIDIA constant.
    """

    def __init__(
        self, strict: TerminationConfig, start_scale: float = 2.0, end_step: int = 5000
    ):
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
