from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EndEffectorTarget:
    position: torch.Tensor  # [...,3]
    rotation_6d: torch.Tensor  # [...,6]
    gripper: torch.Tensor | None = None


class TaskSpaceSafetyGate:
    """Pre-IK validation for V2 manipulation targets.

    This does not invent an IK solver for an unknown user MJCF. Wire `solve_ik` to the IK routine in
    your existing `g1_arm_manipulation.py`; the gate keeps targets inside configured workspaces and
    rejects non-finite commands before they can conflict with SONIC body control.
    """

    def __init__(self, xyz_low: torch.Tensor, xyz_high: torch.Tensor):
        self.xyz_low, self.xyz_high = xyz_low, xyz_high

    def clamp(self, target: EndEffectorTarget) -> EndEffectorTarget:
        pos = torch.maximum(
            torch.minimum(target.position, self.xyz_high.to(target.position)),
            self.xyz_low.to(target.position),
        )
        rot = torch.nan_to_num(target.rotation_6d)
        grip = None if target.gripper is None else target.gripper.clamp(0.0, 1.0)
        return EndEffectorTarget(pos, rot, grip)

    def valid(self, target: EndEffectorTarget) -> torch.Tensor:
        finite = torch.isfinite(target.position).all(-1) & torch.isfinite(
            target.rotation_6d
        ).all(-1)
        inside = (
            (target.position >= self.xyz_low.to(target.position))
            & (target.position <= self.xyz_high.to(target.position))
        ).all(-1)
        return finite & inside
