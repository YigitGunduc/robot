from __future__ import annotations

import torch

from mini_groot_sonic.sim.math_utils import (
    quat_conjugate,
    quat_mul,
    quat_to_rotation_6d,
)


def make_reference_features(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_linvel: torch.Tensor,
    root_angvel: torch.Tensor,
    robot_root_quat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the released SONIC G1 encoder reference frames.

    Inputs have shape [B,F,*]; output is [B,F,2*dof+6]. Root translation and
    velocities are deliberately excluded; the robot encoder receives future
    q/qdot plus reference orientation relative to the current robot root.
    """
    del root_pos, root_linvel, root_angvel
    if robot_root_quat is None:
        robot_root_quat = root_quat[:, 0]
    robot_quat = robot_root_quat[:, None].expand_as(root_quat)
    relative_quat = quat_mul(quat_conjugate(robot_quat), root_quat)
    root_rot6d = quat_to_rotation_6d(relative_quat)
    return torch.cat([joint_pos, joint_vel, root_rot6d], dim=-1)


def flatten_reference_features(
    future_reference: torch.Tensor,
    dof: int,
) -> torch.Tensor:
    """Flatten in released SONIC input-key order: all q, then qdot, then root R6D."""

    if future_reference.ndim == 2:
        return future_reference
    if future_reference.ndim != 3 or future_reference.shape[-1] != 2 * dof + 6:
        raise ValueError("future_reference must be [B,F,2*dof+6] or already flattened")
    return torch.cat(
        [
            future_reference[..., :dof].flatten(1),
            future_reference[..., dof : 2 * dof].flatten(1),
            future_reference[..., 2 * dof :].flatten(1),
        ],
        dim=-1,
    )
