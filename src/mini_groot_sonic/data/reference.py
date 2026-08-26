from __future__ import annotations

import torch

from mini_groot_sonic.sim.math_utils import (
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
    quat_to_rotation_6d,
)


def make_reference_features(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_linvel: torch.Tensor,
    root_angvel: torch.Tensor,
) -> torch.Tensor:
    """Build heading/translation-invariant SONIC reference frames.

    Inputs have shape [B,F,*]; output is [B,F,2*dof+16].
    """
    origin_quat = root_quat[:, :1].expand_as(root_quat)
    root_delta = quat_rotate_inverse(origin_quat, root_pos - root_pos[:, :1])
    relative_quat = quat_mul(quat_conjugate(origin_quat), root_quat)
    root_rot6d = quat_to_rotation_6d(relative_quat)
    local_linvel = quat_rotate_inverse(root_quat, root_linvel)
    # MuJoCo free-joint angular velocity already lives in the local body frame.
    local_angvel = root_angvel
    return torch.cat(
        [
            joint_pos,
            joint_vel,
            root_delta,
            root_rot6d,
            root_pos[..., 2:3],
            local_linvel,
            local_angvel,
        ],
        dim=-1,
    )
