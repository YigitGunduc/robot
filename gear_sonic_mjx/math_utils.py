from __future__ import annotations

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_to_matrix_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(m: torch.Tensor) -> torch.Tensor:
    # Zhou et al. 6D representation, first two rotation-matrix columns.
    return m[..., :, :2].transpose(-1, -2).reshape(m.shape[:-2] + (6,))


def relative_rotation_6d(
    robot_q_wxyz: torch.Tensor, ref_q_wxyz: torch.Tensor
) -> torch.Tensor:
    rel = quat_mul_wxyz(quat_conjugate_wxyz(robot_q_wxyz), ref_q_wxyz)
    return matrix_to_rotation_6d(quat_to_matrix_wxyz(rel))


def quat_angle_error(a_wxyz: torch.Tensor, b_wxyz: torch.Tensor) -> torch.Tensor:
    # Shortest angular distance in radians.
    rel = quat_mul_wxyz(quat_conjugate_wxyz(a_wxyz), b_wxyz)
    w = quat_normalize(rel)[..., 0].abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(w)


def rotate_inverse_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r = quat_to_matrix_wxyz(q)
    return torch.matmul(r.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)


def projected_gravity(root_q_wxyz: torch.Tensor) -> torch.Tensor:
    g_world = torch.zeros(
        root_q_wxyz.shape[:-1] + (3,),
        device=root_q_wxyz.device,
        dtype=root_q_wxyz.dtype,
    )
    g_world[..., 2] = -1.0
    return rotate_inverse_wxyz(root_q_wxyz, g_world)


def quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.matmul(quat_to_matrix_wxyz(q), v.unsqueeze(-1)).squeeze(-1)


def euler_xyz_to_quat_wxyz(euler: torch.Tensor) -> torch.Tensor:
    """XYZ intrinsic Euler angles [rad] -> wxyz quaternion."""
    hx, hy, hz = (0.5 * euler).unbind(-1)
    cx, sx = torch.cos(hx), torch.sin(hx)
    cy, sy = torch.cos(hy), torch.sin(hy)
    cz, sz = torch.cos(hz), torch.sin(hz)
    return torch.stack(
        [
            cx * cy * cz - sx * sy * sz,
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
        ],
        dim=-1,
    )


def heading_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Return a pure-yaw quaternion carrying the heading of ``q`` (wxyz)."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    out = torch.zeros_like(q)
    out[..., 0] = torch.cos(half)
    out[..., 3] = torch.sin(half)
    return out
