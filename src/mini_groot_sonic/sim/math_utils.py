from __future__ import annotations

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[..., 1:]


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conjugate(q), v)


def quat_distance_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # shortest-angle distance for wxyz quaternions
    rel = quat_mul(quat_conjugate(a), b)
    w = rel[..., 0].abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(w)


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert normalized wxyz quaternions to rotation matrices."""
    w, x, y, z = normalize_quat(q).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quat_to_rotation_6d(q: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation using the first two columns."""
    matrix = quat_to_matrix(q)
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)
