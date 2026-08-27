import torch

from mini_groot_sonic.data.reference import make_reference_features
from mini_groot_sonic.sim.math_utils import quat_mul, quat_rotate


def test_reference_is_invariant_to_global_xy_translation_and_yaw():
    b, f, dof = 2, 4, 29
    q = torch.randn(b, f, dof)
    qd = torch.randn_like(q)
    root_pos = torch.randn(b, f, 3)
    root_pos[..., 2] = 0.8
    root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(b, f, 4).clone()
    linvel = torch.randn(b, f, 3)
    angvel = torch.randn(b, f, 3)
    original = make_reference_features(q, qd, root_pos, root_quat, linvel, angvel)

    yaw = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5]).expand(b, f, 4)
    offset = torch.tensor([10.0, -4.0, 0.0])
    transformed_pos = quat_rotate(yaw, root_pos) + offset
    transformed_quat = quat_mul(yaw, root_quat)
    transformed = make_reference_features(
        q,
        qd,
        transformed_pos,
        transformed_quat,
        quat_rotate(yaw, linvel),
        angvel,
    )
    torch.testing.assert_close(original, transformed, atol=1e-5, rtol=1e-5)


def test_reference_orientation_is_relative_to_current_robot():
    b, f, dof = 1, 2, 29
    q = torch.zeros(b, f, dof)
    qd = torch.zeros_like(q)
    root_pos = torch.zeros(b, f, 3)
    root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(b, f, 4).clone()
    zeros = torch.zeros(b, f, 3)
    identity_robot = root_quat[:, 0]
    yaw_robot = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
    identity_features = make_reference_features(
        q, qd, root_pos, root_quat, zeros, zeros, identity_robot
    )
    yaw_features = make_reference_features(
        q, qd, root_pos, root_quat, zeros, zeros, yaw_robot
    )
    assert identity_features.shape[-1] == 2 * dof + 6
    assert not torch.allclose(identity_features[..., -6:], yaw_features[..., -6:])
