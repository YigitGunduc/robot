import torch

from mini_groot_sonic.models.fsq import FiniteScalarQuantizer


def test_fsq_shape_range_and_gradient():
    q = FiniteScalarQuantizer(dim=64, levels=32)
    x = torch.randn(8, 64, requires_grad=True)
    y, idx = q(x)
    assert y.shape == x.shape
    assert idx.shape == x.shape
    assert y.min() >= -1.0001 and y.max() <= 1.0001
    assert idx.min() >= 0 and idx.max() <= 31
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
