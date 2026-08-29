import torch
from sonic_lite_g1.quantizer import ScalarQuantizer


def test_quantizer_levels_and_grad():
    q = ScalarQuantizer(32)
    x = torch.randn(128, requires_grad=True)
    y = q(x)
    assert torch.all(y <= 1.0) and torch.all(y >= -1.0)
    assert torch.unique(y.detach()).numel() <= 32
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
