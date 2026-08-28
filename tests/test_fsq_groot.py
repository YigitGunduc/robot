import sys
import types

import torch

from gear_sonic_mjx.trl.modules.fsq import FSQ
from groot_lite.model import FlowActionTransformer


def test_fsq_shape_gradient():
    x = torch.randn(5, 12, requires_grad=True)
    q = FSQ(12, 32)(x)
    assert q.shape == x.shape
    q.sum().backward()
    assert x.grad is not None


def test_high_dimensional_fsq_does_not_enumerate_implicit_codebook(monkeypatch):
    class EnumeratingFSQMustNotBeConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("attempted to enumerate an astronomical FSQ codebook")

    fake_upstream = types.ModuleType("vector_quantize_pytorch")
    fake_upstream.FSQ = EnumeratingFSQMustNotBeConstructed
    monkeypatch.setitem(sys.modules, "vector_quantize_pytorch", fake_upstream)

    quantizer = FSQ(dim=64, levels=32, num_tokens=2, token_dim=32)
    x = torch.randn(3, 64, requires_grad=True)
    quantized = quantizer(x)
    assert quantizer.upstream is None
    assert quantized.shape == x.shape
    quantized.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_flow_model_mask_and_sampling():
    model = FlowActionTransformer(
        action_dim=8, horizon=4, hidden_dim=32, layers=2, heads=4, condition_dim=32
    )
    cond = torch.randn(3, 32)
    clean = torch.randn(3, 4, 8)
    mask = torch.ones_like(clean, dtype=torch.bool)
    mask[:, :, 6:] = False
    loss = model.flow_matching_loss(clean, cond, mask)
    assert torch.isfinite(loss)
    loss.backward()
    sample = model.sample(cond, steps=4, action_mask=mask)
    assert sample.shape == clean.shape
    assert torch.all(sample[:, :, 6:] == 0)
