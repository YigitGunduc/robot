from __future__ import annotations

import torch
from torch import nn


def load_matching_tensors(module: nn.Module, checkpoint: dict[str, torch.Tensor], prefix: str = "") -> dict[str, list[str]]:
    """Conservative migration helper for an existing G1 checkpoint.

    Only tensors with an exact name *and* exact shape are copied. This is appropriate for preserving
    compatible output/hidden layers; it intentionally refuses to stretch an old locomotion input
    layer into SONIC's 994-D observation because that would silently change learned semantics.
    """
    current = module.state_dict()
    loaded, skipped = [], []
    for name, value in checkpoint.items():
        key = name.removeprefix(prefix) if prefix and name.startswith(prefix) else name
        if key in current and current[key].shape == value.shape:
            current[key] = value
            loaded.append(key)
        else:
            skipped.append(name)
    module.load_state_dict(current)
    return {"loaded": loaded, "skipped": skipped}
