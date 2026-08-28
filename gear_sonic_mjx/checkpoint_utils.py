from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def load_matching_tensors(
    module: nn.Module, checkpoint: dict[str, torch.Tensor], prefix: str = ""
) -> dict[str, list[str]]:
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


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "checkpoint CUDA RNG device count does not match the current runtime"
            )
        torch.cuda.set_rng_state_all(cuda_states)
