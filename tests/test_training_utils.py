from __future__ import annotations

import random

import numpy as np
import torch

from mini_groot_sonic.training.utils import restore_rng_state


class _SerializedDeviceState:
    def __init__(self, cpu_state: torch.Tensor):
        self.cpu_state = cpu_state
        self.cpu_calls = 0

    def cpu(self) -> torch.Tensor:
        self.cpu_calls += 1
        return self.cpu_state


def test_restore_rng_state_moves_generator_states_to_cpu(monkeypatch) -> None:
    torch_state = _SerializedDeviceState(torch.get_rng_state())
    cuda_state = _SerializedDeviceState(torch.get_rng_state())
    restored: dict[str, object] = {}

    monkeypatch.setattr(torch, "set_rng_state", lambda value: restored.setdefault("torch", value))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda value: restored.setdefault("cuda", value),
    )

    restore_rng_state(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch_state,
            "cuda": [cuda_state],
        }
    )

    assert torch_state.cpu_calls == 1
    assert cuda_state.cpu_calls == 1
    assert restored["torch"].device.type == "cpu"
    assert all(value.device.type == "cpu" for value in restored["cuda"])
