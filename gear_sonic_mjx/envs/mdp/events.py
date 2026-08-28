from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PushRanges:
    lin_low: tuple[float, float, float] = (-0.5, -0.5, -0.2)
    lin_high: tuple[float, float, float] = (0.5, 0.5, 0.2)
    ang_low: tuple[float, float, float] = (-0.52, -0.52, -0.78)
    ang_high: tuple[float, float, float] = (0.52, 0.52, 0.78)


def sample_root_velocity_push(batch: int, device: torch.device | str, ranges: PushRanges = PushRanges()) -> torch.Tensor:
    lo = torch.tensor([*ranges.lin_low, *ranges.ang_low], device=device)
    hi = torch.tensor([*ranges.lin_high, *ranges.ang_high], device=device)
    return lo + torch.rand(batch, 6, device=device) * (hi - lo)


def sample_mass_scale(batch: int, num_bodies: int, device: torch.device | str, low: float = 0.8, high: float = 2.5) -> torch.Tensor:
    return low + torch.rand(batch, num_bodies, device=device) * (high - low)
