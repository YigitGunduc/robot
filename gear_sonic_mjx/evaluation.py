from __future__ import annotations

import torch


def mpjpe(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Global mean per-joint/body position error in meters; inputs [B,N,3]."""
    return torch.linalg.vector_norm(current - reference, dim=-1).mean(-1)


def local_mpjpe(current: torch.Tensor, reference: torch.Tensor, current_root: torch.Tensor, reference_root: torch.Tensor) -> torch.Tensor:
    c = current - current_root[:, None]
    r = reference - reference_root[:, None]
    return torch.linalg.vector_norm(c - r, dim=-1).mean(-1)


def tracking_success(terminated_as_failure: torch.Tensor) -> torch.Tensor:
    return 1.0 - terminated_as_failure.float().mean()


class MetricAccumulator:
    def __init__(self):
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, name: str, value: torch.Tensor) -> None:
        self.sum[name] = self.sum.get(name, 0.0) + float(value.detach().sum())
        self.count[name] = self.count.get(name, 0) + value.numel()

    def means(self) -> dict[str, float]:
        return {k: self.sum[k] / max(self.count[k], 1) for k in self.sum}
