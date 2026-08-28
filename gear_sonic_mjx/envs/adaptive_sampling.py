from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AdaptiveSamplerConfig:
    bin_size: int = 50
    init_num_failures: float = 1.0
    uniform_sampling_rate: float = 0.1
    pre_failure_sample_window: int = 200
    max_failure_over_mean: float = 200.0


class AdaptiveMotionSampler:
    """Failure-biased motion-time sampler following SONIC's released knobs.

    The public NVIDIA code tracks failure statistics over fixed-size motion bins and mixes
    adaptive sampling with a uniform component. This implementation keeps the same semantics
    while being simulator-agnostic.
    """

    def __init__(
        self,
        motion_lengths: torch.Tensor,
        cfg: AdaptiveSamplerConfig,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.lengths = motion_lengths.to(self.device, torch.long)
        self.cfg = cfg
        self.num_motions = int(self.lengths.numel())
        bins_per_motion = (
            torch.ceil(self.lengths.float() / cfg.bin_size).to(torch.long).clamp_min(1)
        )
        self.bins_per_motion = bins_per_motion
        self.offsets = torch.zeros(
            self.num_motions + 1, dtype=torch.long, device=self.device
        )
        self.offsets[1:] = torch.cumsum(bins_per_motion, dim=0)
        self.num_bins = int(self.offsets[-1].item())
        self.attempts = torch.zeros(
            self.num_bins, dtype=torch.float32, device=self.device
        )
        self.failures = torch.full(
            (self.num_bins,),
            float(cfg.init_num_failures),
            dtype=torch.float32,
            device=self.device,
        )

    def bin_index(self, motion_id: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        local = torch.div(frame.clamp_min(0), self.cfg.bin_size, rounding_mode="floor")
        local = torch.minimum(local, self.bins_per_motion[motion_id] - 1)
        return self.offsets[motion_id] + local

    @torch.no_grad()
    def record(
        self, motion_id: torch.Tensor, frame: torch.Tensor, failed: torch.Tensor
    ) -> None:
        idx = self.bin_index(motion_id.long(), frame.long())
        ones = torch.ones_like(idx, dtype=torch.float32)
        self.attempts.scatter_add_(0, idx, ones)
        self.failures.scatter_add_(0, idx, failed.float())

    def failure_weights(self) -> torch.Tensor:
        # Prior failures keep unseen bins nonzero. Attempts are clamped so new bins receive pressure.
        rates = self.failures / self.attempts.clamp_min(1.0)
        mean = rates.mean().clamp_min(1e-6)
        rates = rates.clamp_max(mean * self.cfg.max_failure_over_mean)
        adaptive = rates / rates.sum().clamp_min(1e-8)
        uniform = torch.full_like(adaptive, 1.0 / self.num_bins)
        u = float(self.cfg.uniform_sampling_rate)
        return u * uniform + (1.0 - u) * adaptive

    @torch.no_grad()
    def sample(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self.failure_weights()
        bins = torch.multinomial(
            probs, batch_size, replacement=True, generator=generator
        )
        # Map global bin -> motion id.
        motion_id = torch.searchsorted(self.offsets[1:], bins, right=True)
        local_bin = bins - self.offsets[motion_id]
        low = local_bin * self.cfg.bin_size
        high = torch.minimum(low + self.cfg.bin_size, self.lengths[motion_id])
        span = (high - low).clamp_min(1)
        r = torch.rand(batch_size, device=self.device, generator=generator)
        frame = low + torch.floor(r * span.float()).long()
        return motion_id, frame

    @torch.no_grad()
    def record_failure_with_window(
        self, motion_id: torch.Tensor, frame: torch.Tensor
    ) -> None:
        # NVIDIA exposes pre_failure_sample_window=200. Mark neighboring earlier bins as difficult too.
        win = int(self.cfg.pre_failure_sample_window)
        if win <= 0:
            self.record(motion_id, frame, torch.ones_like(frame, dtype=torch.bool))
            return
        offsets = torch.arange(0, win + 1, self.cfg.bin_size, device=self.device)
        mids = motion_id[:, None].expand(-1, offsets.numel()).reshape(-1)
        frames = (frame[:, None] - offsets[None]).clamp_min(0).reshape(-1)
        self.record(mids, frames, torch.ones_like(frames, dtype=torch.bool))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "attempts": self.attempts.detach().cpu(),
            "failures": self.failures.detach().cpu(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("attempts", "failures"):
            value = state[name]
            target = getattr(self, name)
            if value.shape != target.shape:
                raise ValueError(
                    f"adaptive sampler {name} shape mismatch: checkpoint "
                    f"{tuple(value.shape)}, runtime {tuple(target.shape)}"
                )
            target.copy_(value.to(target))
