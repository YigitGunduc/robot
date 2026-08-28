import torch

from gear_sonic_mjx.envs.adaptive_sampling import AdaptiveMotionSampler, AdaptiveSamplerConfig


def test_adaptive_sampler_biases_failures():
    s = AdaptiveMotionSampler(torch.tensor([100, 100]), AdaptiveSamplerConfig(bin_size=50, uniform_sampling_rate=0.1))
    # Make global bin 0 substantially harder.
    for _ in range(100):
        s.record(torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))
    p = s.failure_weights()
    assert p[0] > p[1]
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-6)
