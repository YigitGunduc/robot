import torch

from mini_groot_sonic.config import FlowConfig, GoalConfig, SonicTinyConfig
from mini_groot_sonic.models.flow_policy import TinyFlowMotionPolicy
from mini_groot_sonic.models.sonic_tiny import TinySonicPolicy


def test_tiny_sonic_shapes():
    cfg = SonicTinyConfig(
        encoder_hidden=(64, 64),
        controller_hidden=(128, 64),
        recon_hidden=(64,),
    )
    policy = TinySonicPolicy(cfg, GoalConfig(hidden=(32,)))
    b = 4
    prop = torch.randn(b, cfg.proprio_dim)
    ref = torch.randn(b, cfg.future_frames, cfg.reference_frame_dim)
    goals = torch.randn(b, 6, 7)
    masks = torch.zeros(b, 6)
    masks[:, 3] = 1
    out = policy(prop, ref, goals, masks)
    assert out.action_mean.shape == (b, 29)
    assert out.token.shape == (b, 64)
    assert out.reconstruction.shape == (b, cfg.reference_dim)


def test_flow_loss_and_sample_shapes():
    cfg = FlowConfig(
        action_horizon=8,
        action_dim=64,
        state_dim=68,
        text_dim=32,
        vision_dim=24,
        goal_dim=48,
        model_dim=64,
        num_layers=2,
        num_heads=4,
        inference_steps=2,
    )
    model = TinyFlowMotionPolicy(cfg)
    b = 3
    actions = torch.randn(b, 8, 64)
    state = torch.randn(b, 68)
    text = torch.randn(b, 32)
    vision = torch.randn(b, 24)
    goal = torch.randn(b, 48)
    valid = torch.ones(b, 8)
    out = model.flow_matching_loss(actions, state, text, vision, goal, valid)
    assert out.loss.ndim == 0
    out.loss.backward()
    sample = model.sample(state, text, vision, goal)
    assert sample.shape == (b, 8, 64)


def test_flow_time_distribution_matches_n1d7_noise_biased_schedule():
    cfg = FlowConfig(
        action_horizon=2,
        state_dim=68,
        text_dim=8,
        vision_dim=8,
        goal_dim=0,
        model_dim=16,
        num_layers=1,
        num_heads=2,
    )
    model = TinyFlowMotionPolicy(cfg)
    b = 512
    out = model.flow_matching_loss(
        torch.randn(b, 2, 64),
        torch.randn(b, 68),
        torch.randn(b, 8),
    )
    assert 0.3 < float(out.t.mean()) < 0.5


def test_squashed_policy_actions_and_log_prob_are_finite():
    cfg = SonicTinyConfig(encoder_hidden=(16,), controller_hidden=(16,), recon_hidden=(16,))
    policy = TinySonicPolicy(cfg)
    mean = torch.full((32, cfg.dof), 5.0)
    dist = policy.distribution(mean)
    action = dist.rsample()
    assert torch.all(action > -1) and torch.all(action < 1)
    assert torch.isfinite(dist.log_prob(action)).all()
