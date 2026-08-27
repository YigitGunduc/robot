import torch

from mini_groot_sonic.config import SimConfig
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv


def test_joint_range_action_mapping_round_trip():
    env = object.__new__(MJWarpG1VecEnv)
    env.sim_cfg = SimConfig(
        action_scale=None, joint_limit_margin=0.0, sonic_g1_control=False
    )
    env._action_scale = None
    env._default_joint_pos = torch.tensor([-0.2, 0.3])
    env.joint_low = torch.tensor([-1.0, -0.5])
    env.joint_high = torch.tensor([0.7, 1.5])
    action = torch.tensor([[-1.0, -0.5], [0.25, 1.0]])
    target = env.action_to_target(action)
    reconstructed = env.target_to_action(target)
    torch.testing.assert_close(action, reconstructed)


def test_per_joint_residual_action_mapping_round_trip():
    env = object.__new__(MJWarpG1VecEnv)
    env.sim_cfg = SimConfig(action_scale=None)
    env._default_joint_pos = torch.tensor([-0.2, 0.3])
    env._action_scale = torch.tensor([0.1, 0.5])
    env.joint_low = torch.tensor([-1.0, -0.5])
    env.joint_high = torch.tensor([0.7, 1.5])
    action = torch.tensor([[-1.0, -0.5], [0.25, 1.0]])
    target = env.action_to_target(action)
    reconstructed = env.target_to_action(target)
    torch.testing.assert_close(action, reconstructed)
