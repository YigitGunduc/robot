from types import SimpleNamespace

import torch

from mini_groot_sonic.config import SimConfig
from mini_groot_sonic.sim.mjwarp_env import MJWarpG1VecEnv


def test_proprio_matches_sonic_local_history_order_and_relative_joint_positions():
    env = object.__new__(MJWarpG1VecEnv)
    env.map = SimpleNamespace(root_qpos_adr=0, root_dof_adr=0)
    env.sim_cfg = SimConfig(
        enable_randomization=False,
        enable_observation_noise=False,
    )
    env._joint_qpos_adr = torch.tensor([7, 8])
    env._joint_dof_adr = torch.tensor([6, 7])
    env._default_joint_pos = torch.tensor([-0.2, 0.3])
    env._qpos = torch.tensor(
        [[0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0, -0.1, 0.5]]
    )
    env._qvel = torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.4, -0.5]])
    env._last_action = torch.tensor([[2.0, -3.0]])
    env._all_body_ids = torch.tensor([1])
    env._xpos = torch.zeros(1, 2, 3)
    env._xquat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    env._cvel = torch.zeros(1, 2, 6)

    obs = env.observe()
    expected = torch.tensor(
        [[
            0.0,
            0.0,
            -1.0,
            1.0,
            2.0,
            3.0,
            0.1,
            0.2,
            0.4,
            -0.5,
            2.0,
            -3.0,
        ]]
    )
    torch.testing.assert_close(obs.proprio_frame, expected)
