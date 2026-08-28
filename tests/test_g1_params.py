import torch

from gear_sonic_mjx.g1_parameters import (
    IL_TO_MJ,
    MJ_TO_IL,
    to_mujoco_order_il_to_mj,
    to_policy_order_mj_to_il,
)


def test_joint_order_roundtrip():
    x = torch.arange(29).float()[None]
    il = to_policy_order_mj_to_il(x)
    back = to_mujoco_order_il_to_mj(il)
    assert torch.equal(x, back)
    assert torch.equal(MJ_TO_IL[IL_TO_MJ], torch.arange(29))
