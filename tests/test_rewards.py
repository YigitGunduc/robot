from types import SimpleNamespace

import torch

from mini_groot_sonic.config import RewardConfig
from mini_groot_sonic.training.rewards import SonicStyleReward


def test_reward_is_high_for_matching_reference():
    body_names = ["pelvis", "head", "left_wrist", "right_wrist", "left_ankle", "right_ankle"]
    keypoints = ["head", "left_wrist", "right_wrist", "left_ankle", "right_ankle"]
    joint_names = [f"joint_{i}" for i in range(27)] + ["left_ankle_joint", "right_ankle_joint"]
    fn = SonicStyleReward(RewardConfig(), body_names, keypoints, joint_names)
    b, nb = 2, len(body_names)
    q_ident = torch.tensor([1.0, 0, 0, 0]).repeat(b, nb, 1)
    root_q = torch.tensor([1.0, 0, 0, 0]).repeat(b, 1)
    obs = SimpleNamespace(
        root_pos=torch.zeros(b, 3),
        root_quat=root_q,
        body_pos=torch.zeros(b, nb, 3),
        body_quat=q_ident,
        body_linvel=torch.zeros(b, nb, 3),
        body_angvel=torch.zeros(b, nb, 3),
        joint_pos=torch.zeros(b, 29),
        joint_vel=torch.zeros(b, 29),
    )
    ref = {
        "root_pos": obs.root_pos.clone(),
        "root_quat": obs.root_quat.clone(),
        "body_pos": obs.body_pos.clone(),
        "body_quat": obs.body_quat.clone(),
        "body_linvel": obs.body_linvel.clone(),
        "body_angvel": obs.body_angvel.clone(),
    }
    out = fn(
        obs,
        ref,
        torch.zeros(b, 29),
        torch.zeros(b, 29),
        torch.full((29,), -3.0),
        torch.full((29,), 3.0),
        torch.zeros(b, 29),
        0.02,
    )
    assert torch.all(out.total > 5.0)
    assert not out.done_tracking.any()


def test_horizontal_drift_does_not_terminate_local_tracking():
    body_names = ["pelvis", "torso", "left_wrist", "right_wrist", "left_ankle", "right_ankle"]
    keypoints = ["torso", "left_wrist", "right_wrist", "left_ankle", "right_ankle"]
    fn = SonicStyleReward(RewardConfig(), body_names, keypoints)
    b, nb = 1, len(body_names)
    root_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    body_q = root_q[:, None].expand(b, nb, 4).clone()
    local = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.3, 0.2],
          [0.0, -0.3, 0.2], [0.0, 0.1, -0.7], [0.0, -0.1, -0.7]]]
    )
    ref_root = torch.tensor([[0.0, 0.0, 0.8]])
    obs_root = torch.tensor([[3.0, -2.0, 0.8]])
    obs = SimpleNamespace(
        root_pos=obs_root,
        root_quat=root_q,
        body_pos=obs_root[:, None] + local,
        body_quat=body_q,
        body_linvel=torch.zeros(b, nb, 3),
        body_angvel=torch.zeros(b, nb, 3),
        joint_pos=torch.zeros(b, 29),
        joint_vel=torch.zeros(b, 29),
    )
    ref = {
        "root_pos": ref_root,
        "root_quat": root_q,
        "body_pos": ref_root[:, None] + local,
        "body_quat": body_q,
        "body_linvel": torch.zeros(b, nb, 3),
        "body_angvel": torch.zeros(b, nb, 3),
    }
    out = fn(
        obs, ref, torch.zeros(b, 29), torch.zeros(b, 29),
        torch.full((29,), -2.0), torch.full((29,), 2.0), None, 0.02,
    )
    assert not out.done_tracking.item()
    assert out.terms["relative_body_pos"].item() > 0.99


def test_anchor_height_still_terminates_a_fall():
    body_names = ["pelvis", "head", "left_wrist", "right_wrist", "left_ankle", "right_ankle"]
    keypoints = body_names[1:]
    fn = SonicStyleReward(RewardConfig(), body_names, keypoints)
    b, nb = 1, len(body_names)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(b, nb, 1)
    obs = SimpleNamespace(
        root_pos=torch.tensor([[0.0, 0.0, 0.5]]),
        root_quat=quat[:, 0],
        body_pos=torch.zeros(b, nb, 3),
        body_quat=quat,
        body_linvel=torch.zeros(b, nb, 3),
        body_angvel=torch.zeros(b, nb, 3),
        joint_pos=torch.zeros(b, 29),
        joint_vel=torch.zeros(b, 29),
    )
    ref = {
        "root_pos": torch.tensor([[0.0, 0.0, 0.8]]),
        "root_quat": quat[:, 0],
        "body_pos": torch.zeros(b, nb, 3),
        "body_quat": quat,
        "body_linvel": torch.zeros(b, nb, 3),
        "body_angvel": torch.zeros(b, nb, 3),
    }
    out = fn(
        obs, ref, torch.zeros(b, 29), torch.zeros(b, 29),
        torch.full((29,), -2.0), torch.full((29,), 2.0), None, 0.02,
    )
    assert out.done_tracking.item()
