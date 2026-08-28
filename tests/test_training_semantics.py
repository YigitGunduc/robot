import json
import random

import numpy as np
import pytest
import torch

from gear_sonic_mjx.checkpoint_utils import capture_rng_state, restore_rng_state
from gear_sonic_mjx.config import SonicConfig, TerminationConfig
from gear_sonic_mjx.data_process.splits import (
    build_split_manifest,
    motion_group_key,
    validate_split_manifest,
)
from gear_sonic_mjx.envs.adaptive_sampling import (
    AdaptiveMotionSampler,
    AdaptiveSamplerConfig,
)
from gear_sonic_mjx.envs.mdp.actions import joint_position_target, pd_torque
from gear_sonic_mjx.envs.mdp.terminations import TerminationMetrics, termination_mask
from gear_sonic_mjx.g1_parameters import EFFORT, KD_MJ, KP_MJ


def test_action_scale_and_pd_gain_contract_at_default_pose():
    action = torch.ones(1, 29)
    target = joint_position_target(action)
    torque = pd_torque(
        target,
        joint_position_target(torch.zeros_like(action)),
        torch.zeros_like(action),
        KP_MJ,
        KD_MJ,
    )
    expected = 0.25 * EFFORT
    # NVIDIA doubles ankle and waist roll/pitch Kp after deriving action scale from base Kp.
    expected[torch.tensor([4, 5, 10, 11, 13, 14])] *= 2.0
    torch.testing.assert_close(torque, expected[None], rtol=1e-5, atol=1e-5)


def test_termination_matches_release_semantics():
    cfg = TerminationConfig()
    metrics = TerminationMetrics(
        root_height_error=torch.tensor([0.16, 0.0, 0.0]),
        root_ori_error=torch.tensor([0.0, 0.45, 0.0]),
        ee_height_error=torch.tensor([0.0, 0.0, 0.70]),
        reference_root_height=torch.tensor([1.0, 1.0, 0.4]),
    )
    done, reasons = termination_mask(metrics, cfg)
    assert done.tolist() == [True, True, False]
    assert reasons["anchor_pos"].tolist() == [True, False, False]
    # The released orientation term compares squared angle against 0.2.
    assert reasons["anchor_ori"].tolist() == [False, True, False]
    # A low reference root activates the 0.75 m down-motion EE tolerance.
    assert reasons["ee_body_pos"].tolist() == [False, False, False]


def test_config_rejects_clock_mismatch():
    cfg = SonicConfig()
    cfg.sim.decimation = 3
    with pytest.raises(ValueError, match="clocks disagree"):
        cfg.validate()


def test_split_groups_mirrors_and_prevents_leakage(tmp_path):
    names = [
        "walk_001.npz",
        "walk_001_mirrored.npz",
        "run_001.npz",
        "wave_001.npz",
        "turn_001.npz",
        "jump_001.npz",
        "crouch_001.npz",
        "dance_001.npz",
        "kneel_001.npz",
        "crawl_001.npz",
        "gesture_001.npz",
        "kick_001.npz",
        "reach_001.npz",
        "stand_001.npz",
        "sidestep_001.npz",
        "hop_001.npz",
        "spin_001.npz",
        "bow_001.npz",
        "squat_001.npz",
        "march_001.npz",
    ]
    for name in names:
        (tmp_path / name).touch()
    manifest = build_split_manifest(
        tmp_path, seed=7, train_fraction=0.7, validation_fraction=0.15
    )
    validate_split_manifest(manifest)
    assert motion_group_key("walk_001_mirrored.npz") == motion_group_key("walk_001.npz")
    owners = {
        rel: split for split, relpaths in manifest["splits"].items() for rel in relpaths
    }
    assert owners["walk_001.npz"] == owners["walk_001_mirrored.npz"]
    # The manifest is JSON-safe so it can be persisted and audited before training.
    json.dumps(manifest)


def test_sampler_and_rng_resume_exactly():
    sampler = AdaptiveMotionSampler(
        torch.tensor([100, 80]), AdaptiveSamplerConfig(bin_size=20)
    )
    sampler.record(
        torch.tensor([0, 1]), torch.tensor([5, 45]), torch.tensor([True, False])
    )
    saved_sampler = sampler.state_dict()

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    saved_rng = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))

    restored = AdaptiveMotionSampler(
        torch.tensor([100, 80]), AdaptiveSamplerConfig(bin_size=20)
    )
    restored.load_state_dict(saved_sampler)
    restore_rng_state(saved_rng)
    actual = (random.random(), np.random.rand(), torch.rand(3))

    torch.testing.assert_close(restored.attempts, sampler.attempts)
    torch.testing.assert_close(restored.failures, sampler.failures)
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])
