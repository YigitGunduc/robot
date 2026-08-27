from dataclasses import asdict

import pytest
import torch

from mini_groot_sonic.checkpoint import (
    BODY_CONTROL_STACK_VERSION,
    body_policy_fingerprint,
    require_current_body_control_stack,
    require_matching_control_config,
)
from mini_groot_sonic.config import SimConfig


def test_rejects_legacy_body_checkpoint_action_semantics():
    with pytest.raises(RuntimeError, match="Start a new body run"):
        require_current_body_control_stack({"policy": {}})


def test_accepts_current_body_checkpoint():
    require_current_body_control_stack(
        {"control_stack_version": BODY_CONTROL_STACK_VERSION, "policy": {}}
    )


def test_resume_rejects_changed_action_contract():
    saved = SimConfig(action_clip_value=20.0)
    checkpoint = {"sim_cfg": asdict(saved)}
    require_matching_control_config(checkpoint, saved)
    with pytest.raises(RuntimeError, match="action_clip_value"):
        require_matching_control_config(
            checkpoint,
            SimConfig(action_clip_value=1.0),
        )


def test_resume_allows_device_and_randomization_changes():
    saved = SimConfig(device="cuda:0", enable_randomization=False)
    require_matching_control_config(
        {"sim_cfg": asdict(saved)},
        SimConfig(device="cuda:1", enable_randomization=True),
    )


def test_body_policy_fingerprint_is_stable_and_weight_sensitive():
    first = {"weight": torch.tensor([[1.0, 2.0]])}
    second = {"weight": torch.tensor([[1.0, 3.0]])}
    assert body_policy_fingerprint(first) == body_policy_fingerprint(first)
    assert body_policy_fingerprint(first) != body_policy_fingerprint(second)
