from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mini_groot_sonic.checkpoint import (
    BODY_CONTROL_STACK_VERSION,
    body_policy_fingerprint,
)
from mini_groot_sonic.config import SimConfig, SonicTinyConfig
from mini_groot_sonic.models.runtime import (
    RecedingHorizonTokenController,
    load_body_checkpoint,
    load_flow_checkpoint,
)
from mini_groot_sonic.models.sonic_tiny import TinySonicPolicy


class _Flow:
    def __init__(self):
        self.calls = 0

    def sample(self, state, text, vision, goal):
        value = float(self.calls)
        self.calls += 1
        return torch.full((state.shape[0], 40, 64), value, device=state.device)


class _Body:
    @staticmethod
    def project_external_token(token):
        return token

    @staticmethod
    def decode_token(proprio_history, token):
        return torch.full((token.shape[0], 29), 3.0, device=token.device)


class _Backbone:
    @staticmethod
    def encode_text(texts):
        return torch.zeros(len(texts), 8)


def _obs():
    return SimpleNamespace(
        joint_pos=torch.zeros(1, 29),
        joint_vel=torch.zeros(1, 29),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_pos=torch.tensor([[0.0, 0.0, 0.8]]),
        root_linvel=torch.zeros(1, 3),
        root_angvel=torch.zeros(1, 3),
    )


def test_replanning_blends_the_unconsumed_tail():
    controller = RecedingHorizonTokenController(_Flow(), _Body(), _Backbone(), replan_every=20)
    controller.replan("walk", _obs())
    controller._cursor = 20
    controller.replan("walk", _obs())
    assert torch.all(controller._tokens[:, 0] > 0)
    assert torch.all(controller._tokens[:, 0] < 1)
    assert torch.all(controller._tokens[:, 19] > controller._tokens[:, 0])


def test_runtime_does_not_tanh_body_actions():
    controller = RecedingHorizonTokenController(_Flow(), _Body(), _Backbone())
    action, _ = controller.action("walk", _obs(), torch.zeros(1, 1))
    torch.testing.assert_close(action, torch.full((1, 29), 3.0))


def test_body_loader_restores_checkpoint_simulator_contract(tmp_path: Path):
    sonic = SonicTinyConfig(
        encoder_hidden=(8,),
        controller_hidden=(8,),
        recon_hidden=(8,),
    )
    sim = SimConfig(
        mjcf=Path("trained.xml"),
        physics_dt=0.002,
        decimation=10,
        action_clip_value=20.0,
    )
    policy = TinySonicPolicy(sonic)
    policy_state = policy.state_dict()
    path = tmp_path / "body.pt"
    torch.save(
        {
            "control_stack_version": BODY_CONTROL_STACK_VERSION,
            "policy": policy_state,
            "body_policy_fingerprint": body_policy_fingerprint(policy_state),
            "sonic_cfg": asdict(sonic),
            "sim_cfg": asdict(sim),
        },
        path,
    )
    _, loaded_sonic, loaded_sim = load_body_checkpoint(path, "cpu")
    assert loaded_sonic == sonic
    assert loaded_sim.physics_dt == 0.002
    assert loaded_sim.decimation == 10
    assert loaded_sim.action_clip_value == 20.0
    assert loaded_sim.device == "cpu"


def test_flow_loader_rejects_a_different_body_codebook(tmp_path: Path):
    path = tmp_path / "flow.pt"
    torch.save(
        {
            "body_control_stack_version": BODY_CONTROL_STACK_VERSION,
            "body_policy_fingerprint": "body-a",
        },
        path,
    )
    with pytest.raises(RuntimeError, match="different body policy/codebook"):
        load_flow_checkpoint(
            path,
            "cpu",
            expected_body_fingerprint="body-b",
        )
