from types import SimpleNamespace

import torch

from mini_groot_sonic.models.runtime import RecedingHorizonTokenController


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
