import torch

from gear_sonic_mjx.config import ModelConfig
from gear_sonic_mjx.envs.mdp.observations import ProprioHistory, g1_tokenizer_observation
from gear_sonic_mjx.trl.modules.universal_token_modules import UniversalTokenModule


def test_nvidia_release_observation_shapes():
    cfg = ModelConfig.nvidia_release()
    b = 3
    q = torch.zeros(b, 10, 29)
    qd = torch.zeros_like(q)
    root = torch.tensor([[1.0, 0, 0, 0]]).repeat(b, 1)
    future_root = root[:, None].repeat(1, 10, 1)
    enc = g1_tokenizer_observation(q, qd, root, future_root)
    assert enc.shape == (b, 640)
    hist = ProprioHistory(b, 29, 10)
    prop = hist.push(torch.zeros(b,3), torch.zeros(b,29), torch.zeros(b,29), torch.zeros(b,29), torch.zeros(b,3))
    assert prop.shape == (b, 930)
    model = UniversalTokenModule(cfg, 10, 10)
    assert model.flat_token_dim == 64
    assert model.dynamic_input_dim == 994


def test_small_model_forward():
    cfg = ModelConfig(
        token_dim=4, num_tokens=2,
        g1_encoder_hidden=[32], dynamic_decoder_hidden=[32], kinematic_decoder_hidden=[32], critic_hidden=[32],
    )
    model = UniversalTokenModule(cfg, num_future_frames=2, history_length=2)
    b = 4
    enc = torch.randn(b, model.encoder_input_dim)
    prop = torch.randn(b, model.proprio_dim)
    out = model(enc, prop)
    assert out.token.shape == (b, 2, 4)
    assert out.action_mean.shape == (b, 29)
    assert out.reconstruction.shape == enc.shape


def test_privileged_history_shape():
    import torch
    from gear_sonic_mjx.envs.mdp.observations import PrivilegedHistory
    h = PrivilegedHistory(4, 29, 10)
    out = h.push(torch.zeros(4,3), torch.zeros(4,3), torch.zeros(4,29), torch.zeros(4,29), torch.zeros(4,29))
    assert out.shape == (4, 930)


def test_nvidia_release_widths():
    from gear_sonic_mjx.config import ModelConfig
    c = ModelConfig.nvidia_release()
    assert c.dynamic_decoder_hidden == [4096, 4096, 2048, 2048, 1024, 1024, 512, 512]
    assert c.critic_hidden == [4096, 4096, 2048, 2048, 1024, 1024, 512, 512]
