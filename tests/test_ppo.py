import torch
from torch import nn

from gear_sonic_mjx.config import ModelConfig, PPOConfig
from gear_sonic_mjx.trl.modules.universal_token_modules import UniversalTokenModule
from gear_sonic_mjx.trl.trainer.ppo_trainer_aux_loss import (
    PPOTrainer,
    RolloutStorage,
    SonicActorCritic,
)


def test_ppo_update_runs():
    cfgm = ModelConfig(
        token_dim=2,
        num_tokens=2,
        g1_encoder_hidden=[16],
        dynamic_decoder_hidden=[16],
        kinematic_decoder_hidden=[16],
        critic_hidden=[16],
    )
    token = UniversalTokenModule(cfgm, 2, 2)
    critic_dim = token.encoder_input_dim + token.proprio_dim
    critic = nn.Sequential(nn.Linear(critic_dim, 16), nn.Tanh(), nn.Linear(16, 1))
    ppo = PPOConfig(num_learning_epochs=1, num_mini_batches=1)
    ac = SonicActorCritic(token, critic, 29, ppo.init_noise_std)
    trainer = PPOTrainer(ac, ppo)
    st = RolloutStorage(
        2,
        3,
        token.encoder_input_dim,
        token.proprio_dim,
        critic_dim,
        29,
        torch.device("cpu"),
    )
    for _ in range(2):
        eo = torch.randn(3, token.encoder_input_dim)
        po = torch.randn(3, token.proprio_dim)
        co = torch.cat([eo, po], -1)
        with torch.no_grad():
            a, lp, _ = ac.act(eo, po)
            v = ac.value(co)
        st.add(eo, po, co, a, lp, v, torch.randn(3), torch.zeros(3, dtype=torch.bool))
    st.compute_returns(torch.zeros(3), 0.99, 0.95)
    out = trainer.update(st)
    assert "policy_loss" in out and torch.isfinite(torch.tensor(out["policy_loss"]))
