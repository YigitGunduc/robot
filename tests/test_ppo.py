from mini_groot_sonic.config import PPOConfig, SonicTinyConfig
from mini_groot_sonic.models.sonic_tiny import TinySonicCritic, TinySonicPolicy
from mini_groot_sonic.training.ppo import PPOAuxTrainer


def _trainer() -> PPOAuxTrainer:
    sonic = SonicTinyConfig(
        encoder_hidden=(16,),
        controller_hidden=(16,),
        recon_hidden=(16,),
        critic_hidden=(16,),
    )
    ppo = PPOConfig(
        actor_lr=2e-5,
        actor_lr_min=1e-5,
        actor_lr_max=2e-4,
        target_kl=0.01,
    )
    return PPOAuxTrainer(
        TinySonicPolicy(sonic),
        TinySonicCritic(sonic),
        sonic,
        ppo,
    )


def test_kl_schedule_reduces_and_increases_actor_lr_with_bounds():
    trainer = _trainer()
    trainer._adapt_actor_learning_rate(0.03)
    assert trainer.actor_lr < 2e-5
    trainer._adapt_actor_learning_rate(0.001)
    assert trainer.actor_lr == 2e-5
    for _ in range(20):
        trainer._adapt_actor_learning_rate(0.001)
    assert trainer.actor_lr == trainer.cfg.actor_lr_max
    for _ in range(30):
        trainer._adapt_actor_learning_rate(0.03)
    assert trainer.actor_lr == trainer.cfg.actor_lr_min
