from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


class SonicLiteOnPolicyRunner(MjlabOnPolicyRunner):
    """Local-motion compatibility adapter for mjlab's tracking launcher."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", *, registry_name=None):
        del registry_name
        super().__init__(env, train_cfg, log_dir, device)


@dataclass(kw_only=True)
class SonicLiteModelCfg(RslRlModelCfg):
    class_name: str = "sonic_lite_g1.model:SonicLiteActor"
    future_group: str = "future"
    proprio_group: str = "proprio"
    token_dim: int = 64
    quantization_levels: int = 32
    encoder_hidden_dims: tuple[int, ...] = (256, 128)


def sonic_lite_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config: current mjlab G1 tracker defaults, with a small token actor."""
    return RslRlOnPolicyRunnerCfg(
        actor=SonicLiteModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "actor": ["future", "proprio"],
            "critic": ["critic"],
        },
        clip_actions=1.0,
        experiment_name="sonic_lite_g1",
        logger="tensorboard",
        save_interval=250,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
