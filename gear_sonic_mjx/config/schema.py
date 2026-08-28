from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SimConfig:
    backend: str = "mjwarp"
    sim_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 10.0
    nconmax: int | None = None
    naconmax: int | None = None
    njmax: int | None = None

    @property
    def policy_dt(self) -> float:
        return self.sim_dt * self.decimation


@dataclass
class AdaptiveSamplingConfig:
    enabled: bool = True
    bin_size: int = 50
    init_num_failures: float = 1.0
    uniform_sampling_rate: float = 0.1
    pre_failure_sample_window: int = 200
    max_failure_over_mean: float = 200.0


@dataclass
class MotionConfig:
    source_fps: int = 120
    preprocess_fps: int = 30
    target_fps: int = 50
    num_future_frames: int = 10
    dt_future_ref_frames: float = 0.1
    actor_prop_history_length: int = 10
    actor_actions_history_length: int = 10
    freeze_frame_aug: bool = True
    freeze_frame_aug_prob: float = 0.1
    cat_upper_body_poses: bool = True
    cat_upper_body_poses_prob: float = 0.5
    adaptive_sampling: AdaptiveSamplingConfig = field(
        default_factory=AdaptiveSamplingConfig
    )


@dataclass
class ModelConfig:
    dof: int = 29
    token_dim: int = 32
    num_tokens: int = 2
    preset: str = "small"
    g1_encoder_hidden: list[int] = field(default_factory=lambda: [1024, 768, 512, 256])
    dynamic_decoder_hidden: list[int] = field(
        default_factory=lambda: [1536, 1536, 1024, 512, 512]
    )
    kinematic_decoder_hidden: list[int] = field(
        default_factory=lambda: [1024, 768, 512]
    )
    critic_hidden: list[int] = field(default_factory=lambda: [1536, 1024, 512, 256])

    @property
    def flat_token_dim(self) -> int:
        return self.token_dim * self.num_tokens

    @property
    def proprio_frame_dim(self) -> int:
        # NVIDIA SONIC actor frame: base_ang_vel(3), q(29), qd(29), prev_action(29), gravity(3)
        return 3 + self.dof + self.dof + self.dof + 3

    @property
    def g1_future_frame_dim(self) -> int:
        # q + qd + root orientation 6D
        return self.dof * 2 + 6

    @classmethod
    def nvidia_release(cls) -> ModelConfig:
        return cls(
            token_dim=32,
            num_tokens=2,
            preset="nvidia_release",
            g1_encoder_hidden=[2048, 1024, 512, 512],
            dynamic_decoder_hidden=[4096, 4096, 2048, 2048, 1024, 1024, 512, 512],
            kinematic_decoder_hidden=[2048, 1024, 512, 512],
            critic_hidden=[4096, 4096, 2048, 2048, 1024, 1024, 512, 512],
        )


@dataclass
class PPOConfig:
    num_learning_iterations: int = 100_000
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    actor_learning_rate: float = 2e-5
    critic_learning_rate: float = 1e-3
    max_grad_norm: float = 0.1
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    adaptive_lr_min: float = 1e-5
    adaptive_lr_max: float = 2e-4
    init_noise_std: float = 0.05
    std_clamp_min: float = 0.001
    std_clamp_max: float = 0.5
    save_interval: int = 500
    eval_frequency: int = 500
    aux_reconstruction_coef: float = 0.01


@dataclass
class RewardConfig:
    tracking_anchor_pos: float = 0.5
    tracking_anchor_ori: float = 0.5
    tracking_relative_body_pos: float = 1.0
    tracking_relative_body_ori: float = 1.0
    tracking_body_linvel: float = 1.0
    tracking_body_angvel: float = 1.0
    tracking_vr_5point_local: float = 2.0
    action_rate_l2: float = -0.1
    joint_limit: float = -10.0
    undesired_contacts: float = -0.1
    anti_shake_ang_vel: float = -0.005
    feet_acc: float = -2.5e-6
    std_anchor_pos: float = 0.3
    std_anchor_ori: float = 0.4
    std_relative_body_pos: float = 0.3
    std_relative_body_ori: float = 0.4
    std_body_linvel: float = 1.0
    std_body_angvel: float = 3.14
    std_vr_5point_local: float = 0.1


@dataclass
class ContactConfig:
    """Bodies exempt from SONIC's non-support-contact penalty.

    NVIDIA's release permits contacts on ankles, wrists, and elbows. Every other
    named robot body is penalized when its external contact force exceeds 1 N.
    """

    threshold: float = 1.0
    allowed_body_names: list[str] = field(
        default_factory=lambda: [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_elbow_link",
            "right_elbow_link",
        ]
    )


@dataclass
class TerminationConfig:
    anchor_pos: float = 0.15
    anchor_ori: float = 0.2
    ee_body_pos: float = 0.15
    foot_pos_xyz: float = 0.2
    down_threshold: float = 0.75
    root_height_threshold: float = 0.5


@dataclass
class ObservationNoiseConfig:
    enabled: bool = True
    gravity: float = 0.05
    base_ang_vel: float = 0.2
    joint_pos: float = 0.01
    joint_vel: float = 0.5
    tokenizer_orientation: float = 0.05


@dataclass
class SonicConfig:
    seed: int = 0
    num_envs: int = 4096
    sim: SimConfig = field(default_factory=SimConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    observation_noise: ObservationNoiseConfig = field(
        default_factory=ObservationNoiseConfig
    )
    domain_randomization: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SonicConfig:
        raw = yaml.safe_load(Path(path).read_text())
        adaptive = AdaptiveSamplingConfig(
            **raw.get("motion", {}).pop("adaptive_sampling", {})
        )
        motion = MotionConfig(**raw.get("motion", {}), adaptive_sampling=adaptive)
        cfg = cls(
            seed=raw.get("seed", 0),
            num_envs=raw.get("num_envs", 4096),
            sim=SimConfig(**raw.get("sim", {})),
            motion=motion,
            model=ModelConfig(**raw.get("model", {})),
            ppo=PPOConfig(**raw.get("ppo", {})),
            reward=RewardConfig(**raw.get("reward", {})),
            contact=ContactConfig(**raw.get("contact", {})),
            termination=TerminationConfig(**raw.get("termination", {})),
            observation_noise=ObservationNoiseConfig(
                **raw.get("observation_noise", {})
            ),
            domain_randomization=raw.get("domain_randomization", {}),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.model.dof != 29:
            raise ValueError(f"G1 SONIC requires 29 DOF, got {self.model.dof}")
        if self.sim.sim_dt <= 0 or self.sim.decimation <= 0:
            raise ValueError("sim_dt and decimation must be positive")
        expected_policy_dt = 1.0 / float(self.motion.target_fps)
        if abs(self.sim.policy_dt - expected_policy_dt) > 1e-9:
            raise ValueError(
                "simulation/control clocks disagree: "
                f"sim_dt*decimation={self.sim.policy_dt:g}s but target_fps="
                f"{self.motion.target_fps} requires {expected_policy_dt:g}s"
            )
        future_stride = self.motion.dt_future_ref_frames * self.motion.target_fps
        if abs(future_stride - round(future_stride)) > 1e-9:
            raise ValueError(
                "dt_future_ref_frames * target_fps must be an integer number of frames, "
                f"got {future_stride}"
            )
        if (
            self.motion.actor_prop_history_length
            != self.motion.actor_actions_history_length
        ):
            raise ValueError(
                "actor proprioception and action histories must have the same length"
            )
        if self.model.flat_token_dim != 64:
            raise ValueError(
                f"SONIC/GR00T interface requires a 64-D token, got {self.model.flat_token_dim}"
            )
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
