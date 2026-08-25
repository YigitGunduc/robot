from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class SonicTinyConfig:
    dof: int = 29
    token_dim: int = 64
    token_groups: int = 2
    fsq_levels: int = 32
    future_frames: int = 10
    future_stride: int = 5  # 5 x 20 ms = 100 ms between reference frames at 50 Hz
    prop_history: int = 10
    action_scale: float = 0.5
    encoder_hidden: Sequence[int] = (512, 512, 256)
    controller_hidden: Sequence[int] = (768, 768, 512, 256)
    recon_hidden: Sequence[int] = (512, 512, 256)
    critic_hidden: Sequence[int] = (768, 512, 256)
    init_action_std: float = 0.05
    min_action_std: float = 0.001
    max_action_std: float = 0.5

    @property
    def proprio_dim_per_frame(self) -> int:
        # q, qdot, base angular velocity, projected gravity, previous action
        return self.dof + self.dof + 3 + 3 + self.dof

    @property
    def proprio_dim(self) -> int:
        return self.prop_history * self.proprio_dim_per_frame

    @property
    def reference_dim(self) -> int:
        # G1 encoder input: future q + qdot, matching the released SONIC G1 encoder concept.
        return self.future_frames * self.dof * 2


@dataclass
class GoalConfig:
    # Sparse task-space conditioning. Pose is xyz + quaternion wxyz.
    slot_names: Sequence[str] = (
        "root",
        "head",
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
    )
    pose_dim: int = 7
    hidden: Sequence[int] = (256, 256)

    @property
    def flat_dim(self) -> int:
        # pose + one active-mask scalar for each slot
        return len(self.slot_names) * (self.pose_dim + 1)


@dataclass
class FlowConfig:
    action_dim: int = 64
    action_horizon: int = 40
    state_dim: int = 68
    text_dim: int = 768
    vision_dim: int = 768
    goal_dim: int = 48
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    inference_steps: int = 4
    beta_alpha: float = 1.5
    beta_beta: float = 1.0


@dataclass
class RewardConfig:
    anchor_pos_weight: float = 0.5
    anchor_ori_weight: float = 0.5
    relative_body_pos_weight: float = 1.0
    relative_body_ori_weight: float = 1.0
    body_linvel_weight: float = 1.0
    body_angvel_weight: float = 1.0
    keypoint_weight: float = 2.0
    action_rate_weight: float = -0.1
    joint_limit_weight: float = -10.0
    undesired_contact_weight: float = -0.1
    anti_shake_weight: float = -0.005
    feet_acc_weight: float = -2.5e-6

    anchor_pos_std: float = 0.3
    anchor_ori_std: float = 0.4
    relative_body_pos_std: float = 0.3
    relative_body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    keypoint_std: float = 0.1

    terminate_anchor_pos: float = 0.15
    terminate_anchor_ori: float = 0.2
    terminate_ee_pos: float = 0.15
    terminate_foot_pos: float = 0.2


@dataclass
class PPOConfig:
    num_envs: int = 512
    rollout_steps: int = 32
    ppo_epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.013
    value_coef: float = 1.0
    actor_lr: float = 2e-5
    critic_lr: float = 1e-3
    aux_recon_coef: float = 0.01
    max_grad_norm: float = 0.1
    target_kl: float = 0.01


@dataclass
class SimConfig:
    mjcf: Path = Path("g1.xml")
    device: str = "cuda:0"
    physics_dt: float = 0.005
    decimation: int = 4
    nconmax: int = 48
    njmax: int = 288
    action_scale: float = 0.5
    root_body_name: str = "pelvis"
    keypoint_body_names: Sequence[str] = (
        "head_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )


@dataclass
class ReplayConfig:
    output_dir: Path = Path("replays")
    control_hz: int = 50
    camera_hz: int = 10
    save_rgb: bool = False
    camera_name: str | None = None
    width: int = 224
    height: int = 224


@dataclass
class ProjectConfig:
    sonic: SonicTinyConfig = field(default_factory=SonicTinyConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
