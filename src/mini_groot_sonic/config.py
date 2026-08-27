from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SonicTinyConfig:
    dof: int = 29
    token_dim: int = 64
    token_groups: int = 2
    fsq_levels: int = 32
    future_frames: int = 10
    future_stride: int = 5  # 5 x 20 ms = 100 ms between reference frames at 50 Hz
    prop_history: int = 10
    encoder_hidden: Sequence[int] = (512, 512, 256)
    controller_hidden: Sequence[int] = (768, 768, 512, 256)
    recon_hidden: Sequence[int] = (512, 512, 256)
    critic_hidden: Sequence[int] = (768, 512, 256)
    init_action_std: float = 0.05
    min_action_std: float = 0.001
    max_action_std: float = 0.5
    # q, qdot, and reference-vs-current-robot root rotation (6D), matching the
    # released SONIC G1 robot encoder.
    root_reference_dim: int = 6
    critic_privileged_dim: int = 68

    @property
    def proprio_dim_per_frame(self) -> int:
        # projected gravity, base angular velocity, q relative to default, qdot,
        # previous action -- the released SONIC local_dir_hist ordering.
        return self.dof + self.dof + 3 + 3 + self.dof

    @property
    def proprio_dim(self) -> int:
        return self.prop_history * self.proprio_dim_per_frame

    @property
    def reference_dim(self) -> int:
        return self.future_frames * self.reference_frame_dim

    @property
    def reference_frame_dim(self) -> int:
        return self.dof * 2 + self.root_reference_dim


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
    # q, qdot, projected gravity, root-local linear/angular velocity, root height.
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
    noise_scale: float = 0.999
    state_dropout_prob: float = 0.2
    condition_dropout_prob: float = 0.1
    goal_slot_dim: int = 8
    seed: int = 0


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
    low_motion_root_height: float = 0.5
    terminate_low_motion_height: float = 0.75


@dataclass
class PPOConfig:
    num_envs: int = 512
    rollout_steps: int = 32
    ppo_epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    actor_lr: float = 2e-5
    actor_lr_min: float = 1e-5
    actor_lr_max: float = 2e-4
    kl_adaptation_factor: float = 1.5
    critic_lr: float = 1e-3
    aux_recon_coef: float = 0.01
    max_grad_norm: float = 0.1
    target_kl: float = 0.01
    seed: int = 0
    checkpoint_interval: int = 100
    eval_interval: int = 100
    failure_sampling_alpha: float = 0.9
    failure_sampling_cap: float = 200.0
    adaptive_sampling_bin_frames: int = 50
    pre_failure_sample_window: int = 200
    freeze_frame_probability: float = 0.1


@dataclass
class SimConfig:
    mjcf: Path = Path("g1.xml")
    device: str = "cuda:0"
    physics_dt: float = 0.005
    decimation: int = 4
    nconmax: int = 48
    njmax: int = 288
    # With sonic_g1_control enabled, None selects SONIC's calibrated per-joint
    # residual scales. Otherwise None retains the legacy full-joint-range mapping.
    action_scale: float | None = None
    # Released SONIC keeps Gaussian actions unbounded at the policy and clips
    # only at the environment boundary.
    action_clip_value: float = 20.0
    sonic_g1_control: bool = True
    actuator_mode: str = "auto"  # auto, position, or pd_torque
    joint_stiffness: float = 80.0
    joint_damping: float = 2.0
    joint_limit_margin: float = 0.02
    soft_joint_limit_factor: float = 0.9
    undesired_contact_penetration: float = 1e-4
    enable_randomization: bool = False
    enable_observation_noise: bool = True
    reset_joint_noise: float = 0.02
    reset_velocity_noise: float = 0.05
    observation_joint_pos_noise: float = 0.01
    observation_joint_vel_noise: float = 0.5
    observation_angular_vel_noise: float = 0.2
    gravity_noise: float = 0.05
    reference_joint_noise: float = 0.0
    reference_root_noise: float = 0.05
    motor_strength_range: tuple[float, float] = (0.9, 1.1)
    friction_scale_range: tuple[float, float] = (0.8, 1.2)
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    center_of_mass_noise: float = 0.01
    stiffness_range: tuple[float, float] = (0.9, 1.1)
    damping_range: tuple[float, float] = (0.9, 1.1)
    action_delay_probability: float = 0.15
    push_probability_per_step: float = 0.002
    push_velocity: float = 0.5
    root_body_name: str = "pelvis"
    keypoint_body_names: Sequence[str] = (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    allowed_contact_body_names: Sequence[str] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
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


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load the small YAML config without introducing a Hydra dependency."""
    import yaml

    with Path(path).open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    def section(name: str, cls):
        values = dict(raw.get(name, {}))
        if cls is SimConfig and "mjcf" in values:
            values["mjcf"] = Path(values["mjcf"])
        if cls is ReplayConfig and "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        return cls(**values)

    return ProjectConfig(
        sonic=section("sonic", SonicTinyConfig),
        goal=section("goal", GoalConfig),
        flow=section("flow", FlowConfig),
        reward=section("reward", RewardConfig),
        ppo=section("ppo", PPOConfig),
        sim=section("sim", SimConfig),
        replay=section("replay", ReplayConfig),
    )
