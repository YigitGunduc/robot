from __future__ import annotations

import math

import torch

# Bones-SEED / MuJoCo actuator order. This order is the canonical order in this package.
G1_MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

BONES_CSV_JOINT_NAMES = [f"{n}_dof" for n in G1_MUJOCO_JOINT_NAMES]

# Body set used by the released SONIC tracking command.
SONIC_TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# The released sonic_release experiment overrides the generic five-point list with
# torso + both wrists. The historical reward function keeps the name “5point”.
SONIC_REWARD_POINT_BODY_NAMES = [
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
SONIC_REWARD_POINT_OFFSETS = torch.tensor(
    [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
)
SONIC_FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
SONIC_ANTI_SHAKE_BODY_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "head_link",
]
SONIC_EE_TERMINATION_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# For each MuJoCo index, the corresponding IsaacLab policy index.
MJ_TO_IL = torch.tensor(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=torch.long,
)
# For each IsaacLab index, the corresponding MuJoCo index.
IL_TO_MJ = torch.tensor(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=torch.long,
)

DEFAULT_ANGLES_MJ = torch.tensor(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=torch.float32,
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
OMEGA = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0


def _kp(armature: float) -> float:
    return armature * OMEGA * OMEGA


def _kd(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * OMEGA


K5020, K7520_14, K7520_22, K4010 = map(
    _kp, [ARMATURE_5020, ARMATURE_7520_14, ARMATURE_7520_22, ARMATURE_4010]
)
D5020, D7520_14, D7520_22, D4010 = map(
    _kd, [ARMATURE_5020, ARMATURE_7520_14, ARMATURE_7520_22, ARMATURE_4010]
)

KP_MJ = torch.tensor(
    [
        K7520_22,
        K7520_22,
        K7520_14,
        K7520_22,
        2 * K5020,
        2 * K5020,
        K7520_22,
        K7520_22,
        K7520_14,
        K7520_22,
        2 * K5020,
        2 * K5020,
        K7520_14,
        2 * K5020,
        2 * K5020,
        K5020,
        K5020,
        K5020,
        K5020,
        K5020,
        K4010,
        K4010,
        K5020,
        K5020,
        K5020,
        K5020,
        K5020,
        K4010,
        K4010,
    ],
    dtype=torch.float32,
)

KD_MJ = torch.tensor(
    [
        D7520_22,
        D7520_22,
        D7520_14,
        D7520_22,
        2 * D5020,
        2 * D5020,
        D7520_22,
        D7520_22,
        D7520_14,
        D7520_22,
        2 * D5020,
        2 * D5020,
        D7520_14,
        2 * D5020,
        2 * D5020,
        D5020,
        D5020,
        D5020,
        D5020,
        D5020,
        D4010,
        D4010,
        D5020,
        D5020,
        D5020,
        D5020,
        D5020,
        D4010,
        D4010,
    ],
    dtype=torch.float32,
)

EFFORT = torch.tensor(
    [
        139,
        139,
        88,
        139,
        25,
        25,
        139,
        139,
        88,
        139,
        25,
        25,
        88,
        25,
        25,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
    ],
    dtype=torch.float32,
)

# NVIDIA deploy definition: 0.25 * effort_limit / base stiffness, before the
# ankle and waist roll/pitch Kp multipliers above are applied.
_BASE_KP_FOR_SCALE = torch.tensor(
    [
        K7520_22,
        K7520_22,
        K7520_14,
        K7520_22,
        K5020,
        K5020,
        K7520_22,
        K7520_22,
        K7520_14,
        K7520_22,
        K5020,
        K5020,
        K7520_14,
        K5020,
        K5020,
        K5020,
        K5020,
        K5020,
        K5020,
        K5020,
        K4010,
        K4010,
        K5020,
        K5020,
        K5020,
        K5020,
        K5020,
        K4010,
        K4010,
    ],
    dtype=torch.float32,
)
ACTION_SCALE_MJ = 0.25 * EFFORT / _BASE_KP_FOR_SCALE

LOWER_BODY_MJ = torch.arange(0, 12, dtype=torch.long)
UPPER_BODY_MJ = torch.arange(12, 29, dtype=torch.long)
WRISTS_MJ = torch.tensor([19, 20, 21, 26, 27, 28], dtype=torch.long)
ANKLES_MJ = torch.tensor([4, 5, 10, 11], dtype=torch.long)


def to_policy_order_mj_to_il(x: torch.Tensor) -> torch.Tensor:
    # IL_TO_MJ[il_index] tells us which MuJoCo element belongs in that IsaacLab slot.
    return x.index_select(-1, IL_TO_MJ.to(x.device))


def to_mujoco_order_il_to_mj(x: torch.Tensor) -> torch.Tensor:
    # MJ_TO_IL[mj_index] tells us which IsaacLab element belongs in that MuJoCo slot.
    return x.index_select(-1, MJ_TO_IL.to(x.device))


def action_to_target_q(action_mj: torch.Tensor) -> torch.Tensor:
    return DEFAULT_ANGLES_MJ.to(action_mj) + ACTION_SCALE_MJ.to(action_mj) * action_mj
