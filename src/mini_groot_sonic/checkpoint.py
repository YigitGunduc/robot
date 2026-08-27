from __future__ import annotations

import hashlib
from dataclasses import asdict

import torch

from mini_groot_sonic.config import SimConfig

BODY_CONTROL_STACK_VERSION = 3

_CONTROL_CONFIG_FIELDS = (
    "physics_dt",
    "decimation",
    "action_scale",
    "action_clip_value",
    "sonic_g1_control",
    "actuator_mode",
    "joint_stiffness",
    "joint_damping",
    "joint_limit_margin",
    "soft_joint_limit_factor",
    "enable_observation_noise",
    "observation_joint_pos_noise",
    "observation_joint_vel_noise",
    "observation_angular_vel_noise",
    "gravity_noise",
    "reference_joint_noise",
    "reference_root_noise",
    "root_body_name",
    "keypoint_body_names",
    "allowed_contact_body_names",
)


def body_policy_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """Return a stable identity for the exact body policy/codebook weights."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require_current_body_control_stack(checkpoint: dict) -> None:
    version = int(checkpoint.get("control_stack_version", 1))
    if version != BODY_CONTROL_STACK_VERSION:
        raise RuntimeError(
            f"Checkpoint control stack v{version} is incompatible with "
            f"v{BODY_CONTROL_STACK_VERSION}. Start a new body run: action scaling, "
            "Gaussian action semantics, proprioception, and simulator ownership changed."
        )


def require_matching_control_config(checkpoint: dict, sim_cfg: SimConfig) -> None:
    """Reject resumes that would change the trained action/observation contract."""

    saved = checkpoint.get("sim_cfg")
    if not isinstance(saved, dict):
        raise TypeError("Current body checkpoint is missing its simulator configuration")
    current = asdict(sim_cfg)
    mismatches = []
    for name in _CONTROL_CONFIG_FIELDS:
        saved_value = saved.get(name)
        current_value = current[name]
        if isinstance(saved_value, (list, tuple)):
            saved_value = tuple(saved_value)
        if isinstance(current_value, (list, tuple)):
            current_value = tuple(current_value)
        if saved_value != current_value:
            mismatches.append(f"{name}: checkpoint={saved_value!r}, current={current_value!r}")
    if mismatches:
        raise RuntimeError(
            "Refusing to resume with a different SONIC control contract:\n"
            + "\n".join(mismatches)
        )
