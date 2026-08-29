from __future__ import annotations

from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from .commands import PackedFutureMotionCommandCfg


def _clone_motion_cfg(old) -> PackedFutureMotionCommandCfg:
    return PackedFutureMotionCommandCfg(
        entity_name=old.entity_name,
        resampling_time_range=old.resampling_time_range,
        debug_vis=old.debug_vis,
        motion_file=old.motion_file,
        anchor_body_name=old.anchor_body_name,
        body_names=old.body_names,
        pose_range=dict(old.pose_range),
        velocity_range=dict(old.velocity_range),
        joint_position_range=old.joint_position_range,
        adaptive_kernel_size=old.adaptive_kernel_size,
        adaptive_lambda=old.adaptive_lambda,
        adaptive_uniform_ratio=old.adaptive_uniform_ratio,
        adaptive_alpha=old.adaptive_alpha,
        sampling_mode=old.sampling_mode,
        viz=old.viz,
    )


def sonic_lite_g1_env_cfg(*, play: bool = False):
    """Small tokenized G1 motion-tracking environment.

    We reuse mjlab's current G1 tracker for rewards, contacts, actuator limits,
    action scaling, reset-to-reference, and simulation. Only observations and
    the motion command are changed.
    """
    cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)

    old_motion = cfg.commands["motion"]
    motion = _clone_motion_cfg(old_motion)
    # Mild random-state initialization: enough to prevent memorizing the exact
    # mocap trajectory, but gentler than the full sim-to-real recipe.
    motion.pose_range = {
        "x": (-0.03, 0.03),
        "y": (-0.03, 0.03),
        "z": (-0.01, 0.01),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.10, 0.10),
    }
    motion.velocity_range = {
        "x": (-0.15, 0.15),
        "y": (-0.15, 0.15),
        "z": (-0.05, 0.05),
        "roll": (-0.10, 0.10),
        "pitch": (-0.10, 0.10),
        "yaw": (-0.20, 0.20),
    }
    motion.joint_position_range = (-0.05, 0.05)
    motion.sampling_mode = "adaptive"
    cfg.commands["motion"] = motion

    # Keep future reference separate so the custom actor can force it through
    # the 64-D token bottleneck. Reuse the proven baseline terms for proprio.
    base_actor = cfg.observations["actor"]
    future_terms = {"command": base_actor.terms["command"]}
    proprio_terms = {
        name: term
        for name, term in base_actor.terms.items()
        if name != "command"
    }
    cfg.observations.pop("actor")
    cfg.observations["future"] = ObservationGroupCfg(
        terms=future_terms,
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
    )
    cfg.observations["proprio"] = ObservationGroupCfg(
        terms=proprio_terms,
        concatenate_terms=True,
        enable_corruption=not play,
        history_length=10,
    )

    # The stock critic is intentionally left privileged/asymmetric. It now sees
    # the 5-frame future command because generated_commands() calls our command.

    if play:
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.joint_position_range = (0.0, 0.0)
        motion.sampling_mode = "start"
        # Deterministic evaluation: no pushes or physical-parameter randomization.
        cfg.events.pop("push_robot", None)
        cfg.events.pop("base_com", None)
        cfg.events.pop("foot_friction", None)
    else:
        # For first proof-of-concept training, don't fight large pushes or body
        # parameter randomization. Add these only after tracking is competent.
        cfg.events.pop("push_robot", None)
        cfg.events.pop("base_com", None)
        # Keep nominal contact physics for the initial curriculum.
        cfg.events.pop("foot_friction", None)

    return cfg
