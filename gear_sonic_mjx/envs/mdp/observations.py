from __future__ import annotations

from dataclasses import dataclass

import torch

from gear_sonic_mjx.math_utils import relative_rotation_6d


@dataclass
class ProprioHistory:
    """Actor history matching the released SONIC observation order.

    Per frame: base angular velocity (3), q-default (29), qd (29), previous action (29),
    projected gravity (3) = 93 dimensions. Ten frames = 930 dimensions.
    """

    num_envs: int
    dof: int = 29
    length: int = 10
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.frame_dim = 3 + self.dof + self.dof + self.dof + 3
        self.buffer = torch.zeros(self.num_envs, self.length, self.frame_dim, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.buffer.zero_()
        else:
            self.buffer[env_ids] = 0.0

    def make_frame(
        self,
        base_ang_vel: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        prev_action: torch.Tensor,
        gravity_dir: torch.Tensor,
    ) -> torch.Tensor:
        frame = torch.cat([base_ang_vel, joint_pos_rel, joint_vel, prev_action, gravity_dir], dim=-1)
        if frame.shape[-1] != self.frame_dim:
            raise ValueError(f"Expected proprio frame dim {self.frame_dim}, got {frame.shape[-1]}")
        return frame

    def seed(
        self,
        env_ids: torch.Tensor,
        base_ang_vel: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        prev_action: torch.Tensor,
        gravity_dir: torch.Tensor,
    ) -> None:
        """Fill only reset environments with their current frame, without advancing other worlds."""
        frame = self.make_frame(base_ang_vel, joint_pos_rel, joint_vel, prev_action, gravity_dir)
        if frame.shape[0] == self.num_envs:
            frame = frame.index_select(0, env_ids)
        if frame.shape[0] != env_ids.numel():
            raise ValueError("seed inputs must be full-batch tensors or match env_ids")
        self.buffer[env_ids] = frame[:, None, :].expand(-1, self.length, -1)

    def push(
        self,
        base_ang_vel: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        prev_action: torch.Tensor,
        gravity_dir: torch.Tensor,
    ) -> torch.Tensor:
        frame = self.make_frame(base_ang_vel, joint_pos_rel, joint_vel, prev_action, gravity_dir)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1] = frame
        return self.flat()

    def flat(self) -> torch.Tensor:
        return self.buffer.reshape(self.num_envs, -1)


@dataclass
class PrivilegedHistory:
    """Clean critic history close to SONIC's released asymmetric critic.

    Per frame: base linear velocity (3), base angular velocity (3), q-default (29), qd (29),
    previous action (29) = 93 dimensions. The critic additionally receives future command and
    current privileged body state in :class:`G1SonicTrackingTask`.
    """

    num_envs: int
    dof: int = 29
    length: int = 10
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.frame_dim = 3 + 3 + self.dof + self.dof + self.dof
        self.buffer = torch.zeros(self.num_envs, self.length, self.frame_dim, device=self.device)

    def make_frame(
        self,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        frame = torch.cat([base_lin_vel, base_ang_vel, joint_pos_rel, joint_vel, prev_action], dim=-1)
        if frame.shape[-1] != self.frame_dim:
            raise ValueError(f"Expected critic-history frame dim {self.frame_dim}, got {frame.shape[-1]}")
        return frame

    def seed(
        self,
        env_ids: torch.Tensor,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> None:
        frame = self.make_frame(base_lin_vel, base_ang_vel, joint_pos_rel, joint_vel, prev_action)
        if frame.shape[0] == self.num_envs:
            frame = frame.index_select(0, env_ids)
        if frame.shape[0] != env_ids.numel():
            raise ValueError("seed inputs must be full-batch tensors or match env_ids")
        self.buffer[env_ids] = frame[:, None, :].expand(-1, self.length, -1)

    def push(
        self,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        frame = self.make_frame(base_lin_vel, base_ang_vel, joint_pos_rel, joint_vel, prev_action)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1] = frame
        return self.flat()

    def flat(self) -> torch.Tensor:
        return self.buffer.reshape(self.num_envs, -1)


def g1_tokenizer_observation(
    future_joint_pos: torch.Tensor,
    future_joint_vel: torch.Tensor,
    robot_root_quat_wxyz: torch.Tensor,
    future_root_quat_wxyz: torch.Tensor,
    orientation_noise: float = 0.0,
) -> torch.Tensor:
    """Build the released SONIC G1 encoder input.

    Shapes with release defaults:
      future_joint_pos  [B,10,29]
      future_joint_vel  [B,10,29]
      relative root 6D  [B,10,6]
      flattened result  [B,640]
    """
    b, f, dof = future_joint_pos.shape
    if future_joint_vel.shape != (b, f, dof):
        raise ValueError("future_joint_vel shape mismatch")
    robot_q = robot_root_quat_wxyz[:, None, :].expand(-1, f, -1)
    root6 = relative_rotation_6d(robot_q, future_root_quat_wxyz)
    if orientation_noise > 0:
        root6 = root6 + torch.empty_like(root6).uniform_(-orientation_noise, orientation_noise)
    return torch.cat([future_joint_pos, future_joint_vel, root6], dim=-1).reshape(b, -1)


def dynamic_decoder_observation(token_flat: torch.Tensor, proprio_history: torch.Tensor) -> torch.Tensor:
    return torch.cat([token_flat, proprio_history], dim=-1)
