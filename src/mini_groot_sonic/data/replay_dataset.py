from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mini_groot_sonic.checkpoint import BODY_CONTROL_STACK_VERSION


class ReplayWindowDataset(Dataset):
    """Samples GR00T-style future token chunks from collected replay episodes."""

    def __init__(
        self,
        root: str | Path,
        horizon: int = 40,
        samples_per_episode: int = 16,
        goal_probabilities: tuple[float, float, float, float] = (0.55, 0.20, 0.20, 0.05),
        split: str = "all",
        validation_fraction: float = 0.1,
    ):
        self.root = Path(root)
        self.horizon = horizon
        self.samples_per_episode = samples_per_episode
        self.goal_probs = goal_probabilities
        with (self.root / "episodes.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        versions = {int(row.get("body_control_stack_version", 0)) for row in rows}
        if versions != {BODY_CONTROL_STACK_VERSION}:
            raise RuntimeError(
                f"Replay data targets body control stack versions {sorted(versions)}, "
                f"but v{BODY_CONTROL_STACK_VERSION} is required. Recollect replay data."
            )
        body_fingerprints = {row.get("body_policy_fingerprint") for row in rows}
        if len(body_fingerprints) != 1 or None in body_fingerprints:
            raise RuntimeError(
                "Replay data must come from one exact trained body checkpoint; "
                "reference-PD or mixed-policy replay cannot train the flow model."
            )
        self.body_policy_fingerprint = str(next(iter(body_fingerprints)))
        if split not in {"all", "train", "val"}:
            raise ValueError("split must be 'all', 'train', or 'val'")
        if split == "all" or len(rows) < 2:
            self.rows = rows
        else:
            def group_key(row: dict) -> str:
                return str(row.get("actor_uid") or row.get("source_motion_id") or row["episode_id"])

            groups = sorted(
                {group_key(row) for row in rows},
                key=lambda key: hashlib.sha256(key.encode()).digest(),
            )
            if len(groups) < 2:
                groups = [str(row["episode_id"]) for row in rows]
                group_key = lambda row: str(row["episode_id"])
            n_val = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
            validation_groups = set(groups[:n_val])
            self.rows = [
                r for r in rows
                if (group_key(r) in validation_groups) == (split == "val")
            ]
        if not self.rows:
            raise ValueError("Replay dataset is empty")

    def __len__(self) -> int:
        return len(self.rows) * self.samples_per_episode

    def _sample_goal_mask(self, slots: int) -> np.ndarray:
        p = random.random()
        cuts = np.cumsum(self.goal_probs)
        mask = np.zeros(slots, np.float32)
        if p < cuts[0]:
            return mask
        if p < cuts[1]:
            mask[0] = 1.0  # root/path-style condition
        elif p < cuts[2]:
            mask[random.randrange(1, slots)] = 1.0
        else:
            k = min(slots, random.randint(2, 3))
            ix = random.sample(range(slots), k=k)
            mask[ix] = 1.0
        return mask

    def __getitem__(self, idx: int):
        row = self.rows[idx // self.samples_per_episode]
        with np.load(self.root / row["file"], allow_pickle=False) as d:
            t = len(d["token"])
            start = 0 if t <= self.horizon else random.randint(0, t - self.horizon)
            end = min(t, start + self.horizon)
            token = np.zeros((self.horizon, d["token"].shape[-1]), np.float32)
            valid = np.zeros(self.horizon, np.float32)
            token[: end - start] = d["token"][start:end]
            valid[: end - start] = 1
            if end - start < self.horizon:
                token[end - start :] = token[max(0, end - start - 1)]

            from scipy.spatial.transform import Rotation

            current_root_pos = d["root_pos"][start].astype(np.float32)
            current_root_q_wxyz = d["root_quat"][start].astype(np.float32)
            current_rot = Rotation.from_quat(current_root_q_wxyz[[1, 2, 3, 0]])
            gravity = current_rot.inv().apply(np.asarray([0.0, 0.0, -1.0], np.float32))
            local_linvel = current_rot.inv().apply(d["root_linvel"][start]).astype(np.float32)
            # MuJoCo free-joint angular velocity is already root-local.
            local_angvel = d["root_angvel"][start].astype(np.float32)
            state = np.concatenate(
                [
                    d["q"][start],
                    d["qdot"][start],
                    gravity,
                    local_linvel,
                    local_angvel,
                    d["root_pos"][start, 2:3],
                ]
            ).astype(np.float32)

            goal_slots = d["goal_slots"][min(end - 1, t - 1)].astype(np.float32)
            rel_pos = current_rot.inv().apply(goal_slots[:, :3] - current_root_pos).astype(np.float32)
            slot_xyzw = goal_slots[:, [4, 5, 6, 3]]
            rel_rot = current_rot.inv() * Rotation.from_quat(slot_xyzw)
            rel_xyzw = rel_rot.as_quat().astype(np.float32)
            rel_wxyz = rel_xyzw[:, [3, 0, 1, 2]]
            goal_slots = np.concatenate([rel_pos, rel_wxyz], axis=-1)
            mask = self._sample_goal_mask(goal_slots.shape[0])
            goal_slots *= mask[:, None]  # inactive slots must not leak future state
            goal = np.concatenate([goal_slots, mask[:, None]], axis=-1).reshape(-1).astype(np.float32)

            sample = {
                "caption": random.choice(row.get("captions", [row["caption"]])),
                "state": torch.from_numpy(state.copy()),
                "actions": torch.from_numpy(token),
                "valid": torch.from_numpy(valid),
                "goal": torch.from_numpy(goal),
            }
            if "rgb" in d.files and len(d["rgb"]):
                steps = d["rgb_step"]
                j = int(np.argmin(np.abs(steps - start)))
                sample["rgb"] = torch.from_numpy(d["rgb"][j].copy())
            return sample

    def normalization_stats(self, max_samples: int = 4096) -> dict[str, torch.Tensor]:
        count = min(len(self), max_samples)
        states, goals = [], []
        rng_state = random.getstate()
        random.seed(0)
        try:
            for i in range(count):
                sample = self[i]
                states.append(sample["state"])
                goals.append(sample["goal"])
        finally:
            random.setstate(rng_state)
        state = torch.stack(states)
        goal = torch.stack(goals)
        goal_mean = goal.mean(0)
        goal_std = goal.std(0).clamp_min(1e-4)
        # Mask bits retain their exact 0/1 semantics.
        if goal.shape[1] % 8 == 0:
            goal_mean[7::8] = 0
            goal_std[7::8] = 1
        return {
            "state_mean": state.mean(0),
            "state_std": state.std(0).clamp_min(1e-4),
            "goal_mean": goal_mean,
            "goal_std": goal_std,
        }
