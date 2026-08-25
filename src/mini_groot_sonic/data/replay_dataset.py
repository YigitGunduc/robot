from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ReplayWindowDataset(Dataset):
    """Samples GR00T-style future token chunks from collected replay episodes."""

    def __init__(
        self,
        root: str | Path,
        horizon: int = 40,
        samples_per_episode: int = 16,
        goal_probabilities: tuple[float, float, float, float] = (0.55, 0.20, 0.20, 0.05),
    ):
        self.root = Path(root)
        self.horizon = horizon
        self.samples_per_episode = samples_per_episode
        self.goal_probs = goal_probabilities
        with (self.root / "episodes.jsonl").open(encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
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
        d = np.load(self.root / row["file"], allow_pickle=False)
        t = len(d["token"])
        if t <= self.horizon:
            start = 0
        else:
            start = random.randint(0, t - self.horizon)
        end = min(t, start + self.horizon)
        token = np.zeros((self.horizon, d["token"].shape[-1]), np.float32)
        valid = np.zeros(self.horizon, np.float32)
        token[: end - start] = d["token"][start:end]
        valid[: end - start] = 1
        if end - start < self.horizon:
            token[end - start :] = token[max(0, end - start - 1)]

        state = np.concatenate(
            [
                d["q"][start],
                d["qdot"][start],
                d["root_quat"][start],
                d["root_linvel"][start],
                d["root_angvel"][start],
            ]
        ).astype(np.float32)

        goal_slots = d["goal_slots"][min(end - 1, t - 1)].astype(np.float32)
        # Convert world SE(3) targets into the current root frame. Missing targets then
        # have a location-independent meaning at deployment.
        from scipy.spatial.transform import Rotation
        current_root_pos = d["root_pos"][start].astype(np.float32)
        current_root_q_wxyz = d["root_quat"][start].astype(np.float32)
        current_rot = Rotation.from_quat(current_root_q_wxyz[[1, 2, 3, 0]])
        rel_pos = current_rot.inv().apply(goal_slots[:, :3] - current_root_pos).astype(np.float32)
        slot_xyzw = goal_slots[:, [4, 5, 6, 3]]
        rel_rot = current_rot.inv() * Rotation.from_quat(slot_xyzw)
        rel_xyzw = rel_rot.as_quat().astype(np.float32)
        rel_wxyz = rel_xyzw[:, [3, 0, 1, 2]]
        goal_slots = np.concatenate([rel_pos, rel_wxyz], axis=-1)
        mask = self._sample_goal_mask(goal_slots.shape[0])
        goal = np.concatenate([goal_slots, mask[:, None]], axis=-1).reshape(-1).astype(np.float32)

        sample = {
            "caption": random.choice(row.get("captions", [row["caption"]])),
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(token),
            "valid": torch.from_numpy(valid),
            "goal": torch.from_numpy(goal),
        }
        if "rgb" in d.files and len(d["rgb"]):
            steps = d["rgb_step"]
            j = int(np.argmin(np.abs(steps - start)))
            sample["rgb"] = torch.from_numpy(d["rgb"][j])
        return sample
