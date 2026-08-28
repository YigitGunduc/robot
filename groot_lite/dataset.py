from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenTrajectoryDataset(Dataset):
    """Memory-efficient offline BONES->SONIC-token dataset.

    Uses one cumulative count per clip instead of materializing tens of millions of `(file, frame)`
    tuples. Each .npz contains `tokens[T,D]`, `state[T,S]`, and optionally:
      * `captions[K]`: official BONES full-motion paraphrases
      * `timeline_start[E]`, `timeline_end[E]`, `timeline_text[E]`: local segment labels in seconds
      * `text`: legacy/fallback caption
      * `action_mask[T,D]`

    For each action window, local timeline text is preferred. If none overlaps the window center,
    a random official full-motion caption is sampled. Filename-derived text is only a last fallback.
    """
    def __init__(self, root: str | Path, horizon: int = 16, prefer_timeline: bool = True):
        self.files = sorted(p for p in Path(root).rglob("*.npz") if p.name != "_manifest.npz")
        self.horizon = int(horizon)
        self.prefer_timeline = bool(prefer_timeline)
        counts = []
        self.fps = []
        for p in self.files:
            with np.load(p, allow_pickle=False) as d:
                n = int(d["tokens"].shape[0])
                self.fps.append(float(d["fps"].item()) if "fps" in d else 50.0)
            counts.append(max(0, n - self.horizon + 1))
        self.cumulative = np.cumsum(np.asarray(counts, dtype=np.int64))
        self.total = int(self.cumulative[-1]) if len(self.cumulative) else 0
        if self.total == 0:
            raise FileNotFoundError(f"No usable token trajectories in {root}")

    def __len__(self):
        return self.total

    @staticmethod
    def _choose_text(d, frame: int, horizon: int, fps: float, fallback: str) -> str:
        # Label an action window by its temporal center, which aligns short flow-matching chunks with
        # the atomic action actually occurring in that interval.
        center_sec = (frame + 0.5 * (horizon - 1)) / fps
        if "timeline_start" in d and "timeline_end" in d and "timeline_text" in d:
            starts = d["timeline_start"]
            ends = d["timeline_end"]
            hits = np.flatnonzero((starts <= center_sec) & (center_sec < ends))
            if len(hits):
                return str(d["timeline_text"][int(hits[0])])
        if "captions" in d and len(d["captions"]):
            return str(d["captions"][random.randrange(len(d["captions"]))])
        if "text" in d:
            return str(d["text"].item())
        return fallback

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        fi = bisect_right(self.cumulative, idx)
        prev = 0 if fi == 0 else int(self.cumulative[fi - 1])
        t = int(idx - prev)
        with np.load(self.files[fi], allow_pickle=False) as d:
            tokens = torch.from_numpy(d["tokens"][t:t + self.horizon]).float()
            state = torch.from_numpy(d["state"][t]).float()
            if self.prefer_timeline:
                text = self._choose_text(d, t, self.horizon, self.fps[fi], self.files[fi].stem.replace("_", " "))
            elif "captions" in d and len(d["captions"]):
                text = str(d["captions"][random.randrange(len(d["captions"]))])
            elif "text" in d:
                text = str(d["text"].item())
            else:
                text = self.files[fi].stem.replace("_", " ")
            if "action_mask" in d:
                mask = torch.from_numpy(d["action_mask"][t:t + self.horizon]).bool()
            else:
                mask = torch.ones_like(tokens, dtype=torch.bool)
        return {"text": text, "state": state, "actions": tokens, "action_mask": mask}
