from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EpisodeWriter:
    """Simple append-only replay dataset writer.

    Numeric arrays live in compressed NPZ files; captions and schema metadata live
    in JSONL. This avoids a hard LeRobot/pyarrow dependency while preserving a
    straightforward conversion path later.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.episodes = self.root / "episodes"
        self.episodes.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "episodes.jsonl"

    def write(self, episode_id: str, arrays: dict[str, np.ndarray], metadata: dict) -> Path:
        path = self.episodes / f"{episode_id}.npz"
        np.savez_compressed(path, **arrays)
        row = {"episode_id": episode_id, "file": str(path.relative_to(self.root)), **metadata}
        with self.meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path
