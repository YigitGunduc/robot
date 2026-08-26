from __future__ import annotations

import json
import os
import tempfile
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
        with tempfile.NamedTemporaryFile(dir=self.episodes, suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            np.savez_compressed(tmp_path, **arrays)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        row = {"episode_id": episode_id, "file": str(path.relative_to(self.root)), **metadata}
        existing = []
        if self.meta_path.exists():
            with self.meta_path.open(encoding="utf-8") as f:
                existing = [json.loads(line) for line in f if line.strip()]
        existing = [item for item in existing if item.get("episode_id") != episode_id]
        existing.append(row)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.root,
            suffix=".jsonl",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            meta_tmp = Path(tmp.name)
            for item in existing:
                tmp.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(meta_tmp, self.meta_path)
        return path
