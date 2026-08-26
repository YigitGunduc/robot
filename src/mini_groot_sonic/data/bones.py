from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

DESCRIPTION_COLUMNS = (
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
)


@dataclass
class BonesClip:
    motion_id: str
    caption: str
    fps: float
    joint_pos: np.ndarray  # [T,29], radians
    joint_vel: np.ndarray  # [T,29], rad/s
    root_pos: np.ndarray   # [T,3], meters
    root_quat: np.ndarray  # [T,4], wxyz

    @property
    def length(self) -> int:
        return int(self.joint_pos.shape[0])


class BonesSeedIndex:
    """Index official BONES-SEED metadata and G1 CSV files.

    Expected official layout:
      root/metadata/seed_metadata_v00*.{parquet,csv}
      root/g1/csv/<date>/<motion>.csv

    The metadata identifier column has changed across versions, so matching is
    auto-detected by scoring string columns against actual G1 CSV stems.
    """

    def __init__(self, root: str | Path, metadata_path: str | Path | None = None):
        self.root = Path(root)
        self.g1_root = self.root / "g1" / "csv"
        if not self.g1_root.exists():
            raise FileNotFoundError(f"BONES G1 directory not found: {self.g1_root}")
        self.files = sorted(self.g1_root.rglob("*.csv"))
        if not self.files:
            raise FileNotFoundError(f"No BONES G1 CSVs found under {self.g1_root}")
        self.by_stem = {p.stem: p for p in self.files}

        if metadata_path is None:
            candidates = sorted((self.root / "metadata").glob("seed_metadata_v*.parquet"))
            if not candidates:
                candidates = sorted((self.root / "metadata").glob("seed_metadata_v*.csv"))
            if not candidates:
                raise FileNotFoundError("Could not find BONES metadata parquet/csv")
            metadata_path = candidates[-1]
        metadata_path = Path(metadata_path)
        if metadata_path.suffix == ".parquet":
            try:
                self.meta = pd.read_parquet(metadata_path)
            except Exception:
                csv_fallback = metadata_path.with_suffix(".csv")
                if not csv_fallback.exists():
                    raise
                self.meta = pd.read_csv(csv_fallback, low_memory=False)
        else:
            self.meta = pd.read_csv(metadata_path, low_memory=False)

        self.id_column = self._detect_id_column()
        self._row_by_stem: dict[str, int] = {}
        for i, v in self.meta[self.id_column].items():
            stem = self._normalize_id(v)
            if stem in self.by_stem and stem not in self._row_by_stem:
                self._row_by_stem[stem] = int(i)
        self.temporal_labels: dict[str, list[dict]] = {}
        temporal_candidates = sorted(
            (self.root / "metadata").glob("seed_metadata_*_temporal_labels.jsonl")
        )
        if temporal_candidates:
            with temporal_candidates[-1].open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = self._normalize_id(item.get("filename", ""))
                    events = item.get("events", [])
                    if key and isinstance(events, list):
                        self.temporal_labels[key] = events

    @staticmethod
    def _normalize_id(v) -> str:
        s = str(v)
        return Path(s).stem

    def _detect_id_column(self) -> str:
        stems = set(self.by_stem.keys())
        best_col = None
        best_score = -1
        for col in self.meta.columns:
            if self.meta[col].dtype.kind not in "OUS":
                continue
            sample = self.meta[col].dropna().astype(str).head(5000)
            score = sum(Path(x).stem in stems for x in sample)
            if score > best_score:
                best_score, best_col = score, col
        if best_col is None or best_score <= 0:
            raise RuntimeError(
                "Could not auto-detect BONES motion identifier column. "
                "Inspect metadata and adapt BonesSeedIndex._detect_id_column()."
            )
        return str(best_col)

    def captions_for(self, stem: str, preferred_column: str | None = None) -> list[str]:
        row_idx = self._row_by_stem.get(stem)
        if row_idx is None:
            return [stem.replace("_", " ")]
        row = self.meta.loc[row_idx]
        out = []
        columns = list(DESCRIPTION_COLUMNS)
        if preferred_column in columns:
            columns.remove(preferred_column)
            columns.insert(0, preferred_column)
        for col in columns:
            if col in self.meta.columns and pd.notna(row[col]):
                text = str(row[col]).strip()
                if text and text.lower() != "nan" and text not in out:
                    out.append(text)
        return out or [stem.replace("_", " ")]

    def segments_for(self, stem: str) -> list[dict]:
        return self.temporal_labels.get(stem, [])

    def metadata_for(self, stem: str) -> dict[str, str]:
        row_idx = self._row_by_stem.get(stem)
        if row_idx is None:
            return {}
        row = self.meta.loc[row_idx]
        out = {}
        for key in ("actor_uid", "take_actor", "content_name", "package", "category"):
            if key in self.meta.columns and pd.notna(row[key]):
                out[key] = str(row[key])
        return out

    def iter_records(
        self,
        limit: int | None = None,
        preferred_column: str | None = None,
        seed: int | None = None,
    ) -> Iterable[tuple[str, Path, list[str]]]:
        files = self.files.copy()
        if seed is not None:
            random.Random(seed).shuffle(files)
        for count, p in enumerate(files, start=1):
            yield p.stem, p, self.captions_for(p.stem, preferred_column)
            if limit is not None and count >= limit:
                return


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def _finite_difference(x: np.ndarray, fps: float) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros_like(x)
    dt = 1.0 / fps
    out = np.gradient(x, dt, axis=0, edge_order=1)
    return np.asarray(out, dtype=np.float32)


def load_g1_csv(
    path: str | Path,
    joint_names: list[str],
    caption: str = "",
    source_fps: float = 120.0,
    target_fps: float = 50.0,
) -> BonesClip:
    path = Path(path)
    df = pd.read_csv(path)
    needed_root = [
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ]
    missing = [c for c in needed_root if c not in df.columns]
    if missing:
        raise ValueError(f"Missing BONES root columns in {path}: {missing}")

    joint_cols = []
    for name in joint_names:
        exact = f"{name}_dof"
        if exact in df.columns:
            joint_cols.append(exact)
            continue
        # Gentle normalization for MJCF naming differences.
        aliases = [
            c for c in df.columns
            if c.endswith("_dof") and c[:-4].lower().replace("-", "_") == name.lower().replace("-", "_")
        ]
        if len(aliases) != 1:
            raise ValueError(f"Could not map joint {name!r} to a BONES CSV column in {path.name}")
        joint_cols.append(aliases[0])

    joint_pos = np.deg2rad(df[joint_cols].to_numpy(np.float32))
    root_pos = df[["root_translateX", "root_translateY", "root_translateZ"]].to_numpy(np.float32) / 100.0
    euler_deg = df[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy(np.float32)
    # BONES docs specify extrinsic XYZ Euler degrees. scipy lowercase xyz is extrinsic.
    root_quat = _xyzw_to_wxyz(Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()).astype(np.float32)

    if abs(target_fps - source_fps) > 1e-6:
        duration = (len(df) - 1) / source_fps
        old_t = np.arange(len(df), dtype=np.float64) / source_fps
        new_len = max(2, round(duration * target_fps) + 1)
        new_t = np.arange(new_len, dtype=np.float64) / target_fps
        new_t[-1] = min(new_t[-1], old_t[-1])
        joint_pos = np.stack([np.interp(new_t, old_t, joint_pos[:, j]) for j in range(joint_pos.shape[1])], axis=-1).astype(np.float32)
        root_pos = np.stack([np.interp(new_t, old_t, root_pos[:, j]) for j in range(3)], axis=-1).astype(np.float32)
        rotations = Rotation.from_quat(_wxyz_to_xyzw(root_quat))
        root_quat = _xyzw_to_wxyz(Slerp(old_t, rotations)(new_t).as_quat()).astype(np.float32)

    joint_vel = _finite_difference(joint_pos, target_fps)
    return BonesClip(
        motion_id=path.stem,
        caption=caption,
        fps=target_fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos=root_pos,
        root_quat=root_quat,
    )
