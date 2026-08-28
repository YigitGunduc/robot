from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


FULL_CAPTION_COLUMNS = (
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
)


@dataclass(frozen=True)
class TimelineEvent:
    start_time: float
    end_time: float
    description: str


@dataclass(frozen=True)
class MotionAnnotations:
    filename: str
    captions: tuple[str, ...] = ()
    overview: str | None = None
    events: tuple[TimelineEvent, ...] = ()
    category: str | None = None
    package: str | None = None


def _clean_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def load_seed_metadata(path: str | Path) -> dict[str, MotionAnnotations]:
    """Load BONES-SEED metadata parquet/csv keyed by `filename`.

    The official metadata has up to seven useful whole-motion language fields. We retain all
    non-empty unique variants so GR00T-Lite can sample paraphrases during training instead of
    learning filename-derived labels.
    """
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported metadata format: {path}")
    if "filename" not in df.columns:
        raise ValueError("BONES metadata must contain a `filename` column")

    out: dict[str, MotionAnnotations] = {}
    for row in df.to_dict(orient="records"):
        filename = str(row["filename"])
        seen: set[str] = set()
        captions: list[str] = []
        for col in FULL_CAPTION_COLUMNS:
            if col not in row:
                continue
            text = _clean_text(row.get(col))
            if text and text not in seen:
                captions.append(text)
                seen.add(text)
        out[filename] = MotionAnnotations(
            filename=filename,
            captions=tuple(captions),
            category=_clean_text(row.get("category")),
            package=_clean_text(row.get("package")),
        )
    return out


def load_seed_timelines(path: str | Path) -> dict[str, tuple[str | None, tuple[TimelineEvent, ...]]]:
    """Load NVIDIA/BONES timeline JSONL keyed by motion filename."""
    path = Path(path)
    out: dict[str, tuple[str | None, tuple[TimelineEvent, ...]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            filename = str(row["filename"])
            events = []
            for event in row.get("events", []) or []:
                desc = str(event.get("description", "")).strip()
                if not desc:
                    continue
                events.append(TimelineEvent(
                    start_time=float(event["start_time"]),
                    end_time=float(event["end_time"]),
                    description=desc,
                ))
            overview = str(row.get("overview_description", "")).strip() or None
            out[filename] = (overview, tuple(events))
    return out


def merge_annotations(
    metadata: dict[str, MotionAnnotations] | None,
    timelines: dict[str, tuple[str | None, tuple[TimelineEvent, ...]]] | None,
) -> dict[str, MotionAnnotations]:
    metadata = metadata or {}
    timelines = timelines or {}
    keys = set(metadata) | set(timelines)
    out: dict[str, MotionAnnotations] = {}
    for key in keys:
        base = metadata.get(key, MotionAnnotations(filename=key))
        overview, events = timelines.get(key, (None, ()))
        captions = list(base.captions)
        if overview and overview not in captions:
            captions.append(overview)
        out[key] = MotionAnnotations(
            filename=key,
            captions=tuple(captions),
            overview=overview,
            events=events,
            category=base.category,
            package=base.package,
        )
    return out
