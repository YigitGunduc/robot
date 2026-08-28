from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SPLIT_NAMES = ("train", "validation", "test")
_MIRROR_TOKEN = re.compile(
    r"(^|[_\-])(mirror(?:ed)?|flip(?:ped)?)(?=[_\-]|$)", re.IGNORECASE
)


def motion_group_key(path_or_name: str | Path) -> str:
    """Return a stable content key that keeps named mirror/flip variants together."""
    stem = Path(path_or_name).stem.lower()
    stem = _MIRROR_TOKEN.sub("_", stem)
    stem = re.sub(r"[_\-]+", "_", stem).strip("_")
    return stem


def deterministic_split(
    group: str,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def build_split_manifest(
    root: str | Path,
    seed: int = 0,
    train_fraction: float = 0.9,
    validation_fraction: float = 0.05,
    group_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    root = Path(root)
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train + validation fractions must leave a non-empty test fraction"
        )
    files = sorted(
        p
        for p in root.rglob("*.npz")
        if p.name not in {"_manifest.npz", "_packed_metadata.npz"}
    )
    if not files:
        raise FileNotFoundError(f"no motion clips under {root}")

    overrides = group_overrides or {}
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    groups: dict[str, str] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        group = overrides.get(path.stem, motion_group_key(rel))
        split = deterministic_split(group, seed, train_fraction, validation_fraction)
        splits[split].append(rel)
        groups[rel] = group

    if any(len(splits[name]) == 0 for name in SPLIT_NAMES):
        raise ValueError(
            "one or more splits are empty; use a larger dataset or adjust seed/fractions: "
            + ", ".join(f"{name}={len(splits[name])}" for name in SPLIT_NAMES)
        )
    return {
        "version": 1,
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(1.0 - train_fraction - validation_fraction),
        "splits": splits,
        "groups": groups,
    }


def save_split_manifest(manifest: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_split_files(
    root: str | Path, manifest_path: str | Path, split: str
) -> list[Path]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("version") != 1:
        raise ValueError("unsupported split manifest version")
    relpaths = manifest["splits"][split]
    files = [root / rel for rel in relpaths]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"split manifest references missing files: {missing[:5]}"
        )
    return files


def validate_split_manifest(manifest: dict[str, object]) -> None:
    splits = manifest["splits"]
    sets = {name: set(splits[name]) for name in SPLIT_NAMES}
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"{left}/{right} split overlap: {sorted(overlap)[:5]}")
    group_owner: dict[str, str] = {}
    groups = manifest["groups"]
    for split, relpaths in splits.items():
        for rel in relpaths:
            group = groups[rel]
            previous = group_owner.setdefault(group, split)
            if previous != split:
                raise ValueError(
                    f"content group {group!r} leaks across {previous}/{split}"
                )
