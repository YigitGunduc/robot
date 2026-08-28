from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints may be loaded with map_location="cuda", which also moves the
    # CPU generator state tensor to CUDA. PyTorch's CPU generator only accepts a
    # CPU ByteTensor, and CUDA generator states are likewise safest to restore
    # from their serialized CPU representation.
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([generator_state.cpu() for generator_state in state["cuda"]])


def save_config_snapshot(path: str | Path, **configs) -> None:
    serializable = {name: asdict(cfg) for name, cfg in configs.items()}
    for values in serializable.values():
        for key, value in list(values.items()):
            if isinstance(value, Path):
                values[key] = str(value)
    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def split_motion_paths(
    paths: list[Path],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Prefer actor-disjoint splits, then source-motion-disjoint splits."""
    group_candidates = {path: _motion_group_candidates(path) for path in paths}
    groups: dict[str, list[Path]] = defaultdict(list)
    for candidate_index in range(3):
        groups = defaultdict(list)
        for path in paths:
            groups[group_candidates[path][candidate_index]].append(path)
        if len(groups) >= 2:
            break
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        shuffled = paths.copy()
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * validation_fraction)) if len(shuffled) > 1 else 0
        return shuffled[n_val:] or shuffled, shuffled[:n_val]
    n_val = min(len(keys) - 1, max(1, round(len(keys) * validation_fraction)))
    validation_keys = set(keys[:n_val])
    training = [path for key, members in groups.items() if key not in validation_keys for path in members]
    validation = [path for key, members in groups.items() if key in validation_keys for path in members]
    return training, validation


def motion_group_key(path: str | Path) -> str:
    return _motion_group_candidates(Path(path))[0]


def _motion_group_candidates(path: Path) -> tuple[str, str, str]:
    clip_key = f"clip:{path.stem}"
    actor_key: str | None = None
    source_key: str | None = None
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        if "actor_uid" in data.files:
            actor = str(data["actor_uid"].item()).strip()
            if actor and actor.lower() not in {"nan", "none", "null"}:
                actor_key = f"actor:{actor}"
        if "source_motion_id" in data.files:
            source = str(data["source_motion_id"].item()).strip()
            if source and source.lower() not in {"nan", "none", "null"}:
                source_key = f"source:{source}"
    source_key = source_key or clip_key
    return actor_key or source_key, source_key, clip_key


def split_curriculum_motion_paths(
    stage_paths: list[list[Path]],
    validation_fraction: float,
    seed: int,
) -> list[tuple[list[Path], list[Path]]]:
    """Create one reserved validation-group set that remains stable across stages."""

    if not stage_paths or not stage_paths[-1]:
        raise ValueError("Curriculum requires at least one non-empty stage")
    unique_paths = list(dict.fromkeys(path for paths in stage_paths for path in paths))
    group_candidates = {path: _motion_group_candidates(path) for path in unique_paths}
    key_by_path: dict[Path, str] | None = None
    for candidate_index in range(3):
        candidate_keys = {
            path: group_candidates[path][candidate_index] for path in unique_paths
        }
        if all(len({candidate_keys[path] for path in paths}) >= 2 for paths in stage_paths):
            key_by_path = candidate_keys
            break
    if key_by_path is None:
        raise ValueError(
            "Every curriculum stage needs at least two motions for a stable "
            "train/validation split"
        )
    final_keys = sorted({key_by_path[path] for path in stage_paths[-1]})
    rng = random.Random(seed)
    rng.shuffle(final_keys)
    n_validation = min(
        len(final_keys) - 1,
        max(1, round(len(final_keys) * validation_fraction)),
    )
    validation_keys = set(final_keys[:n_validation])
    for stage_index, paths in enumerate(stage_paths):
        stage_keys = {key_by_path[path] for path in paths}
        if not (stage_keys & validation_keys):
            candidates = sorted(stage_keys - validation_keys)
            random.Random(seed + stage_index + 1).shuffle(candidates)
            if len(stage_keys) > 1 and candidates:
                validation_keys.add(candidates[0])

    splits = []
    for paths in stage_paths:
        training = [path for path in paths if key_by_path[path] not in validation_keys]
        validation = [path for path in paths if key_by_path[path] in validation_keys]
        if not training or not validation:
            raise ValueError(
                "Every curriculum stage needs at least one disjoint training and validation group"
            )
        splits.append((training, validation))
    return splits
