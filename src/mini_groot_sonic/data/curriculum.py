from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

STAGE_NAMES = (
    "balance",
    "neutral_walk",
    "walk_variations",
    "turns",
    "jog_run",
)


@dataclass(frozen=True)
class MotionCurriculumRecord:
    filename: str
    motion_id: str
    semantic_stage: int | None
    semantic_reason: str
    quality_passed: bool
    quality_reason: str
    duration_seconds: float
    mean_horizontal_speed: float
    p95_horizontal_speed: float
    p95_yaw_rate: float
    total_abs_yaw: float
    pelvis_height_range: float
    p95_joint_speed: float
    p95_upper_body_speed: float
    max_body_speed: float
    max_joint_speed: float
    min_body_height: float
    difficulty_score: float = math.nan
    first_admitted_stage: int | None = None


@dataclass(frozen=True)
class CurriculumStage:
    index: int
    name: str
    target_size: int
    filenames: tuple[str, ...]
    new_filenames: tuple[str, ...]


def _scalar(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return value.item() if value.size == 1 else value.tolist()
    return value


def _text(value: object) -> str:
    value = _scalar(value)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _false_like(value: object) -> bool:
    return _text(value).lower() in {"", "0", "0.0", "false", "no", "none", "null", "nan"}


def _true_like(value: object) -> bool:
    return _text(value).lower() in {"1", "1.0", "true", "yes"}


def structured_metadata_from_npz(data: Mapping[str, object]) -> dict[str, object]:
    return {key: _scalar(data[key]) for key in data if key.startswith("content_") or key in {
        "package",
        "category",
        "is_neutral",
        "is_mirror",
        "move_name",
        "filename",
    }}


def classify_structured_motion(
    metadata: Mapping[str, object],
    motion_id: str,
) -> tuple[int | None, str]:
    """Assign a semantic stage without searching free-form descriptions."""

    package = _text(metadata.get("package", "")).lower()
    category = _text(metadata.get("category", "")).lower()
    movement = _text(metadata.get("content_type_of_movement", "")).lower()
    body_position = _text(metadata.get("content_body_position", "")).lower()
    identifier = motion_id.lower()

    if _true_like(metadata.get("is_mirror", "")):
        return None, "mirrored duplicate"
    if not _false_like(metadata.get("content_props", "")):
        return None, "uses props"
    if _true_like(metadata.get("content_complex_action", "")):
        return None, "complex multi-phase action"
    if package and package != "locomotion":
        return None, f"package is {package!r}, not locomotion"

    structured = f"{category} {movement} {body_position}"
    filename_signal = identifier
    if any(token in structured or token in filename_signal for token in ("jog", "run", "sprint")):
        return 4, "jog/run locomotion"
    if any(token in structured or token in filename_signal for token in ("turn", "pivot", "rotate")):
        return 3, "turning locomotion"
    if "walk" in structured or "walk" in filename_signal:
        neutral = _true_like(metadata.get("is_neutral", ""))
        styled = "style" in category or (metadata.get("is_neutral") is not None and not neutral)
        return (2, "styled walking") if styled else (1, "neutral walking")
    if any(token in structured or token in filename_signal for token in ("stand", "idle", "baseline")):
        return 0, "standing/baseline"
    return None, "not a supported locomotion curriculum motion"


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def motion_kinematic_features(data: Mapping[str, object]) -> dict[str, float | bool | str]:
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_linvel = np.asarray(data["root_linvel"], dtype=np.float64)
    root_angvel = np.asarray(data["root_angvel"], dtype=np.float64)
    body_pos = np.asarray(data["body_pos"], dtype=np.float64)
    body_linvel = np.asarray(data["body_linvel"], dtype=np.float64)
    body_names = [str(value) for value in np.asarray(data["body_names"]).tolist()]
    fps = float(_scalar(data["fps"]))

    arrays = (joint_vel, root_pos, root_linvel, root_angvel, body_pos, body_linvel)
    if fps <= 0 or not all(np.isfinite(values).all() for values in arrays):
        return {"quality_passed": False, "quality_reason": "non-finite values or invalid fps"}

    horizontal_speed = np.linalg.norm(root_linvel[:, :2], axis=-1)
    yaw_rate = np.abs(root_angvel[:, 2])
    joint_speed = np.abs(joint_vel)
    body_speed = np.linalg.norm(body_linvel, axis=-1)
    upper_indices = [
        index
        for index, name in enumerate(body_names)
        if any(part in name.lower() for part in ("torso", "shoulder", "elbow", "wrist"))
    ]
    upper_speed = body_speed[:, upper_indices] if upper_indices else body_speed
    min_body_height = float(body_pos[..., 2].min())
    max_body_speed = float(body_speed.max())
    max_joint_speed = float(joint_speed.max())

    floor_estimate = float(body_pos[..., 2].min())
    lowest_per_frame = body_pos[..., 2].min(axis=1)
    airborne = lowest_per_frame - floor_estimate > 0.2
    excessive_airborne = _longest_true_run(airborne) >= max(1, round(0.6 * fps))
    failures = []
    if min_body_height < -0.05:
        failures.append("body below floor")
    if max_body_speed > 15.0:
        failures.append("body velocity above 15 m/s")
    if max_joint_speed > 40.0:
        failures.append("joint velocity above 40 rad/s")
    if excessive_airborne:
        failures.append("extended airborne motion")

    return {
        "quality_passed": not failures,
        "quality_reason": "; ".join(failures) if failures else "passed",
        "duration_seconds": len(root_pos) / fps,
        "mean_horizontal_speed": float(horizontal_speed.mean()),
        "p95_horizontal_speed": float(np.quantile(horizontal_speed, 0.95)),
        "p95_yaw_rate": float(np.quantile(yaw_rate, 0.95)),
        "total_abs_yaw": float(yaw_rate.sum() / fps),
        "pelvis_height_range": float(np.ptp(root_pos[:, 2])),
        "p95_joint_speed": float(np.quantile(joint_speed, 0.95)),
        "p95_upper_body_speed": float(np.quantile(upper_speed, 0.95)),
        "max_body_speed": max_body_speed,
        "max_joint_speed": max_joint_speed,
        "min_body_height": min_body_height,
    }


def analyze_motion(path: str | Path) -> MotionCurriculumRecord:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        motion_id = _text(data["motion_id"]) if "motion_id" in data.files else path.stem
        metadata = structured_metadata_from_npz(data)
        semantic_stage, semantic_reason = classify_structured_motion(metadata, motion_id)
        features = motion_kinematic_features(data)
    defaults = {
        "duration_seconds": math.nan,
        "mean_horizontal_speed": math.nan,
        "p95_horizontal_speed": math.nan,
        "p95_yaw_rate": math.nan,
        "total_abs_yaw": math.nan,
        "pelvis_height_range": math.nan,
        "p95_joint_speed": math.nan,
        "p95_upper_body_speed": math.nan,
        "max_body_speed": math.nan,
        "max_joint_speed": math.nan,
        "min_body_height": math.nan,
    }
    defaults.update(features)
    return MotionCurriculumRecord(
        filename=path.name,
        motion_id=motion_id,
        semantic_stage=semantic_stage,
        semantic_reason=semantic_reason,
        quality_passed=bool(defaults.pop("quality_passed")),
        quality_reason=str(defaults.pop("quality_reason")),
        **defaults,
    )


def _percentile_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    average_ranks = starts + (counts - 1) / 2.0
    return average_ranks[inverse] / max(1, len(array) - 1)


def score_motion_difficulty(records: Sequence[MotionCurriculumRecord]) -> list[MotionCurriculumRecord]:
    eligible_indices = [
        index
        for index, record in enumerate(records)
        if record.semantic_stage is not None and record.quality_passed
    ]
    if not eligible_indices:
        return list(records)
    metrics = (
        ("p95_horizontal_speed", 0.30),
        ("p95_yaw_rate", 0.25),
        ("p95_joint_speed", 0.20),
        ("pelvis_height_range", 0.15),
        ("p95_upper_body_speed", 0.10),
    )
    scores = np.zeros(len(eligible_indices), dtype=np.float64)
    for name, weight in metrics:
        scores += weight * _percentile_ranks(
            [float(getattr(records[index], name)) for index in eligible_indices]
        )
    replacements = {index: scores[offset] for offset, index in enumerate(eligible_indices)}
    return [
        MotionCurriculumRecord(**{**asdict(record), "difficulty_score": float(replacements[index])})
        if index in replacements
        else record
        for index, record in enumerate(records)
    ]


def build_curriculum(
    motion_paths: Sequence[str | Path],
    stage_sizes: Sequence[int],
    *,
    seed: int = 0,
) -> tuple[list[CurriculumStage], list[MotionCurriculumRecord]]:
    if len(stage_sizes) != len(STAGE_NAMES):
        raise ValueError(f"Expected {len(STAGE_NAMES)} cumulative stage sizes")
    if any(size <= 0 for size in stage_sizes) or list(stage_sizes) != sorted(stage_sizes):
        raise ValueError("Stage sizes must be positive and non-decreasing")

    records = score_motion_difficulty([analyze_motion(path) for path in motion_paths])
    rng = random.Random(seed)
    tie_breakers = {record.filename: rng.random() for record in records}
    stages: list[CurriculumStage] = []
    previous: tuple[str, ...] = ()
    admitted_at: dict[str, int] = {}
    for stage_index, (name, target_size) in enumerate(zip(STAGE_NAMES, stage_sizes, strict=True)):
        candidates = [
            record
            for record in records
            if record.quality_passed
            and record.semantic_stage is not None
            and record.semantic_stage <= stage_index
        ]
        candidates.sort(
            key=lambda record: (
                0 if record.semantic_stage == stage_index else 1,
                record.difficulty_score,
                tie_breakers[record.filename],
                record.filename,
            )
        )
        selected_list = list(previous)
        selected_set = set(previous)
        for record in candidates:
            if len(selected_list) >= target_size:
                break
            if record.filename not in selected_set:
                selected_list.append(record.filename)
                selected_set.add(record.filename)
        selected = tuple(selected_list)
        if not selected:
            raise ValueError(f"No eligible motions for curriculum stage {stage_index} ({name})")
        previous_set = set(previous)
        new = tuple(filename for filename in selected if filename not in previous_set)
        for filename in new:
            admitted_at.setdefault(filename, stage_index)
        stages.append(CurriculumStage(stage_index, name, target_size, selected, new))
        previous = selected

    audited = [
        MotionCurriculumRecord(
            **{**asdict(record), "first_admitted_stage": admitted_at.get(record.filename)}
        )
        for record in records
    ]
    return stages, audited


def write_curriculum_artifacts(
    output_dir: str | Path,
    motion_dir: str | Path,
    stages: Sequence[CurriculumStage],
    records: Sequence[MotionCurriculumRecord],
    *,
    seed: int,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "curriculum.json"
    manifest = {
        "version": 1,
        "motion_dir": str(Path(motion_dir)),
        "seed": seed,
        "stages": [asdict(stage) for stage in stages],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for stage in stages:
        (output_dir / f"stage_{stage.index}_{stage.name}.txt").write_text(
            "\n".join(stage.filenames) + "\n",
            encoding="utf-8",
        )
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with (output_dir / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return manifest_path


def load_curriculum_manifest(path: str | Path) -> tuple[Path, list[CurriculumStage]]:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    motion_dir = Path(raw["motion_dir"])
    stages = [
        CurriculumStage(
            index=int(item["index"]),
            name=str(item["name"]),
            target_size=int(item["target_size"]),
            filenames=tuple(item["filenames"]),
            new_filenames=tuple(item["new_filenames"]),
        )
        for item in raw["stages"]
    ]
    return motion_dir, stages
