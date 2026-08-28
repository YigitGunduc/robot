from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from mini_groot_sonic.config import SonicTinyConfig
from mini_groot_sonic.data.curriculum import (
    STAGE_NAMES,
    build_curriculum,
    classify_structured_motion,
    load_curriculum_manifest,
    write_curriculum_artifacts,
)
from mini_groot_sonic.data.motion_bank import MotionBank
from mini_groot_sonic.training.curriculum import PromotionCriteria, promotion_decision
from mini_groot_sonic.training.utils import split_curriculum_motion_paths


def _write_motion(
    path: Path,
    motion_id: str,
    movement: str,
    *,
    actor: str,
    speed: float = 0.0,
    yaw_rate: float = 0.0,
    neutral: bool = True,
    joint_speed: float = 0.1,
) -> None:
    frames = 20
    root_pos = np.zeros((frames, 3), np.float32)
    root_pos[:, 0] = np.arange(frames) * speed / 50.0
    root_pos[:, 2] = 0.8
    root_linvel = np.zeros((frames, 3), np.float32)
    root_linvel[:, 0] = speed
    root_angvel = np.zeros((frames, 3), np.float32)
    root_angvel[:, 2] = yaw_rate
    body_pos = np.repeat(root_pos[:, None], 3, axis=1)
    body_pos[:, :, 2] += np.asarray([0.0, 0.2, -0.1], np.float32)
    body_linvel = np.repeat(root_linvel[:, None], 3, axis=1)
    root_quat = np.zeros((frames, 4), np.float32)
    root_quat[:, 0] = 1.0
    body_quat = np.repeat(root_quat[:, None], 3, axis=1)
    np.savez_compressed(
        path,
        motion_id=np.asarray(motion_id, dtype=object),
        source_motion_id=np.asarray(motion_id, dtype=object),
        actor_uid=np.asarray(actor, dtype=object),
        package=np.asarray("Locomotion", dtype=object),
        category=np.asarray(
            "Basic Locomotion Styles" if not neutral else "Basic Locomotion Neutral",
            dtype=object,
        ),
        is_neutral=np.asarray("1" if neutral else "0", dtype=object),
        is_mirror=np.asarray("0", dtype=object),
        content_props=np.asarray("0", dtype=object),
        content_complex_action=np.asarray("0", dtype=object),
        content_type_of_movement=np.asarray(movement, dtype=object),
        content_body_position=np.asarray("standing", dtype=object),
        joint_pos=np.zeros((frames, 29), np.float32),
        joint_vel=np.full((frames, 29), joint_speed, np.float32),
        root_pos=root_pos,
        root_quat=root_quat,
        root_linvel=root_linvel,
        root_angvel=root_angvel,
        body_pos=body_pos,
        body_quat=body_quat,
        body_linvel=body_linvel,
        body_angvel=np.zeros_like(body_linvel),
        body_names=np.asarray(["pelvis", "left_wrist", "left_foot"], dtype=object),
        fps=np.asarray(50.0, np.float32),
        caption=np.asarray(movement, dtype=object),
    )


def test_structured_classifier_rejects_interactions_without_reading_caption():
    stage, reason = classify_structured_motion(
        {
            "package": "Interactions",
            "content_props": "box",
            "content_type_of_movement": "walking",
        },
        "walk_forward",
    )
    assert stage is None
    assert reason in {"uses props", "package is 'interactions', not locomotion"}


def test_curriculum_is_cumulative_and_kinematically_audited(tmp_path: Path):
    specifications = (
        ("stand_idle", "standing", 0.0, 0.0, True, 0.1),
        ("walk_forward", "walking", 0.4, 0.0, True, 0.3),
        ("walk_style", "walking", 0.5, 0.0, False, 0.4),
        ("turn_left", "turning", 0.3, 0.5, True, 0.5),
        ("run_forward", "running", 1.5, 0.0, True, 0.8),
        ("run_broken", "running", 1.5, 0.0, True, 45.0),
    )
    paths = []
    for index, (name, movement, speed, yaw, neutral, joint_speed) in enumerate(specifications):
        path = tmp_path / f"{name}.npz"
        _write_motion(
            path,
            name,
            movement,
            actor=f"actor-{index}",
            speed=speed,
            yaw_rate=yaw,
            neutral=neutral,
            joint_speed=joint_speed,
        )
        paths.append(path)

    stages, records = build_curriculum(paths, (1, 2, 3, 4, 5), seed=7)
    assert tuple(stage.name for stage in stages) == STAGE_NAMES
    assert [len(stage.filenames) for stage in stages] == [1, 2, 3, 4, 5]
    assert all(set(left.filenames) <= set(right.filenames) for left, right in pairwise(stages))
    assert stages[0].filenames == ("stand_idle.npz",)
    assert stages[-1].new_filenames == ("run_forward.npz",)
    broken = next(record for record in records if record.motion_id == "run_broken")
    assert not broken.quality_passed
    assert "joint velocity" in broken.quality_reason
    assert all(
        np.isfinite(record.difficulty_score)
        for record in records
        if record.quality_passed and record.semantic_stage is not None
    )
    manifest_path = write_curriculum_artifacts(
        tmp_path / "curriculum",
        tmp_path,
        stages,
        records,
        seed=7,
    )
    motion_dir, loaded_stages = load_curriculum_manifest(manifest_path)
    assert motion_dir == tmp_path
    assert loaded_stages == stages
    assert (manifest_path.parent / "audit.csv").exists()


def test_promotion_requires_all_metrics():
    criteria = PromotionCriteria()
    metrics = {
        "success_rate": 0.9,
        "mpjpe": 0.05,
        "root_position_error": 0.04,
        "root_orientation_error": 0.1,
        "evaluated_motions": 2.0,
    }
    assert promotion_decision(metrics, criteria) == (True, [])
    metrics["success_rate"] = 0.5
    passed, failures = promotion_decision(metrics, criteria)
    assert not passed
    assert any("success_rate" in failure for failure in failures)


def test_curriculum_split_reserves_groups_across_all_stages(tmp_path: Path):
    paths = []
    for index in range(6):
        path = tmp_path / f"walk_{index}.npz"
        _write_motion(path, path.stem, "walking", actor=f"actor-{index}")
        paths.append(path)
    splits = split_curriculum_motion_paths(
        [paths[:3], paths[:4], paths],
        validation_fraction=0.25,
        seed=3,
    )
    validation_names = {
        path.name
        for _, validation in splits
        for path in validation
    }
    assert all(training and validation for training, validation in splits)
    assert all(
        path.name not in validation_names
        for training, _ in splits
        for path in training
    )


def test_curriculum_split_falls_back_to_source_when_first_stage_has_one_actor(
    tmp_path: Path,
):
    paths = []
    for index in range(5):
        path = tmp_path / f"stand_{index}.npz"
        _write_motion(path, path.stem, "standing", actor="shared-actor")
        paths.append(path)

    splits = split_curriculum_motion_paths(
        [paths[:2], paths[:3], paths],
        validation_fraction=0.25,
        seed=3,
    )
    validation_names = {
        path.name
        for _, validation in splits
        for path in validation
    }
    assert all(training and validation for training, validation in splits)
    assert all(
        path.name not in validation_names
        for training, _ in splits
        for path in training
    )


def test_failure_sampling_blends_uniform_coverage_with_failed_motions():
    bank = MotionBank.__new__(MotionBank)
    bank.base_sampling_weights = torch.ones(3)
    bank.failure_ema = torch.tensor([0.0, 0.5, 0.0])
    bank.failure_sampling_alpha = 0.5
    bank.failure_sampling_cap = 4.0
    weights = bank.sampling_weights()
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    assert weights[1] > weights[0]
    assert weights[0] > 0 and weights[2] > 0


def test_failure_sampling_targets_one_second_segments(tmp_path: Path):
    path = tmp_path / "long_stand.npz"
    _write_motion(path, "long_stand", "standing", actor="actor")
    cfg = SonicTinyConfig(future_frames=2, future_stride=1)
    bank = MotionBank(
        [path],
        cfg,
        "cpu",
        adaptive_sampling_bin_frames=10,
        pre_failure_sample_window=0,
    )
    assert len(bank.failure_ema) == 2
    bank.update_failures(torch.tensor([0]), torch.tensor([15]), torch.tensor([True]))
    assert bank.failure_ema[1] > bank.failure_ema[0]
    assert bank.sampling_weights()[1] > bank.sampling_weights()[0]


def test_freeze_frame_augmentation_freezes_pose_and_velocities(tmp_path: Path):
    path = tmp_path / "walk.npz"
    _write_motion(path, "walk", "walking", actor="actor", speed=0.5)
    cfg = SonicTinyConfig(future_frames=2, future_stride=1)
    torch.manual_seed(3)
    bank = MotionBank([path], cfg, "cpu", freeze_frame_probability=1.0)
    freeze_at = int(bank.freeze_frames[0])
    assert freeze_at >= 0
    expected = bank.root_pos[0, freeze_at].expand_as(bank.root_pos[0, freeze_at:])
    torch.testing.assert_close(bank.root_pos[0, freeze_at:], expected)
    assert not bank.joint_vel[0, freeze_at:].any()
    assert not bank.body_linvel[0, freeze_at:].any()
