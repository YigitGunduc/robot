from pathlib import Path

import numpy as np
import pandas as pd

from gear_sonic_mjx.data_process.bones import (
    MotionClip,
    resample_motion,
    resampled_frame_count,
    should_filter_out,
)
from gear_sonic_mjx.data_process.splits import motion_group_key
from gear_sonic_mjx.data_process.subset import (
    build_easy_subset_manifest,
    easy_motion_class,
    materialize_subset,
)


def test_nvidia_filter_examples():
    assert should_filter_out("subject_walk_to_chair_003.csv")
    assert should_filter_out("acrobatics_cartwheel_01.csv")
    assert should_filter_out("stairs_up_001.csv")
    assert not should_filter_out("subject_walk_forward_003.csv")
    assert not should_filter_out("dance_wave_004.csv")


def test_motion_resample():
    T = 121
    clip = MotionClip(
        "walk",
        120.0,
        np.stack([np.linspace(0, 1, T), np.zeros(T), np.ones(T)], -1).astype(
            np.float32
        ),
        np.tile(np.array([1, 0, 0, 0], np.float32), (T, 1)),
        np.zeros((T, 29), np.float32),
        np.zeros((T, 29), np.float32),
    )
    r = resample_motion(clip, 50.0)
    # Matches NVIDIA's arange(0, duration, 1 / target_fps): the endpoint is excluded.
    assert r.num_frames == 50
    assert abs(r.duration - 0.98) < 1e-6
    np.testing.assert_allclose(r.root_pos[-1, 0], 0.98, atol=1e-6)


def test_resampled_frame_count_does_not_stretch_timeline():
    assert resampled_frame_count(121, 120.0, 50.0) == 50
    assert resampled_frame_count(120, 120.0, 50.0) == 50


def test_easy_subset_is_deterministic_balanced_and_mirror_safe(tmp_path):
    motions = tmp_path / "motions"
    rows = []
    specs = {
        "walk": (
            "walk_ff_loop_180_R_normal_pace",
            "Basic Locomotion Neutral",
            "walking",
        ),
        "turn": ("turn_walk_090_R", "Basic Locomotion Neutral", "walking, turning"),
        "gesture": ("wave_R", "Gestures", "gesture"),
    }
    relpaths = []
    for motion_class, (prefix, category, movement) in specs.items():
        for index in range(6):
            for mirror in (False, True):
                filename = f"{prefix}_{index:03d}__A001" + ("_M" if mirror else "")
                relative = f"g1/csv/000000/{filename}.npz"
                clip = MotionClip(
                    filename,
                    30.0,
                    np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]], np.float32),
                    np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (2, 1)),
                    np.zeros((2, 29), np.float32),
                    np.zeros((2, 29), np.float32),
                )
                clip.save_npz(motions / relative)
                relpaths.append(relative)
                rows.append(
                    {
                        "filename": filename,
                        "move_g1_path": str(Path(relative).with_suffix(".csv")),
                        "category": category,
                        "content_type_of_movement": movement,
                        "content_body_position": "standing",
                        "content_short_description": motion_class,
                        "content_props": "0",
                        "content_complex_action": 0,
                        "is_neutral": 1.0,
                        "move_duration_frames": 300,
                    }
                )
    np.savez_compressed(
        motions / "_manifest.npz",
        relpaths=np.asarray(relpaths),
        num_frames=np.full(len(relpaths), 2, np.int32),
        fps=np.full(len(relpaths), 30.0, np.float32),
    )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)

    first = build_easy_subset_manifest(motions, metadata, max_clips=20, seed=7)
    second = build_easy_subset_manifest(motions, metadata, max_clips=20, seed=7)
    assert first == second
    assert len(first["selected_relpaths"]) == 20
    assert all(first["class_counts"][name] > 0 for name in specs)
    assert (
        max(first["class_counts"].values()) - min(first["class_counts"].values()) <= 2
    )
    selected_stems = {Path(path).stem for path in first["selected_relpaths"]}
    for stem in selected_stems:
        base = stem.removesuffix("_M")
        assert base in selected_stems
        assert base + "_M" in selected_stems

    output = tmp_path / "easy"
    stats = materialize_subset(motions, output, first)
    assert stats["clips"] == 20
    manifest = np.load(output / "_manifest.npz", allow_pickle=False)
    assert len(manifest["relpaths"]) == 20


def test_bones_terminal_m_mirror_grouping():
    assert motion_group_key("walk_ff_001__A001.npz") == motion_group_key(
        "walk_ff_001__A001_M.npz"
    )
    assert motion_group_key("walk_ff_001__A001.npz") == motion_group_key(
        "walk_ff_001__A014_M.npz"
    )


def test_easy_subset_rejects_crossed_arms_and_backward_turn_false_positives():
    base = {
        "category": "Basic Locomotion Neutral",
        "content_type_of_movement": "walking, turning",
        "content_body_position": "standing",
        "content_short_description": "turning while walking",
        "content_props": "0",
        "content_complex_action": 0,
        "is_neutral": 1.0,
        "move_duration_frames": 300,
    }
    assert (
        easy_motion_class(
            {**base, "filename": "crossed_arms_idle_turn_045_R_001__A001"}
        )
        is None
    )
    assert (
        easy_motion_class({**base, "filename": "turn_walk_backward_045_R_001__A001"})
        is None
    )
    assert (
        easy_motion_class({**base, "filename": "turn_walk_045_R_001__A001"}) == "turn"
    )
