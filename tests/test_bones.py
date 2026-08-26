from pathlib import Path

import numpy as np
import pandas as pd

from mini_groot_sonic.data.bones import BonesSeedIndex, load_g1_csv


def test_bones_csv_loader(tmp_path: Path):
    joints = [f"j{i}" for i in range(29)]
    n = 13
    data = {
        "Frame": np.arange(n),
        "root_translateX": np.linspace(0, 10, n),
        "root_translateY": np.zeros(n),
        "root_translateZ": np.ones(n) * 80,
        "root_rotateX": np.zeros(n),
        "root_rotateY": np.zeros(n),
        "root_rotateZ": np.linspace(0, 15, n),
    }
    for j in joints:
        data[f"{j}_dof"] = np.linspace(0, 5, n)
    p = tmp_path / "m.csv"
    pd.DataFrame(data).to_csv(p, index=False)
    clip = load_g1_csv(p, joints, "test", source_fps=120, target_fps=60)
    assert clip.joint_pos.shape[1] == 29
    assert clip.root_quat.shape[1] == 4
    assert np.isfinite(clip.joint_vel).all()
    assert np.isclose(clip.root_pos[0, 2], 0.8)


def test_bones_index_filters_before_seeded_limit(tmp_path: Path):
    csv_root = tmp_path / "g1" / "csv"
    metadata_root = tmp_path / "metadata"
    csv_root.mkdir(parents=True)
    metadata_root.mkdir()
    rows = [
        {
            "motion_id": f"unextracted_{index:05d}",
            "content_short_description": "unused motion",
        }
        for index in range(5001)
    ] + [
        {"motion_id": "easy_walk", "content_short_description": "walk forward"},
        {"motion_id": "easy_stand", "content_short_description": "stand idle"},
        {"motion_id": "hard_flip", "content_short_description": "back flip"},
    ]
    for row in rows:
        (csv_root / f"{row['motion_id']}.csv").touch()
    pd.DataFrame(rows).to_csv(metadata_root / "seed_metadata_v001.csv", index=False)

    index = BonesSeedIndex(tmp_path)
    records = list(
        index.iter_records(
            limit=2,
            seed=7,
            include_keywords=("walk", "stand", "flip"),
            exclude_keywords=("flip",),
        )
    )
    assert {stem for stem, _, _ in records} == {"easy_walk", "easy_stand"}
