from pathlib import Path

import numpy as np
import pandas as pd

from mini_groot_sonic.data.bones import load_g1_csv


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
