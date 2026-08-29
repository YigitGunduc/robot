from pathlib import Path
import numpy as np

from sonic_lite_g1.data.pack_motions import pack
from sonic_lite_g1.data.select_bones import classify, difficulty


def test_classify():
    assert classify("neutral_walk_forward.csv") == "walk"
    assert classify("stand_idle_01.csv") == "stand"
    assert classify("walking_on_edge.csv") is None
    assert classify("box_jump.csv") is None


def test_difficulty(tmp_path: Path):
    n = 20
    root = np.zeros((n, 3)); root[:, 2] = 0.8
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    joints = np.zeros((n, 29))
    x = np.concatenate([root, quat, joints], axis=1)
    p = tmp_path / "stand.csv"
    np.savetxt(p, x, delimiter=",")
    assert difficulty(p, 120.0) < 0.01


def test_pack(tmp_path: Path):
    files = []
    for i, n in enumerate([4, 6]):
        p = tmp_path / f"m{i}.npz"
        np.savez_compressed(
            p,
            joint_pos=np.zeros((n, 29), np.float32),
            joint_vel=np.zeros((n, 29), np.float32),
            body_pos_w=np.zeros((n, 14, 3), np.float32),
            body_quat_w=np.zeros((n, 14, 4), np.float32),
            body_lin_vel_w=np.zeros((n, 14, 3), np.float32),
            body_ang_vel_w=np.zeros((n, 14, 3), np.float32),
        )
        files.append(p)
    out = tmp_path / "pack.npz"
    pack(files, out)
    with np.load(out) as d:
        assert d["joint_pos"].shape[0] == 10
        assert d["clip_starts"].tolist() == [0, 4]
        assert d["clip_lengths"].tolist() == [4, 6]


def test_native_bones_conversion(tmp_path):
    from sonic_lite_g1.data.bones_to_mjlab_csv import convert_file
    import numpy as np

    header = [
        "Frame", "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        *[f"joint_{i}_dof" for i in range(29)],
    ]
    src = tmp_path / "walk_test.csv"
    rows = []
    for i in range(5):
        rows.append([i, 100 + i, 0, 80, 0, 0, 90, *([10] * 29)])
    src.write_text(",".join(header) + "\n" + "\n".join(",".join(map(str, r)) for r in rows))
    dst = tmp_path / "out.csv"
    convert_file(src, dst)
    x = np.loadtxt(dst, delimiter=",")
    assert x.shape == (5, 36)
    assert np.isclose(x[0, 0], 1.0)
    assert np.isclose(x[0, 2], 0.8)
    assert np.allclose(x[:, 7:], np.deg2rad(10.0), atol=1e-7)
    # 90-deg yaw -> qz=qw=sqrt(1/2) in xyzw order.
    assert np.allclose(x[0, 3:7], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
    from sonic_lite_g1.data.select_bones import difficulty
    assert np.isfinite(difficulty(src, 120.0))
