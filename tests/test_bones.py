import numpy as np

from gear_sonic_mjx.data_process.bones import (
    MotionClip,
    resample_motion,
    resampled_frame_count,
    should_filter_out,
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
