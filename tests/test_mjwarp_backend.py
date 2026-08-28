from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic_mjx.data_process.bones import MotionClip, resample_motion
from gear_sonic_mjx.data_process.fk_cache import MujocoFKCache
from gear_sonic_mjx.g1_parameters import (
    DEFAULT_ANGLES_MJ,
    G1_MUJOCO_JOINT_NAMES,
    SONIC_TRACKED_BODY_NAMES,
)


def _minimal_g1_xml(*, semantic_bodies: bool = False) -> str:
    bodies = []
    motors = []
    child_names = [name for name in SONIC_TRACKED_BODY_NAMES if name != "pelvis"]
    child_names.append("head_link")
    child_names.extend(f"body_{index}" for index in range(29 - len(child_names)))
    for index, name in enumerate(G1_MUJOCO_JOINT_NAMES):
        body_name = child_names[index] if semantic_bodies else f"body_{index}"
        bodies.append(
            f'<body name="{body_name}" pos="0 0 {0.03 + index * 0.001}">'
            f'<joint name="{name}" type="hinge" axis="0 0 1" '
            'range="-2 2" damping="0.01"/>'
            '<geom type="sphere" size="0.01" mass="0.01" contype="0" conaffinity="0"/>'
            "</body>"
        )
        motors.append(f'<motor name="motor_{index}" joint="{name}" gear="1"/>')
    return (
        '<mujoco model="minimal_g1"><compiler angle="radian"/>'
        '<option timestep="0.005" gravity="0 0 0"/>'
        '<worldbody><body name="pelvis" pos="0 0 1"><freejoint name="root"/>'
        '<geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>'
        + "".join(bodies)
        + "</body></worldbody><actuator>"
        + "".join(motors)
        + "</actuator></mujoco>"
    )


def test_real_mjwarp_backend_api_and_torque_mapping(tmp_path: Path):
    pytest.importorskip("mujoco_warp")
    warp = pytest.importorskip("warp")
    warp.config.kernel_cache_dir = str(tmp_path / "warp_cache")
    from gear_sonic_mjx.sim.mjwarp_backend import MjWarpBatchSim

    path = tmp_path / "minimal_g1.xml"
    path.write_text(_minimal_g1_xml())
    sim = MjWarpBatchSim(path, nworld=2, timestep=0.005)
    assert sim.qpos.shape[0] == 2
    assert sim.ctrl.shape == (2, 29)
    assert sim.cfrc_ext is not None

    root = torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    joint = torch.zeros(2, 29)
    sim.set_state(torch.arange(2), root, quat, joint, torch.zeros_like(joint))
    command = torch.arange(29, dtype=torch.float32).repeat(2, 1)
    sim.write_torque(command)
    actuator_ids = sim.index.actuator.to(sim.ctrl.device)
    torch.testing.assert_close(sim.ctrl[:, actuator_ids], command.to(sim.ctrl))
    sim.configure_startup_domain_randomization(
        mass_body_names=["body_0"], num_variants=2, seed=3
    )
    sim.step()
    sim.assert_no_overflow()
    assert torch.isfinite(sim.qpos).all()


def test_resampled_fk_is_recomputed_from_final_joint_pose(tmp_path: Path):
    mujoco = pytest.importorskip("mujoco")
    path = tmp_path / "minimal_g1.xml"
    path.write_text(_minimal_g1_xml())
    frames = 7
    joint = np.linspace(0.0, 0.4, frames, dtype=np.float32)[:, None] * np.ones(
        (1, 29), np.float32
    )
    clip = MotionClip(
        "motion",
        30.0,
        np.column_stack(
            [np.linspace(0.0, 0.2, frames), np.zeros(frames), np.ones(frames)]
        ).astype(np.float32),
        np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (frames, 1)),
        joint,
        np.gradient(joint, 1.0 / 30.0, axis=0).astype(np.float32),
    )
    clip = resample_motion(clip, 50.0)
    fk = MujocoFKCache(path, ["body_0", "body_10"])
    fk.augment(clip)

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in clip.body_names
    ]
    joint_qpos = [
        int(
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        )
        for name in G1_MUJOCO_JOINT_NAMES
    ]
    for frame in range(clip.num_frames):
        data.qpos[:3] = clip.root_pos[frame]
        data.qpos[3:7] = clip.root_quat_wxyz[frame]
        data.qpos[joint_qpos] = clip.joint_pos[frame]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(clip.body_pos[frame], data.xpos[body_ids], atol=1e-6)
        np.testing.assert_allclose(
            np.abs(clip.body_quat_wxyz[frame]),
            np.abs(data.xquat[body_ids]),
            atol=1e-6,
        )


def test_task_tracks_frame_clock_and_finishes_stationary_motion(tmp_path: Path):
    warp = pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    warp.config.kernel_cache_dir = str(tmp_path / "warp_cache")
    from gear_sonic_mjx.config import SonicConfig
    from gear_sonic_mjx.envs.g1_tracking_task import G1SonicTrackingTask
    from gear_sonic_mjx.envs.motion_library import BonesMotionLibrary
    from gear_sonic_mjx.preflight import PreflightReport, validate_mjcf
    from gear_sonic_mjx.sim.mjwarp_backend import MjWarpBatchSim

    path = tmp_path / "semantic_g1.xml"
    path.write_text(_minimal_g1_xml(semantic_bodies=True))
    motion_root = tmp_path / "motions"
    frames = 5
    clip = MotionClip(
        "stationary",
        50.0,
        np.tile(np.array([0.0, 0.0, 1.0], np.float32), (frames, 1)),
        np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (frames, 1)),
        np.tile(DEFAULT_ANGLES_MJ.numpy(), (frames, 1)),
        np.zeros((frames, 29), np.float32),
    )
    MujocoFKCache(path, SONIC_TRACKED_BODY_NAMES).augment(clip)
    clip.save_npz(motion_root / "stationary.npz")

    config = SonicConfig(num_envs=2)
    config.observation_noise.enabled = False
    config.motion.freeze_frame_aug = False
    config.domain_randomization = {}
    report = PreflightReport()
    validate_mjcf(path, config, report)
    assert report.passed, report.errors
    library = BonesMotionLibrary(motion_root, target_fps=50.0)
    simulator = MjWarpBatchSim(path, nworld=2, timestep=0.005)
    task = G1SonicTrackingTask(
        simulator, library, config, auto_reset=False, enforce_episode_length=False
    )
    encoder, proprio, critic = task.reset_to(torch.zeros(2, dtype=torch.long))
    assert encoder.shape == (2, 640)
    assert proprio.shape == (2, 930)
    assert critic.shape == (2, 1645)

    for expected_frame in range(1, frames):
        step = task.step(torch.zeros(2, 29, device=simulator.device))
        assert step.info["motion_frame"].tolist() == [expected_frame, expected_frame]
        assert step.done.tolist() == [expected_frame == frames - 1] * 2
        assert step.info["failed"].tolist() == [0.0, 0.0]
        assert torch.isfinite(step.reward).all()
