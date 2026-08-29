from __future__ import annotations

import argparse
from pathlib import Path

from gear_sonic_mjx.envs.motion_library import open_motion_library
from gear_sonic_mjx.g1_parameters import G1_MUJOCO_JOINT_NAMES


def ensure_offscreen_framebuffer(model, width: int, height: int) -> None:
    """Make MuJoCo's offscreen framebuffer large enough for the requested video."""
    if width <= 0 or height <= 0:
        raise ValueError(f"render dimensions must be positive, got {width}x{height}")
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a BONES reference clip on the exact G1 MJCF for visual preflight"
    )
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--motions", required=True)
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    import imageio.v2 as iio
    import mujoco

    library = open_motion_library(args.motions, args.fps)
    if not 0 <= args.motion_id < len(library):
        raise IndexError(f"motion id must be in [0, {len(library)})")
    clip = library._load(args.motion_id)
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    ensure_offscreen_framebuffer(model, args.width, args.height)
    data = mujoco.MjData(model)
    free_ids = [
        jid
        for jid in range(model.njnt)
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_ids) != 1:
        raise ValueError(f"expected one free joint, found {len(free_ids)}")
    root_adr = int(model.jnt_qposadr[free_ids[0]])
    joint_qadr = []
    for name in G1_MUJOCO_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"MJCF missing canonical joint {name!r}")
        joint_qadr.append(int(model.jnt_qposadr[jid]))

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.2
    camera.azimuth = 135.0
    camera.elevation = -18.0
    count = (
        clip.num_frames
        if args.max_frames is None
        else min(clip.num_frames, args.max_frames)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    writer = iio.get_writer(partial, fps=clip.fps, codec="libx264")
    try:
        for frame in range(count):
            data.qpos[root_adr : root_adr + 3] = clip.root_pos[frame]
            data.qpos[root_adr + 3 : root_adr + 7] = clip.root_quat_wxyz[frame]
            data.qpos[joint_qadr] = clip.joint_pos[frame]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = clip.root_pos[frame]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    partial.replace(output)
    print(f"rendered {clip.name!r}: {count} frames at {clip.fps:g} Hz -> {output}")


if __name__ == "__main__":
    main()
