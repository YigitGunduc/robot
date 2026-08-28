from __future__ import annotations

import argparse
from pathlib import Path

from gear_sonic_mjx.data_process.bones import MotionClip
from gear_sonic_mjx.data_process.fk_cache import MujocoFKCache
from gear_sonic_mjx.g1_parameters import SONIC_TRACKED_BODY_NAMES


def all_robot_bodies(mjcf: str) -> list[str]:
    import mujoco

    m = mujoco.MjModel.from_xml_path(mjcf)
    names = []
    for bid in range(1, m.nbody):  # skip world
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name:
            names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser(
        description="Augment preprocessed BONES clips with MuJoCo body FK for SONIC body-space rewards"
    )
    ap.add_argument("--motions", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument(
        "--bodies", nargs="*", help="Body names; default is every named non-world body"
    )
    args = ap.parse_args()
    bodies = args.bodies or SONIC_TRACKED_BODY_NAMES
    fk = MujocoFKCache(args.mjcf, bodies)
    files = sorted(
        p
        for p in Path(args.motions).rglob("*.npz")
        if p.name not in {"_manifest.npz", "_packed_metadata.npz"}
    )
    for i, p in enumerate(files):
        clip = MotionClip.load_npz(p)
        fk.augment(clip).save_npz(p)
        if i % 100 == 0:
            print(f"FK {i}/{len(files)} {p.name}")
    print(f"done: {len(files)} clips, {len(bodies)} bodies")


if __name__ == "__main__":
    main()
