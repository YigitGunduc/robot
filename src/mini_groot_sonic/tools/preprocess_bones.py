from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mini_groot_sonic.config import SimConfig
from mini_groot_sonic.data.bones import BonesSeedIndex, load_g1_csv
from mini_groot_sonic.data.preprocess import precompute_mujoco_kinematics, save_preprocessed
from mini_groot_sonic.sim.g1_mapping import G1ModelMap


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert official BONES-SEED G1 CSVs into compact training NPZ clips")
    ap.add_argument("--bones-root", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--source-fps", type=float, default=120.0)
    ap.add_argument("--target-fps", type=float, default=50.0)
    ap.add_argument("--caption-field", default="content_short_description")
    args = ap.parse_args()

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("Install simulator extras first: pip install -e '.[sim]'") from exc

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    # Only the actuator/joint order is needed here; body names are discovered during kinematics preprocessing.
    dummy_bodies = []
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise SystemExit("Expected one floating base in G1 MJCF")
    # Manual minimal mapping since body list may vary across G1 XMLs.
    joint_names = []
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        if jid >= 0 and model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    if len(joint_names) != 29:
        raise SystemExit(f"Expected 29 actuated G1 joints, found {len(joint_names)}")

    index = BonesSeedIndex(args.bones_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, (stem, path, captions) in enumerate(index.iter_records(args.limit), start=1):
        caption = captions[0]
        clip = load_g1_csv(path, joint_names, caption, args.source_fps, args.target_fps)
        arrays = precompute_mujoco_kinematics(clip, args.mjcf, joint_names)
        arrays["captions"] = np.asarray(captions, dtype=object)
        save_preprocessed(out / f"{stem}.npz", arrays)
        print(f"[{n}] {stem}: {clip.length} frames, {len(captions)} captions")


if __name__ == "__main__":
    main()
