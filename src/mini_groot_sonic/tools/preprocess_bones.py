from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mini_groot_sonic.data.bones import (
    SONIC_DEFAULT_FILTER_KEYWORDS,
    BonesClip,
    BonesSeedIndex,
    load_g1_csv,
)
from mini_groot_sonic.data.preprocess import (
    precompute_mujoco_kinematics,
    save_preprocessed,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert official BONES-SEED G1 CSVs into compact training NPZ clips")
    ap.add_argument("--bones-root", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0, help="Deterministic BONES subset seed")
    ap.add_argument("--source-fps", type=float, default=120.0)
    ap.add_argument("--target-fps", type=float, default=50.0)
    ap.add_argument("--caption-field", default="content_short_description")
    ap.add_argument(
        "--include-keywords",
        default="",
        help="Comma-separated filename/path terms; keep records matching at least one",
    )
    ap.add_argument(
        "--exclude-keywords",
        default=",".join(SONIC_DEFAULT_FILTER_KEYWORDS),
        help="Comma-separated filename/path terms to reject (defaults to SONIC's denylist)",
    )
    ap.add_argument("--no-temporal-segments", action="store_true")
    ap.add_argument(
        "--min-clip-seconds",
        type=float,
        default=1.0,
        help="Discard temporal segments too short for the default future-reference window",
    )
    args = ap.parse_args()
    include_keywords = [term for term in args.include_keywords.split(",") if term.strip()]
    exclude_keywords = [term for term in args.exclude_keywords.split(",") if term.strip()]

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("Install simulator extras first: pip install -e '.[sim]'") from exc

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    # Only the actuator/joint order is needed here; body names are discovered during kinematics preprocessing.
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise SystemExit("Expected one floating base in G1 MJCF")
    # Manual minimal mapping since body list may vary across G1 XMLs.
    joint_names = []
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        if jid >= 0 and int(model.jnt_type[jid]) in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    if len(joint_names) != 29:
        raise SystemExit(f"Expected 29 actuated G1 joints, found {len(joint_names)}")

    index = BonesSeedIndex(args.bones_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, (stem, path, captions) in enumerate(
        index.iter_records(
            args.limit,
            args.caption_field,
            args.seed,
            include_keywords,
            exclude_keywords,
        ),
        start=1,
    ):
        caption = captions[0]
        clip = load_g1_csv(path, joint_names, caption, args.source_fps, args.target_fps)
        minimum_frames = max(2, round(args.min_clip_seconds * clip.fps))
        if clip.length < minimum_frames:
            print(f"[{n}] skipped {stem}: only {clip.length} frames")
            continue
        metadata = index.metadata_for(stem)
        segments = [] if args.no_temporal_segments else index.segments_for(stem)
        clip_specs: list[tuple[BonesClip, list[str]]] = []
        for segment_index, event in enumerate(segments):
            start = max(0, round(float(event["start_time"]) * clip.fps))
            end = min(clip.length, round(float(event["end_time"]) * clip.fps) + 1)
            description = str(event.get("description", "")).strip()
            if end - start < minimum_frames or not description:
                continue
            description_lower = description.lower()
            if any(term.strip().lower() in description_lower for term in exclude_keywords):
                continue
            clip_specs.append(
                (
                    BonesClip(
                        motion_id=f"{stem}__segment_{segment_index:02d}",
                        caption=description,
                        fps=clip.fps,
                        joint_pos=clip.joint_pos[start:end],
                        joint_vel=clip.joint_vel[start:end],
                        root_pos=clip.root_pos[start:end],
                        root_quat=clip.root_quat[start:end],
                    ),
                    [description],
                )
            )
        if not clip_specs and segments and exclude_keywords:
            print(f"[{n}] skipped {stem}: no safe temporal segments remained")
            continue
        if not clip_specs:
            clip_specs = [(clip, captions)]

        for prepared_clip, prepared_captions in clip_specs:
            arrays = precompute_mujoco_kinematics(prepared_clip, args.mjcf, joint_names)
            arrays["captions"] = np.asarray(prepared_captions, dtype=object)
            arrays["source_motion_id"] = np.asarray(stem, dtype=object)
            for key, value in metadata.items():
                arrays[key] = np.asarray(value, dtype=object)
            save_preprocessed(out / f"{prepared_clip.motion_id}.npz", arrays)
        print(
            f"[{n}] {stem}: {clip.length} frames, {len(captions)} captions, "
            f"{len(clip_specs)} training clip(s)"
        )


if __name__ == "__main__":
    main()
