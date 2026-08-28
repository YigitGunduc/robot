from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gear_sonic_mjx.config import SonicConfig
from gear_sonic_mjx.data_process.annotations import (
    load_seed_metadata,
    load_seed_timelines,
    merge_annotations,
)
from gear_sonic_mjx.data_process.bones import MotionClip, resample_motion
from gear_sonic_mjx.data_process.splits import SPLIT_NAMES, load_split_files
from gear_sonic_mjx.envs.mdp.observations import g1_tokenizer_observation
from gear_sonic_mjx.envs.motion_library import open_motion_library
from gear_sonic_mjx.math_utils import projected_gravity
from gear_sonic_mjx.trl.modules.universal_token_modules import UniversalTokenModule


def load_module(
    checkpoint: str, cfg: SonicConfig, device: torch.device
) -> UniversalTokenModule:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    # Prefer the model/motion configuration stored with the training checkpoint so a --network
    # nvidia checkpoint cannot be accidentally reconstructed with the default small widths.
    saved_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(saved_cfg, SonicConfig):
        cfg.model = saved_cfg.model
        cfg.motion = saved_cfg.motion
    model = UniversalTokenModule(
        cfg.model, cfg.motion.num_future_frames, cfg.motion.actor_prop_history_length
    ).to(device)
    state = ckpt.get("token_module", ckpt.get("model", ckpt))
    # Accept trainer checkpoints where keys are prefixed by token_module.
    if any(k.startswith("token_module.") for k in state):
        state = {
            k.removeprefix("token_module."): v
            for k, v in state.items()
            if k.startswith("token_module.")
        }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _unique_annotation_stems(annotations):
    indexed = {}
    ambiguous = set()
    for key, value in annotations.items():
        stem = Path(key).stem
        if stem in indexed:
            ambiguous.add(stem)
        else:
            indexed[stem] = value
    for stem in ambiguous:
        indexed.pop(stem, None)
    return indexed


@torch.no_grad()
def encode_clip(
    model: UniversalTokenModule,
    clip: MotionClip,
    cfg: SonicConfig,
    device: torch.device,
    batch_size: int = 512,
):
    clip = resample_motion(clip, cfg.motion.target_fps)
    T = clip.num_frames
    stride = max(1, round(cfg.motion.dt_future_ref_frames * clip.fps))
    chunks = []
    for start in range(0, T, batch_size):
        frames = np.arange(start, min(start + batch_size, T))
        idx = np.minimum(
            frames[:, None] + np.arange(cfg.motion.num_future_frames)[None] * stride,
            T - 1,
        )
        q = torch.from_numpy(clip.joint_pos[idx]).to(device)
        qd = torch.from_numpy(clip.joint_vel[idx]).to(device)
        rq = torch.from_numpy(clip.root_quat_wxyz[frames]).to(device)
        fq = torch.from_numpy(clip.root_quat_wxyz[idx]).to(device)
        obs = g1_tokenizer_observation(q, qd, rq, fq)
        _, z = model.encode(obs)
        chunks.append(z.cpu())
    tokens = torch.cat(chunks).numpy().astype(np.float32)
    # Current N1.7 G1-SONIC state uses body joint positions + projected gravity (hands omitted here).
    rootq = torch.from_numpy(clip.root_quat_wxyz).float()
    state = (
        torch.cat(
            [torch.from_numpy(clip.joint_pos).float(), projected_gravity(rootq)], -1
        )
        .numpy()
        .astype(np.float32)
    )
    return tokens, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--motions", required=True, help="Preprocessed BONES .npz directory"
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--config",
        default=str(
            Path(__file__).parents[1] / "gear_sonic_mjx/config/sonic_release_mjx.yaml"
        ),
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--metadata", default=None, help="Official BONES seed_metadata_v004.parquet/csv"
    )
    ap.add_argument(
        "--timelines", default=None, help="BONES/NVIDIA temporal-label JSONL"
    )
    ap.add_argument(
        "--split-manifest", help="JSON manifest created by scripts/split_bones.py"
    )
    ap.add_argument("--split", choices=SPLIT_NAMES, help="manifest split to encode")
    ap.add_argument(
        "--require-captions",
        action="store_true",
        help="fail if any encoded clip cannot be matched to official metadata/timelines",
    )
    args = ap.parse_args()
    cfg = SonicConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_module(args.checkpoint, cfg, device)
    metadata = load_seed_metadata(args.metadata) if args.metadata else {}
    timelines = load_seed_timelines(args.timelines) if args.timelines else {}
    annotations = merge_annotations(metadata, timelines)
    annotations_by_stem = _unique_annotation_stems(annotations)
    src, dst = Path(args.motions), Path(args.output)
    if bool(args.split_manifest) != bool(args.split):
        raise ValueError("--split-manifest and --split must be provided together")
    packed = (src / "_packed_metadata.npz").is_file()
    if packed and args.split_manifest:
        raise ValueError(
            "packed libraries are already split; omit --split-manifest/--split"
        )
    if packed:
        library = open_motion_library(src, cfg.motion.target_fps)
        items = [
            (index, Path(library.source_relpaths[index]))
            for index in range(len(library))
        ]
    else:
        files = (
            load_split_files(src, args.split_manifest, args.split)
            if args.split_manifest
            else sorted(p for p in src.rglob("*.npz") if p.name != "_manifest.npz")
        )
        items = [(path, path.relative_to(src)) for path in files]
    captioned = 0
    unmatched = []
    for i, (source, relative) in enumerate(items):
        clip = library._load(source) if packed else MotionClip.load_npz(source)
        tokens, state = encode_clip(model, clip, cfg, device)
        out = dst / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        ann = annotations.get(clip.name) or annotations_by_stem.get(
            Path(clip.name).stem
        )
        payload = {
            "tokens": tokens,
            "state": state,
            "fps": np.asarray(cfg.motion.target_fps, dtype=np.float32),
            "text": np.asarray(clip.name.replace("_", " ")),  # fallback only
        }
        if ann is not None:
            if ann.captions:
                payload["captions"] = np.asarray(ann.captions, dtype="U2048")
            if ann.events:
                payload["timeline_start"] = np.asarray(
                    [e.start_time for e in ann.events], dtype=np.float32
                )
                payload["timeline_end"] = np.asarray(
                    [e.end_time for e in ann.events], dtype=np.float32
                )
                payload["timeline_text"] = np.asarray(
                    [e.description for e in ann.events], dtype="U2048"
                )
            if ann.category:
                payload["category"] = np.asarray(ann.category)
            if ann.package:
                payload["package"] = np.asarray(ann.package)
            if ann.captions or ann.events:
                captioned += 1
            else:
                unmatched.append(clip.name)
        else:
            unmatched.append(clip.name)
        np.savez_compressed(out, **payload)
        if i % 100 == 0:
            print(f"encoded {i}/{len(items)}: {relative.name}")
    coverage = captioned / max(len(items), 1)
    print(
        f"caption/timeline coverage: {captioned}/{len(items)} "
        f"({coverage:.2%}); unmatched examples={unmatched[:10]}"
    )
    if args.require_captions and captioned != len(items):
        raise RuntimeError(
            f"official annotations matched only {captioned}/{len(items)} clips; "
            "refusing caption-supervised export"
        )


if __name__ == "__main__":
    main()
