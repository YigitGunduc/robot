from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gear_sonic_mjx.config import SonicConfig
from gear_sonic_mjx.data_process.bones import MotionClip, resample_motion
from gear_sonic_mjx.data_process.annotations import load_seed_metadata, load_seed_timelines, merge_annotations
from gear_sonic_mjx.envs.mdp.observations import g1_tokenizer_observation
from gear_sonic_mjx.math_utils import projected_gravity
from gear_sonic_mjx.trl.modules.universal_token_modules import UniversalTokenModule


def load_module(checkpoint: str, cfg: SonicConfig, device: torch.device) -> UniversalTokenModule:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    # Prefer the model/motion configuration stored with the training checkpoint so a --network
    # nvidia checkpoint cannot be accidentally reconstructed with the default small widths.
    saved_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(saved_cfg, SonicConfig):
        cfg.model = saved_cfg.model
        cfg.motion = saved_cfg.motion
    model = UniversalTokenModule(cfg.model, cfg.motion.num_future_frames, cfg.motion.actor_prop_history_length).to(device)
    state = ckpt.get("token_module", ckpt.get("model", ckpt))
    # Accept trainer checkpoints where keys are prefixed by token_module.
    if any(k.startswith("token_module.") for k in state):
        state = {k.removeprefix("token_module."): v for k, v in state.items() if k.startswith("token_module.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def encode_clip(model: UniversalTokenModule, clip: MotionClip, cfg: SonicConfig, device: torch.device, batch_size: int = 512):
    clip = resample_motion(clip, cfg.motion.target_fps)
    T = clip.num_frames
    stride = max(1, int(round(cfg.motion.dt_future_ref_frames * clip.fps)))
    chunks = []
    for start in range(0, T, batch_size):
        frames = np.arange(start, min(start + batch_size, T))
        idx = np.minimum(frames[:, None] + np.arange(cfg.motion.num_future_frames)[None] * stride, T - 1)
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
    state = torch.cat([torch.from_numpy(clip.joint_pos).float(), projected_gravity(rootq)], -1).numpy().astype(np.float32)
    return tokens, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", required=True, help="Preprocessed BONES .npz directory")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "gear_sonic_mjx/config/sonic_release_mjx.yaml"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metadata", default=None, help="Official BONES seed_metadata_v004.parquet/csv")
    ap.add_argument("--timelines", default=None, help="BONES/NVIDIA temporal-label JSONL")
    args = ap.parse_args()
    cfg = SonicConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_module(args.checkpoint, cfg, device)
    metadata = load_seed_metadata(args.metadata) if args.metadata else {}
    timelines = load_seed_timelines(args.timelines) if args.timelines else {}
    annotations = merge_annotations(metadata, timelines)
    src, dst = Path(args.motions), Path(args.output)
    files = sorted(p for p in src.rglob("*.npz") if p.name != "_manifest.npz")
    for i, p in enumerate(files):
        clip = MotionClip.load_npz(p)
        tokens, state = encode_clip(model, clip, cfg, device)
        out = (dst / p.relative_to(src)); out.parent.mkdir(parents=True, exist_ok=True)
        ann = annotations.get(clip.name)
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
                payload["timeline_start"] = np.asarray([e.start_time for e in ann.events], dtype=np.float32)
                payload["timeline_end"] = np.asarray([e.end_time for e in ann.events], dtype=np.float32)
                payload["timeline_text"] = np.asarray([e.description for e in ann.events], dtype="U2048")
            if ann.category:
                payload["category"] = np.asarray(ann.category)
            if ann.package:
                payload["package"] = np.asarray(ann.package)
        np.savez_compressed(out, **payload)
        if i % 100 == 0:
            print(f"encoded {i}/{len(files)}: {p.name}")

if __name__ == "__main__":
    main()
