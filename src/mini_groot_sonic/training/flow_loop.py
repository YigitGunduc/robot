from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mini_groot_sonic.checkpoint import BODY_CONTROL_STACK_VERSION
from mini_groot_sonic.config import FlowConfig
from mini_groot_sonic.data.replay_dataset import ReplayWindowDataset
from mini_groot_sonic.models.flow_policy import TinyFlowMotionPolicy
from mini_groot_sonic.models.frozen_backbones import FrozenSiglip2
from mini_groot_sonic.training.utils import (
    restore_rng_state,
    rng_state,
    save_config_snapshot,
    seed_everything,
)


def _collate(batch):
    out = {
        "caption": [x["caption"] for x in batch],
        "state": torch.stack([x["state"] for x in batch]),
        "actions": torch.stack([x["actions"] for x in batch]),
        "valid": torch.stack([x["valid"] for x in batch]),
        "goal": torch.stack([x["goal"] for x in batch]),
    }
    if all("rgb" in x for x in batch):
        out["rgb"] = torch.stack([x["rgb"] for x in batch])
    return out


def train_flow_policy(
    replay_root: str | Path,
    flow_cfg: FlowConfig,
    output_dir: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    hf_model: str = "google/siglip2-base-patch16-224",
    device: str = "cuda",
    use_vision: bool = False,
    validation_fraction: float = 0.1,
    num_workers: int = 2,
    use_amp: bool = True,
    resume_from: str | Path | None = None,
) -> TinyFlowMotionPolicy:
    seed_everything(flow_cfg.seed)
    device_t = torch.device(device)
    ds = ReplayWindowDataset(
        replay_root,
        horizon=flow_cfg.action_horizon,
        split="train",
        validation_fraction=validation_fraction,
    )
    val_ds = ReplayWindowDataset(
        replay_root,
        horizon=flow_cfg.action_horizon,
        split="val",
        validation_fraction=validation_fraction,
        samples_per_episode=4,
    )
    if ds.body_policy_fingerprint != val_ds.body_policy_fingerprint:
        raise RuntimeError("Training and validation replay target different body policies")
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": _collate,
        "pin_memory": device_t.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    dl = DataLoader(ds, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    backbone = FrozenSiglip2(hf_model, device=device)

    # Probe actual frozen embedding dimensions rather than hard-coding HF internals.
    with torch.no_grad():
        tprobe = backbone.encode_text(["walk forward"])
        flow_cfg.text_dim = int(tprobe.shape[-1])
        if use_vision:
            first = ds[0]
            if "rgb" not in first:
                raise ValueError("use_vision=True but replay dataset has no RGB frames")
            from PIL import Image
            vprobe = backbone.encode_images([Image.fromarray(first["rgb"].numpy())])
            flow_cfg.vision_dim = int(vprobe.shape[-1])

    model = TinyFlowMotionPolicy(flow_cfg).to(device_t)
    norm = ds.normalization_stats()
    model.set_normalization_stats(*(norm[k].to(device_t) for k in ("state_mean", "state_std", "goal_mean", "goal_std")))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(output_dir / "config.json", flow=flow_cfg)

    start_epoch = 0
    best_val = float("inf")
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device_t, weights_only=False)
        version = int(ckpt.get("body_control_stack_version", 0))
        if version != BODY_CONTROL_STACK_VERSION:
            raise RuntimeError(
                f"Cannot resume flow checkpoint for body stack v{version}; "
                f"retrain it against v{BODY_CONTROL_STACK_VERSION} replay data"
            )
        if ckpt.get("body_policy_fingerprint") != ds.body_policy_fingerprint:
            raise RuntimeError(
                "Cannot resume flow training with replay from a different body policy"
            )
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        restore_rng_state(ckpt.get("rng_state"))

    from PIL import Image

    def embeddings(batch):
        with torch.no_grad():
            text = backbone.encode_text(batch["caption"]).to(device_t)
            vision = None
            if use_vision:
                images = [Image.fromarray(x.numpy()) for x in batch["rgb"]]
                vision = backbone.encode_images(images).to(device_t)
        return text, vision

    amp_enabled = use_amp and device_t.type == "cuda"
    for epoch in range(start_epoch, epochs):
        model.train()
        running = torch.zeros((), device=device_t)
        count = 0
        for batch in dl:
            state = batch["state"].to(device_t)
            actions = batch["actions"].to(device_t)
            valid = batch["valid"].to(device_t)
            goal = batch["goal"].to(device_t)
            text, vision = embeddings(batch)
            with torch.autocast(device_type=device_t.type, dtype=torch.bfloat16, enabled=amp_enabled):
                out = model.flow_matching_loss(actions, state, text, vision, goal, valid)
            optim.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += out.loss.detach()
            count += 1
        train_mean = float(running / max(count, 1))
        model.eval()
        val_running = torch.zeros((), device=device_t)
        val_count = 0
        sampled_token_mse = float("nan")
        sampled_token_delta = float("nan")
        with torch.no_grad():
            for batch in val_dl:
                state = batch["state"].to(device_t)
                actions = batch["actions"].to(device_t)
                valid = batch["valid"].to(device_t)
                goal = batch["goal"].to(device_t)
                text, vision = embeddings(batch)
                out = model.flow_matching_loss(actions, state, text, vision, goal, valid)
                val_running += out.loss
                val_count += 1
                if val_count == 1:
                    sampled = model.sample(state, text, vision, goal)
                    mask = valid[..., None]
                    sampled_token_mse = float(
                        ((sampled - actions).square() * mask).sum()
                        / mask.sum().clamp_min(1.0)
                        / actions.shape[-1]
                    )
                    sampled_token_delta = float((sampled[:, 1:] - sampled[:, :-1]).square().mean())
        val_mean = float(val_running / max(val_count, 1))
        print(
            f"epoch={epoch:03d} train_flow_loss={train_mean:.6f} "
            f"val_flow_loss={val_mean:.6f} sampled_token_mse={sampled_token_mse:.6f} "
            f"sampled_token_delta={sampled_token_delta:.6f}"
        )
        best_val = min(best_val, val_mean)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "flow_cfg": asdict(flow_cfg),
            "epoch": epoch,
            "best_val": best_val,
            "rng_state": rng_state(),
            "validation": {
                "flow_loss": val_mean,
                "sampled_token_mse": sampled_token_mse,
                "sampled_token_delta": sampled_token_delta,
            },
            "body_control_stack_version": BODY_CONTROL_STACK_VERSION,
            "body_policy_fingerprint": ds.body_policy_fingerprint,
        }
        torch.save(checkpoint, output_dir / f"flow_{epoch:04d}.pt")
        if val_mean <= best_val:
            torch.save(checkpoint, output_dir / "flow_best.pt")
    return model
