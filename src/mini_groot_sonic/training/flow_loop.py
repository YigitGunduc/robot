from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mini_groot_sonic.config import FlowConfig
from mini_groot_sonic.data.replay_dataset import ReplayWindowDataset
from mini_groot_sonic.models.flow_policy import TinyFlowMotionPolicy
from mini_groot_sonic.models.frozen_backbones import FrozenSiglip2


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
) -> TinyFlowMotionPolicy:
    device_t = torch.device(device)
    ds = ReplayWindowDataset(replay_root, horizon=flow_cfg.action_horizon)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate)
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
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    for epoch in range(epochs):
        model.train()
        running = 0.0
        count = 0
        for batch in dl:
            state = batch["state"].to(device_t)
            actions = batch["actions"].to(device_t)
            valid = batch["valid"].to(device_t)
            goal = batch["goal"].to(device_t)
            with torch.no_grad():
                text = backbone.encode_text(batch["caption"]).to(device_t)
                vision = None
                if use_vision:
                    images = [Image.fromarray(x.numpy()) for x in batch["rgb"]]
                    vision = backbone.encode_images(images).to(device_t)

            out = model.flow_matching_loss(actions, state, text, vision, goal, valid)
            optim.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += float(out.loss.detach())
            count += 1
        mean = running / max(count, 1)
        print(f"epoch={epoch:03d} flow_loss={mean:.6f}")
        torch.save(
            {"model": model.state_dict(), "flow_cfg": asdict(flow_cfg), "epoch": epoch},
            output_dir / f"flow_{epoch:04d}.pt",
        )
    return model
