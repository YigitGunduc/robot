from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from groot_lite.dataset import TokenTrajectoryDataset
from groot_lite.model import FrozenSiglip2Backbone, GrootLitePolicy


def cosine_warmup(step: int, total: int, warmup_fraction: float) -> float:
    warm = max(1, int(total * warmup_fraction))
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(total - warm, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train compact GR00T-like flow policy on SONIC token trajectories")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parents[1] / "gear_sonic_mjx/config/groot_lite.yaml"))
    ap.add_argument("--output", default="runs/groot_lite")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = TokenTrajectoryDataset(args.data, cfg["model"]["action_horizon"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    backbone = FrozenSiglip2Backbone(cfg["backbone"]["model_name"]).to(device)
    policy = GrootLitePolicy(
        backbone,
        state_dim=cfg["model"]["state_dim"], action_dim=cfg["model"]["action_dim"],
        horizon=cfg["model"]["action_horizon"], hidden_dim=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["layers"], heads=cfg["model"]["heads"],
    ).to(device)
    # Optimizer sees only trainable condition/action modules; the HF backbone remains frozen.
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"], betas=(0.95, 0.999))
    total = int(cfg["train"]["max_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_warmup(s, total, cfg["train"]["warmup_fraction"]))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    iterator = iter(dl)

    policy.train()
    for step in range(total):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dl); batch = next(iterator)
        state = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)
        # BONES Stage-A is text-only. Vision features are deliberately absent until synchronized camera demos exist.
        with torch.no_grad():
            text = backbone.encode_text(list(batch["text"]), device)
        loss = policy.loss(text, state, actions, image_features=None, action_mask=mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step()

        if step % 100 == 0:
            print(f"step={step:06d} loss={loss.item():.6f} lr={sched.get_last_lr()[0]:.3e}")
        if step and step % 5000 == 0:
            torch.save({
                "step": step,
                "model": {k: v for k, v in policy.state_dict().items() if not k.startswith("backbone.model.")},
                "optimizer": opt.state_dict(), "config": cfg,
            }, output / f"step_{step:07d}.pt")


if __name__ == "__main__":
    main()
