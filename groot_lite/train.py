from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from gear_sonic_mjx.checkpoint_utils import capture_rng_state, restore_rng_state
from groot_lite.dataset import TokenTrajectoryDataset
from groot_lite.model import FrozenSiglip2Backbone, GrootLitePolicy


def cosine_warmup(step: int, total: int, warmup_fraction: float) -> float:
    warm = max(1, int(total * warmup_fraction))
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(total - warm, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _head_state(policy: GrootLitePolicy) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in policy.state_dict().items()
        if not key.startswith("backbone.model.")
    }


def _save_checkpoint(path, step, policy, optimizer, scheduler, config) -> None:
    payload = {
        "training_state_version": 2,
        "step": step,
        "model": _head_state(policy),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
        "config": config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def _validate(policy, backbone, loader, device, batches: int = 8) -> float:
    policy.eval()
    losses = []
    devices = [device.index or 0] if device.type == "cuda" else []
    # Flow matching is stochastic. A fixed validation RNG makes checkpoints comparable.
    python_rng = random.getstate()
    try:
        random.seed(0)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            for index, batch in enumerate(loader):
                if index >= batches:
                    break
                state = batch["state"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)
                mask = batch["action_mask"].to(device, non_blocking=True)
                text = backbone.encode_text(list(batch["text"]), device)
                losses.append(
                    float(policy.loss(text, state, actions, action_mask=mask).item())
                )
    finally:
        random.setstate(python_rng)
    policy.train()
    if not losses:
        raise RuntimeError("validation loader produced no batches")
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train compact GR00T-like flow policy on frozen SONIC token trajectories"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--validation-data")
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).parents[1] / "gear_sonic_mjx/config/groot_lite.yaml"
        ),
    )
    parser.add_argument("--output", default="runs/groot_lite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
    total = int(config["train"]["max_steps"])
    if total <= 0 or args.save_interval <= 0 or args.eval_interval <= 0:
        raise ValueError(
            "training steps and save/evaluation intervals must be positive"
        )
    _seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = TokenTrajectoryDataset(args.data, config["model"]["action_horizon"])
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("training set is smaller than one drop-last batch")
    validation_loader = None
    if args.validation_data:
        validation_dataset = TokenTrajectoryDataset(
            args.validation_data, config["model"]["action_horizon"]
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    backbone = FrozenSiglip2Backbone(config["backbone"]["model_name"]).to(device)
    policy = GrootLitePolicy(
        backbone,
        state_dim=config["model"]["state_dim"],
        action_dim=config["model"]["action_dim"],
        horizon=config["model"]["action_horizon"],
        hidden_dim=config["model"]["hidden_dim"],
        layers=config["model"]["layers"],
        heads=config["model"]["heads"],
    ).to(device)
    trainable = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup(step, total, config["train"]["warmup_fraction"]),
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = policy.load_state_dict(checkpoint["model"], strict=False)
        invalid_missing = [
            key for key in missing if not key.startswith("backbone.model.")
        ]
        if invalid_missing or unexpected:
            raise ValueError(
                f"checkpoint/model mismatch: missing={invalid_missing}, unexpected={unexpected}"
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "rng" in checkpoint:
            restore_rng_state(checkpoint["rng"])
        start_step = int(checkpoint["step"]) + 1

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    policy.train()
    last_step = start_step - 1
    for step in range(start_step, total):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        state = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        mask = batch["action_mask"].to(device, non_blocking=True)
        with torch.no_grad():
            text = backbone.encode_text(list(batch["text"]), device)
        loss = policy.loss(text, state, actions, action_mask=mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite GR00T loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite GR00T gradient at step {step}")
        optimizer.step()
        scheduler.step()
        last_step = step

        if step % 100 == 0:
            print(
                f"step={step:06d} train_loss={loss.item():.6f} "
                f"lr={scheduler.get_last_lr()[0]:.3e}"
            )
        if validation_loader is not None and step and step % args.eval_interval == 0:
            validation_loss = _validate(policy, backbone, validation_loader, device)
            print(f"step={step:06d} validation_loss={validation_loss:.6f}")
        if step and step % args.save_interval == 0:
            _save_checkpoint(
                output / f"step_{step:07d}.pt",
                step,
                policy,
                optimizer,
                scheduler,
                config,
            )

    if last_step >= start_step:
        final_path = output / f"step_{last_step:07d}.pt"
        _save_checkpoint(final_path, last_step, policy, optimizer, scheduler, config)
        print(f"saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
