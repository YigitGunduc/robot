from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from gear_sonic_mjx.config import SonicConfig
from gear_sonic_mjx.data_process.annotations import load_seed_metadata
from gear_sonic_mjx.envs.g1_tracking_task import G1SonicTrackingTask
from gear_sonic_mjx.envs.motion_library import open_motion_library
from gear_sonic_mjx.sim.mjwarp_backend import MjWarpBatchSim
from gear_sonic_mjx.trl.modules.base_module import MLP
from gear_sonic_mjx.trl.modules.universal_token_modules import UniversalTokenModule
from gear_sonic_mjx.trl.trainer.ppo_trainer_aux_loss import SonicActorCritic

METRICS = (
    "mpjpe_global",
    "mpjpe_local",
    "mpjpe_aligned",
    "root_height_error",
    "root_orientation_error",
    "joint_position_error",
)


def _annotation_by_stem(path: str | None) -> dict[str, object]:
    if not path:
        return {}
    exact = load_seed_metadata(path)
    unique: dict[str, object] = {}
    duplicates: set[str] = set()
    for key, value in exact.items():
        stem = Path(key).stem
        if stem in unique:
            duplicates.add(stem)
        else:
            unique[stem] = value
    for stem in duplicates:
        unique.pop(stem, None)
    return unique


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "motions": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            summary[metric] = float(np.mean(values))
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    summary["per_category"] = {
        category: {
            "motions": len(items),
            "success_rate": float(np.mean([item["success"] for item in items])),
            "mpjpe_local": float(
                np.mean(
                    [item["mpjpe_local"] for item in items if "mpjpe_local" in item]
                )
            ),
        }
        for category, items in sorted(by_category.items())
    }
    summary["hardest_motions"] = [
        row["name"]
        for row in sorted(
            rows,
            key=lambda row: (bool(row["success"]), -float(row.get("mpjpe_local", 0.0))),
        )[:20]
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Held-out deterministic SONIC evaluation on MJWarp"
    )
    parser.add_argument("--mjcf", required=True)
    parser.add_argument(
        "--motions", required=True, help="held-out packed or NPZ motion library"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metadata", help="BONES metadata parquet/csv for per-category metrics"
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--min-success", type=float, default=0.95)
    parser.add_argument("--max-local-mpjpe-mm", type=float, default=40.0)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config")
    if not isinstance(cfg, SonicConfig):
        raise TypeError("checkpoint does not contain a SonicConfig")
    motions = open_motion_library(args.motions, cfg.motion.target_fps)
    cfg.num_envs = min(args.num_envs, len(motions))
    cfg.observation_noise.enabled = False
    cfg.motion.freeze_frame_aug = False
    cfg.domain_randomization = {}
    cfg.validate()

    sim = MjWarpBatchSim(
        args.mjcf,
        cfg.num_envs,
        cfg.sim.sim_dt,
        cfg.sim.nconmax,
        cfg.sim.njmax,
        cfg.sim.naconmax,
    )
    task = G1SonicTrackingTask(
        sim,
        motions,
        cfg,
        require_fk_cache=True,
        auto_reset=False,
        enforce_episode_length=False,
    )
    token = UniversalTokenModule(
        cfg.model, cfg.motion.num_future_frames, cfg.motion.actor_prop_history_length
    ).to(sim.device)
    critic = MLP(task.critic_dim, cfg.model.critic_hidden, 1).to(sim.device)
    model = SonicActorCritic(
        token,
        critic,
        cfg.model.dof,
        cfg.ppo.init_noise_std,
        cfg.ppo.std_clamp_min,
        cfg.ppo.std_clamp_max,
    ).to(sim.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    annotations = _annotation_by_stem(args.metadata)
    rows: list[dict[str, object]] = []
    all_lengths = motions.lengths.to(sim.device)
    for batch_start in range(0, len(motions), cfg.num_envs):
        actual = min(cfg.num_envs, len(motions) - batch_start)
        ids = torch.arange(batch_start, batch_start + actual, device=sim.device)
        if actual < cfg.num_envs:
            ids = torch.cat([ids, ids[-1:].expand(cfg.num_envs - actual)])
        enc, prop, _ = task.reset_to(ids)
        active = torch.arange(cfg.num_envs, device=sim.device) < actual
        sums = {name: torch.zeros(cfg.num_envs, device=sim.device) for name in METRICS}
        counts = torch.zeros(cfg.num_envs, device=sim.device)
        failed = torch.zeros(cfg.num_envs, dtype=torch.bool, device=sim.device)
        max_steps = int(all_lengths[ids].max().item()) + 1

        for _ in range(max_steps):
            with torch.no_grad():
                action, _, _ = model.act(enc, prop, deterministic=True)
                step = task.step(action)
            for name in METRICS:
                if name in step.info:
                    sums[name][active] += step.info[name][active]
            counts[active] += 1
            just_done = active & step.done
            failed[just_done] = step.info["failed"][just_done].bool()
            active &= ~step.done
            enc, prop = step.encoder_obs, step.proprio_obs
            if not active.any():
                break
            # Finished worlds still participate in every batched MJWarp step. Re-seed them at
            # their terminal reference pose each iteration so a short clip cannot fall/overflow
            # while longer clips in the same batch are still being evaluated.
            inactive_ids = (~active).nonzero(as_tuple=False).squeeze(-1)
            if inactive_ids.numel():
                terminal_frames = all_lengths[ids[inactive_ids]] - 1
                enc, prop, _ = task.reset_to(
                    ids[inactive_ids], terminal_frames, env_ids=inactive_ids
                )
        if active.any():
            raise RuntimeError(
                "evaluation exceeded the longest motion without termination"
            )

        for local in range(actual):
            motion_id = batch_start + local
            clip = motions._load(motion_id)
            ann = annotations.get(Path(clip.name).stem)
            category = getattr(ann, "category", None) or clip.name.split("_")[0]
            row: dict[str, object] = {
                "motion_id": motion_id,
                "name": clip.name,
                "category": category,
                "success": not bool(failed[local].item()),
                "evaluated_steps": int(counts[local].item()),
            }
            for name in METRICS:
                row[name] = float(
                    (sums[name][local] / counts[local].clamp_min(1)).item()
                )
            rows.append(row)
        print(
            f"evaluated {min(batch_start + actual, len(motions))}/{len(motions)} motions"
        )

    summary = _summary(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "motions": rows}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    passed = (
        float(summary["success_rate"]) >= args.min_success
        and float(summary["mpjpe_local"]) * 1000.0 <= args.max_local_mpjpe_mm
    )
    print(f"evaluation gates: {'PASS' if passed else 'FAIL'}")
    if args.fail_on_gate and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
