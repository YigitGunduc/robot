from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from gear_sonic_mjx.config import SonicConfig
from gear_sonic_mjx.sim.mjwarp_backend import MjWarpBatchSim


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MJWarp physics throughput and capacity for a G1 MJCF"
    )
    parser.add_argument("--mjcf", required=True)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).parents[1] / "gear_sonic_mjx/config/sonic_release_mjx.yaml"
        ),
    )
    parser.add_argument(
        "--worlds", nargs="+", type=int, default=[256, 1024, 2048, 4096]
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="mjwarp_benchmark.json")
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.steps <= 0:
        raise ValueError(
            "warmup steps must be non-negative and measured steps positive"
        )

    cfg = SonicConfig.from_yaml(args.config)
    results = []
    for worlds in args.worlds:
        row: dict[str, object] = {"worlds": worlds}
        try:
            sim = MjWarpBatchSim(
                args.mjcf,
                worlds,
                cfg.sim.sim_dt,
                cfg.sim.nconmax,
                cfg.sim.njmax,
                cfg.sim.naconmax,
            )
            sim.ctrl.zero_()
            sim.step(args.warmup_steps)
            sim.assert_no_overflow()
            _synchronize(sim.device)
            started = time.perf_counter()
            sim.step(args.steps)
            _synchronize(sim.device)
            elapsed = time.perf_counter() - started
            sim.assert_no_overflow()
            row.update(
                {
                    "passed": True,
                    "device": str(sim.device),
                    "seconds": elapsed,
                    "physics_world_steps_per_second": worlds * args.steps / elapsed,
                    "realtime_factor_per_world": args.steps * cfg.sim.sim_dt / elapsed,
                }
            )
        except Exception as exc:  # noqa: BLE001 - benchmark records OOM/backend failures
            row.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
        results.append(row)
        print(json.dumps(row, indent=2))
        if "sim" in locals():
            del sim
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    passed = [row for row in results if row.get("passed")]
    if not passed:
        raise RuntimeError("no requested MJWarp world count passed")
    best = max(passed, key=lambda row: row["physics_world_steps_per_second"])
    report = {
        "mjcf": str(Path(args.mjcf).resolve()),
        "sim_dt": cfg.sim.sim_dt,
        "measured_steps": args.steps,
        "recommended_worlds_for_physics_throughput": best["worlds"],
        "results": results,
        "note": "Confirm end-to-end PPO memory/throughput at this scale before a full run.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
