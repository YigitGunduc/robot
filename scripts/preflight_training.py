from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gear_sonic_mjx.config import SonicConfig
from gear_sonic_mjx.preflight import (
    PreflightReport,
    validate_mjcf,
    validate_motion_library,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast SONIC config, G1 MJCF, joint-order, timing, and FK preflight"
    )
    parser.add_argument("--mjcf", required=True)
    parser.add_argument(
        "--motions", help="preprocessed/FK-augmented or packed motion library"
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).parents[1] / "gear_sonic_mjx/config/sonic_release_mjx.yaml"
        ),
    )
    parser.add_argument("--output", default="preflight_report.json")
    parser.add_argument("--max-clips", type=int, default=100)
    args = parser.parse_args()

    report = PreflightReport()
    try:
        cfg = SonicConfig.from_yaml(args.config)
        report.checks["clock"] = {
            "physics_hz": 1.0 / cfg.sim.sim_dt,
            "policy_hz": 1.0 / cfg.sim.policy_dt,
            "motion_hz": cfg.motion.target_fps,
            "future_stride_frames": round(
                cfg.motion.dt_future_ref_frames * cfg.motion.target_fps
            ),
        }
    except Exception as exc:  # noqa: BLE001 - preflight must serialize config failures
        report.error(f"configuration validation failed: {exc}")
        cfg = None
    if cfg is not None:
        model = validate_mjcf(args.mjcf, cfg, report)
        if model is not None and args.motions:
            validate_motion_library(
                args.motions,
                args.mjcf,
                cfg,
                report,
                max_clips=args.max_clips,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({**asdict(report), "passed": report.passed}, indent=2) + "\n"
    )
    print(output.read_text())
    if not report.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
