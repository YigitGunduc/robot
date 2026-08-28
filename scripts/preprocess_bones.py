from __future__ import annotations

import argparse
from gear_sonic_mjx.data_process.bones import preprocess_bones_tree


def main() -> None:
    ap = argparse.ArgumentParser(description="Bones-SEED -> compact SONIC-MJX motion cache")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-fps", type=float, default=120.0)
    ap.add_argument("--fps", type=float, default=30.0, help="NVIDIA preprocessing uses 30 fps")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--add-keywords", nargs="*", default=[])
    args = ap.parse_args()
    stats = preprocess_bones_tree(
        args.input,
        args.output,
        source_fps=args.source_fps,
        preprocess_fps=args.fps,
        extra_filter_keywords=args.add_keywords,
        skip_filtered=not args.no_filter,
    )
    print(stats)


if __name__ == "__main__":
    main()
