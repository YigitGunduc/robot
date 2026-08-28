from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .bones import NVIDIA_FILTER_KEYWORDS, should_filter_out


def filter_and_copy(
    source: str,
    dest: str,
    extra_keywords: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    src, dst = Path(source), Path(dest)
    stats = {"total": 0, "copied": 0, "filtered": 0}
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.name == "metadata.pkl":
            continue
        stats["total"] += 1
        rel = p.relative_to(src)
        if should_filter_out(str(rel), extra_keywords):
            stats["filtered"] += 1
            continue
        stats["copied"] += 1
        if not dry_run:
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--add-keywords", nargs="*", default=[])
    args = ap.parse_args()
    stats = filter_and_copy(args.source, args.dest, args.add_keywords, args.dry_run)
    print(stats)
    print("NVIDIA keywords:", ", ".join(NVIDIA_FILTER_KEYWORDS))


if __name__ == "__main__":
    main()
