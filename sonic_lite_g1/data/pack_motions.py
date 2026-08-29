from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REQUIRED = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def pack(files: list[Path], out: Path) -> None:
    if not files:
        raise ValueError("no motion files")
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in REQUIRED}
    starts: list[int] = []
    lengths: list[int] = []
    names: list[str] = []
    cursor = 0
    reference_shapes: dict[str, tuple[int, ...]] = {}

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in REQUIRED if k not in data]
            if missing:
                raise ValueError(f"{path}: missing {missing}")
            n = int(data["joint_pos"].shape[0])
            if n < 2:
                continue
            for key in REQUIRED:
                arr = np.asarray(data[key])
                if arr.shape[0] != n:
                    raise ValueError(f"{path}: inconsistent frame count for {key}")
                tail = arr.shape[1:]
                if key in reference_shapes and tail != reference_shapes[key]:
                    raise ValueError(
                        f"{path}: shape mismatch for {key}: {tail} vs {reference_shapes[key]}"
                    )
                reference_shapes.setdefault(key, tail)
                chunks[key].append(arr)

        starts.append(cursor)
        lengths.append(n)
        names.append(path.name)
        cursor += n

    if not names:
        raise ValueError("no valid motion files with at least 2 frames")
    payload = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    payload["clip_starts"] = np.asarray(starts, dtype=np.int64)
    payload["clip_lengths"] = np.asarray(lengths, dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps({"source_files": names, "num_clips": len(names), "num_frames": cursor}, indent=2)
    )
    print(f"packed {len(names)} clips / {cursor} frames -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=Path, help="directory containing mjlab-compatible NPZ files")
    src.add_argument("--list", type=Path, help="text file with one NPZ path per line")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.dir:
        files = sorted(args.dir.rglob("*.npz"))
    else:
        files = [Path(x.strip()).expanduser() for x in args.list.read_text().splitlines() if x.strip()]
    pack(files, args.out)


if __name__ == "__main__":
    main()
