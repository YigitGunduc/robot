from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gear_sonic_mjx.data_process.bones import (
    MotionClip,
    resample_motion,
    resampled_frame_count,
)


@dataclass
class MotionBatch:
    motion_id: torch.Tensor
    frame: torch.Tensor
    root_pos: torch.Tensor
    root_quat_wxyz: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class BonesMotionLibrary:
    """Lazy BONES-SEED motion library with 50 Hz runtime sampling.

    NVIDIA preprocesses public CSVs at 30 Hz and configures its motion library for a 50 Hz target.
    We preserve the same two-rate contract: cached files can be 30 Hz, while clips are resampled to
    the controller target rate when cached in memory.
    """

    def __init__(
        self, root: str | Path, target_fps: float = 50.0, cache_size: int = 256
    ):
        self.root = Path(root)
        self.target_fps = float(target_fps)
        self.cache_size = int(cache_size)
        self._cache: dict[int, MotionClip] = {}
        self._cache_order: list[int] = []
        manifest = self.root / "_manifest.npz"
        if manifest.exists():
            d = np.load(manifest, allow_pickle=False)
            self.files = [self.root / str(x) for x in d["relpaths"].tolist()]
            src_frames = d["num_frames"].astype(np.int64)
            src_fps = d["fps"].astype(np.float64)
            target_frames = np.asarray(
                [
                    resampled_frame_count(int(n), float(fps), self.target_fps)
                    for n, fps in zip(src_frames, src_fps, strict=True)
                ],
                dtype=np.int64,
            )
            self.lengths = torch.from_numpy(target_frames).long()
        else:
            self.files = sorted(
                p for p in self.root.rglob("*.npz") if p.name != "_manifest.npz"
            )
            if not self.files:
                raise FileNotFoundError(f"No .npz motions found under {self.root}")
            # Backward-compatible one-time metadata scan for caches created before manifest support.
            self.lengths = torch.tensor(
                [self._load(i).num_frames for i in range(len(self.files))],
                dtype=torch.long,
            )
            self._cache.clear()
            self._cache_order.clear()
        if not self.files:
            raise FileNotFoundError(f"No motion files found under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, idx: int) -> MotionClip:
        if idx in self._cache:
            return self._cache[idx]
        clip = MotionClip.load_npz(self.files[idx])
        clip = resample_motion(clip, self.target_fps)
        self._cache[idx] = clip
        self._cache_order.append(idx)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return clip

    def sample_future_numpy(
        self, motion_id: int, frame: int, num_future: int = 10, dt: float = 0.1
    ) -> dict[str, np.ndarray]:
        clip = self._load(int(motion_id))
        stride = max(1, round(dt * clip.fps))
        idx = np.minimum(frame + np.arange(num_future) * stride, clip.num_frames - 1)
        return {
            "root_pos": clip.root_pos[idx],
            "root_quat_wxyz": clip.root_quat_wxyz[idx],
            "joint_pos": clip.joint_pos[idx],
            "joint_vel": clip.joint_vel[idx],
        }

    def batch_future(
        self,
        motion_ids: torch.Tensor,
        frames: torch.Tensor,
        num_future: int,
        dt: float,
        device: torch.device | str,
        frame_cap: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        caps = [None] * len(motion_ids) if frame_cap is None else frame_cap.tolist()
        rows = []
        for m, f, cap in zip(motion_ids.tolist(), frames.tolist(), caps):
            clip = self._load(int(m))
            stride = max(1, round(dt * clip.fps))
            idx = f + np.arange(num_future) * stride
            if cap is not None:
                idx = np.minimum(idx, int(cap))
            idx = np.minimum(idx, clip.num_frames - 1)
            rows.append(
                {
                    "root_pos": clip.root_pos[idx],
                    "root_quat_wxyz": clip.root_quat_wxyz[idx],
                    "joint_pos": clip.joint_pos[idx],
                    "joint_vel": clip.joint_vel[idx],
                }
            )
        return {
            k: torch.as_tensor(
                np.stack([r[k] for r in rows]), dtype=torch.float32, device=device
            )
            for k in rows[0]
        }

    def batch_current(
        self, motion_ids: torch.Tensor, frames: torch.Tensor, device: torch.device | str
    ) -> dict[str, torch.Tensor | None]:
        rows = []
        for m, f in zip(motion_ids.tolist(), frames.tolist()):
            clip = self._load(int(m))
            i = min(int(f), clip.num_frames - 1)
            rows.append((clip, i))
        out: dict[str, torch.Tensor | None] = {
            "root_pos": torch.as_tensor(
                np.stack([c.root_pos[i] for c, i in rows]),
                dtype=torch.float32,
                device=device,
            ),
            "root_quat_wxyz": torch.as_tensor(
                np.stack([c.root_quat_wxyz[i] for c, i in rows]),
                dtype=torch.float32,
                device=device,
            ),
            "joint_pos": torch.as_tensor(
                np.stack([c.joint_pos[i] for c, i in rows]),
                dtype=torch.float32,
                device=device,
            ),
            "joint_vel": torch.as_tensor(
                np.stack([c.joint_vel[i] for c, i in rows]),
                dtype=torch.float32,
                device=device,
            ),
        }
        for key in ["body_pos", "body_quat_wxyz", "body_linvel", "body_angvel"]:
            if all(getattr(c, key) is not None for c, _ in rows):
                out[key] = torch.as_tensor(
                    np.stack([getattr(c, key)[i] for c, i in rows]),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                out[key] = None
        return out

    def upper_body_mix(
        self,
        q_future: torch.Tensor,
        qd_future: torch.Tensor,
        p: float = 0.5,
        upper_start: int = 12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SONIC upper-body recombination using one shared permutation for q and qd."""
        if p <= 0 or q_future.shape[0] < 2:
            return q_future, qd_future
        b = q_future.shape[0]
        mask = torch.rand(b, device=q_future.device) < p
        perm = torch.randperm(b, device=q_future.device)
        q_out, qd_out = q_future.clone(), qd_future.clone()
        q_out[mask, :, upper_start:] = q_future[perm[mask], :, upper_start:]
        qd_out[mask, :, upper_start:] = qd_future[perm[mask], :, upper_start:]
        return q_out, qd_out

    def freeze_frame(
        self, q: torch.Tensor, qd: torch.Tensor, probability: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NVIDIA TrackingCommandCfg exposes freeze_frame_aug_prob=0.1.
        if probability <= 0:
            return q, qd
        mask = torch.rand(q.shape[0], device=q.device) < probability
        if mask.any():
            q = q.clone()
            qd = qd.clone()
            q[mask] = q[mask, :1].expand(-1, q.shape[1], -1)
            qd[mask] = 0.0
        return q, qd


class PackedBonesMotionLibrary:
    """Vectorized memory-mapped BONES library for high-throughput GPU PPO.

    Build with ``scripts/pack_bones_mmap.py`` after FK augmentation. All clips are resampled once
    to the controller target FPS and concatenated into `.npy` arrays. Runtime sampling then uses
    vectorized integer gathers instead of opening/decompressing thousands of NPZs every policy step.

    The packed arrays stay on host storage/OS page cache and only the requested batch is copied to
    the policy/simulator device, so the full ~288 h public set does not have to fit in VRAM.
    """

    META = "_packed_metadata.npz"

    def __init__(self, root: str | Path, target_fps: float = 50.0):
        self.root = Path(root)
        meta_path = self.root / self.META
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        meta = np.load(meta_path, allow_pickle=False)
        self.target_fps = float(meta["fps"].item())
        if abs(self.target_fps - float(target_fps)) > 1e-6:
            raise ValueError(f"packed FPS is {self.target_fps}, requested {target_fps}")
        self.lengths = torch.from_numpy(meta["lengths"].astype(np.int64)).long()
        self.offsets_np = meta["offsets"].astype(np.int64)
        self.body_names = (
            tuple(str(x) for x in meta["body_names"].tolist())
            if "body_names" in meta and len(meta["body_names"])
            else None
        )
        self.names = (
            [str(x) for x in meta["names"].tolist()]
            if "names" in meta
            else [str(i) for i in range(len(self.lengths))]
        )
        self.source_relpaths = (
            [str(x) for x in meta["source_relpaths"].tolist()]
            if "source_relpaths" in meta
            else [f"{index:08d}.npz" for index in range(len(self.lengths))]
        )
        self.arrays: dict[str, np.ndarray] = {}
        for key in [
            "root_pos",
            "root_quat_wxyz",
            "joint_pos",
            "joint_vel",
            "body_pos",
            "body_quat_wxyz",
            "body_linvel",
            "body_angvel",
        ]:
            path = self.root / f"{key}.npy"
            if path.exists():
                self.arrays[key] = np.load(path, mmap_mode="r")
        for key in ["root_pos", "root_quat_wxyz", "joint_pos", "joint_vel"]:
            if key not in self.arrays:
                raise FileNotFoundError(self.root / f"{key}.npy")

    def __len__(self) -> int:
        return int(self.lengths.numel())

    def _global_current_indices(
        self, motion_ids: torch.Tensor, frames: torch.Tensor
    ) -> np.ndarray:
        m = motion_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        f = frames.detach().cpu().numpy().astype(np.int64, copy=False)
        lens = self.lengths.numpy()[m]
        f = np.minimum(np.maximum(f, 0), lens - 1)
        return self.offsets_np[m] + f

    def _to_device(self, x: np.ndarray, device: torch.device | str) -> torch.Tensor:
        # np.asarray makes fancy-index results regular contiguous arrays and avoids a read-only
        # memmap warning in torch.from_numpy.
        return torch.from_numpy(np.ascontiguousarray(x)).to(
            device=device, dtype=torch.float32
        )

    def batch_future(
        self,
        motion_ids: torch.Tensor,
        frames: torch.Tensor,
        num_future: int,
        dt: float,
        device: torch.device | str,
        frame_cap: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        m = motion_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        f = frames.detach().cpu().numpy().astype(np.int64, copy=False)
        stride = max(1, round(float(dt) * self.target_fps))
        rel = f[:, None] + np.arange(num_future, dtype=np.int64)[None] * stride
        if frame_cap is not None:
            caps = (
                frame_cap.detach().cpu().numpy().astype(np.int64, copy=False)[:, None]
            )
            rel = np.minimum(rel, caps)
        lens = self.lengths.numpy()[m][:, None]
        rel = np.minimum(np.maximum(rel, 0), lens - 1)
        idx = self.offsets_np[m][:, None] + rel
        return {
            key: self._to_device(self.arrays[key][idx], device)
            for key in ["root_pos", "root_quat_wxyz", "joint_pos", "joint_vel"]
        }

    def batch_current(
        self, motion_ids: torch.Tensor, frames: torch.Tensor, device: torch.device | str
    ) -> dict[str, torch.Tensor | None]:
        idx = self._global_current_indices(motion_ids, frames)
        out: dict[str, torch.Tensor | None] = {}
        for key in ["root_pos", "root_quat_wxyz", "joint_pos", "joint_vel"]:
            out[key] = self._to_device(self.arrays[key][idx], device)
        for key in ["body_pos", "body_quat_wxyz", "body_linvel", "body_angvel"]:
            out[key] = (
                self._to_device(self.arrays[key][idx], device)
                if key in self.arrays
                else None
            )
        return out

    def _load(self, idx: int) -> MotionClip:
        """Compatibility probe/small-clip path. PPO sampling does not use this method."""
        start = int(self.offsets_np[idx])
        n = int(self.lengths[idx])
        sl = slice(start, start + n)
        kwargs = {k: np.asarray(v[sl]) for k, v in self.arrays.items()}
        return MotionClip(
            name=self.names[idx],
            fps=self.target_fps,
            root_pos=kwargs["root_pos"],
            root_quat_wxyz=kwargs["root_quat_wxyz"],
            joint_pos=kwargs["joint_pos"],
            joint_vel=kwargs["joint_vel"],
            body_names=self.body_names,
            body_pos=kwargs.get("body_pos"),
            body_quat_wxyz=kwargs.get("body_quat_wxyz"),
            body_linvel=kwargs.get("body_linvel"),
            body_angvel=kwargs.get("body_angvel"),
        )

    def upper_body_mix(
        self,
        q_future: torch.Tensor,
        qd_future: torch.Tensor,
        p: float = 0.5,
        upper_start: int = 12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if p <= 0 or q_future.shape[0] < 2:
            return q_future, qd_future
        b = q_future.shape[0]
        mask = torch.rand(b, device=q_future.device) < p
        perm = torch.randperm(b, device=q_future.device)
        q_out, qd_out = q_future.clone(), qd_future.clone()
        q_out[mask, :, upper_start:] = q_future[perm[mask], :, upper_start:]
        qd_out[mask, :, upper_start:] = qd_future[perm[mask], :, upper_start:]
        return q_out, qd_out

    def freeze_frame(
        self, q: torch.Tensor, qd: torch.Tensor, probability: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if probability <= 0:
            return q, qd
        mask = torch.rand(q.shape[0], device=q.device) < probability
        if mask.any():
            q = q.clone()
            qd = qd.clone()
            q[mask] = q[mask, :1].expand(-1, q.shape[1], -1)
            qd[mask] = 0.0
        return q, qd


def open_motion_library(
    root: str | Path, target_fps: float = 50.0, cache_size: int = 256
) -> BonesMotionLibrary | PackedBonesMotionLibrary:
    root = Path(root)
    if (root / PackedBonesMotionLibrary.META).exists():
        return PackedBonesMotionLibrary(root, target_fps)
    return BonesMotionLibrary(root, target_fps, cache_size)
