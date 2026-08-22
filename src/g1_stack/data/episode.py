from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from g1_stack.core.types import (
    ActuatorCommand,
    MissionRequest,
    PhysicalIntent,
    RobotState,
    SafetyDecision,
    WholeBodyReference,
)


@dataclass(frozen=True, slots=True)
class LoadedEpisode:
    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


class EpisodeRecorder:
    """Append-only episode recorder producing a JSON manifest and compressed NumPy data."""

    def __init__(self, output_root: Path, *, label: str = "scripted") -> None:
        self.output_root = output_root.expanduser().resolve()
        self.label = _safe_label(label)
        self.path: Path | None = None
        self._manifest: dict[str, Any] | None = None
        self._rows: dict[str, list[Any]] = {}

    def start(
        self,
        state: RobotState,
        request: MissionRequest,
        intent: PhysicalIntent,
        *,
        configuration: dict[str, Any],
    ) -> Path:
        if self.path is not None:
            raise RuntimeError("EpisodeRecorder.start() can only be called once")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = self.output_root / f"{timestamp}-{self.label}-{uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self._manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "actuator_names": list(state.actuator_names),
            "mission_request": asdict(request),
            "intent": asdict(intent),
            "configuration": configuration,
            "events": [],
        }
        self._rows = {
            "time_s": [],
            "qpos": [],
            "qvel": [],
            "actuator_positions": [],
            "actuator_velocities": [],
            "actuator_forces": [],
            "base_position": [],
            "base_quaternion_wxyz": [],
            "contact_count": [],
            "reference_joint_positions": [],
            "reference_base_linear_velocity_body_m_s": [],
            "reference_base_angular_velocity_body_rad_s": [],
            "requested_command": [],
            "applied_command": [],
            "limited": [],
            "stopped": [],
        }
        return self.path

    def record(
        self,
        state: RobotState,
        reference: WholeBodyReference,
        requested: ActuatorCommand,
        decision: SafetyDecision,
    ) -> None:
        self._require_started()
        if (
            reference.joint_names != state.actuator_names
            or requested.names != state.actuator_names
            or decision.command.names != state.actuator_names
        ):
            raise ValueError("Recorded commands must match state actuator order")
        self._rows["time_s"].append(state.time_s)
        self._rows["qpos"].append(state.qpos)
        self._rows["qvel"].append(state.qvel)
        self._rows["actuator_positions"].append(state.actuator_positions)
        self._rows["actuator_velocities"].append(state.actuator_velocities)
        self._rows["actuator_forces"].append(state.actuator_forces)
        self._rows["base_position"].append(state.base_position)
        self._rows["base_quaternion_wxyz"].append(state.base_quaternion_wxyz)
        self._rows["contact_count"].append(state.contact_count)
        self._rows["reference_joint_positions"].append(reference.joint_position_targets)
        self._rows["reference_base_linear_velocity_body_m_s"].append(
            reference.base_linear_velocity_body_m_s
        )
        self._rows["reference_base_angular_velocity_body_rad_s"].append(
            reference.base_angular_velocity_body_rad_s
        )
        self._rows["requested_command"].append(requested.values)
        self._rows["applied_command"].append(decision.command.values)
        self._rows["limited"].append(bool(decision.limited_actuators))
        self._rows["stopped"].append(decision.stopped)

    def event(self, kind: str, **details: Any) -> None:
        manifest = self._require_started()
        manifest["events"].append(
            {"kind": kind, "recorded_at": datetime.now(UTC).isoformat(), **details}
        )

    def close(self, *, success: bool, stop_reason: str, steps: int) -> Path:
        manifest = self._require_started()
        assert self.path is not None
        manifest.update({"success": success, "stop_reason": stop_reason, "steps": steps})
        arrays = {name: np.asarray(values) for name, values in self._rows.items()}
        np.savez_compressed(self.path / "episode.npz", **arrays)
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.path

    def _require_started(self) -> dict[str, Any]:
        if self._manifest is None:
            raise RuntimeError("EpisodeRecorder.start() must be called first")
        return self._manifest


def load_episode(path: Path) -> LoadedEpisode:
    episode_path = path.expanduser().resolve()
    manifest_path = episode_path / "manifest.json"
    data_path = episode_path / "episode.npz"
    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Not an episode directory: {episode_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return LoadedEpisode(path=episode_path, manifest=manifest, arrays=arrays)


def _safe_label(value: str) -> str:
    label = "".join(
        character if character.isalnum() or character in "-_" else "-" for character in value
    )
    return label.strip("-") or "episode"
