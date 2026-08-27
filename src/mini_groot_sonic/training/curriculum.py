from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from mini_groot_sonic.config import PPOConfig, RewardConfig, SimConfig, SonicTinyConfig
from mini_groot_sonic.data.curriculum import CurriculumStage, load_curriculum_manifest
from mini_groot_sonic.training.body_loop import train_body_controller
from mini_groot_sonic.training.utils import split_curriculum_motion_paths


@dataclass(frozen=True)
class PromotionCriteria:
    success_rate: float = 0.80
    mpjpe: float = 0.08
    root_position_error: float = 0.08
    root_orientation_error: float = 0.20
    minimum_evaluated_motions: int = 1


@dataclass
class CurriculumTrainState:
    current_stage: int = 0
    stage_start_iteration: int = 0
    consecutive_passes: int = 0
    status: str = "training"
    latest_checkpoint: str | None = None
    manifest_sha256: str | None = None
    history: list[dict] = field(default_factory=list)


def promotion_decision(
    metrics: dict[str, float],
    criteria: PromotionCriteria,
) -> tuple[bool, list[str]]:
    checks = (
        ("success_rate", ">=", criteria.success_rate),
        ("mpjpe", "<=", criteria.mpjpe),
        ("root_position_error", "<=", criteria.root_position_error),
        ("root_orientation_error", "<=", criteria.root_orientation_error),
        ("evaluated_motions", ">=", float(criteria.minimum_evaluated_motions)),
    )
    failures = []
    for name, operator, threshold in checks:
        value = metrics.get(name)
        passed = value is not None and (
            value >= threshold if operator == ">=" else value <= threshold
        )
        if not passed:
            observed = "missing" if value is None else f"{value:.5f}"
            failures.append(f"{name}={observed} (needs {operator} {threshold:.5f})")
    return not failures, failures


def latest_numbered_checkpoint(directory: str | Path) -> Path | None:
    paths = list(Path(directory).glob("body_[0-9]*.pt"))
    return (
        max(paths, key=lambda path: int(path.stem.rsplit("_", 1)[1]))
        if paths
        else None
    )


def _write_state(path: Path, state: CurriculumTrainState) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_state(path: Path) -> CurriculumTrainState:
    if not path.exists():
        return CurriculumTrainState()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CurriculumTrainState(**raw)


def _resolve_stages(
    manifest_path: str | Path,
    motion_dir_override: str | Path | None,
) -> tuple[list[CurriculumStage], list[list[Path]]]:
    manifest_motion_dir, stages = load_curriculum_manifest(manifest_path)
    motion_dir = Path(motion_dir_override) if motion_dir_override is not None else manifest_motion_dir
    resolved = [[motion_dir / filename for filename in stage.filenames] for stage in stages]
    missing = [str(path) for paths in resolved for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Curriculum references missing motions:\n{preview}")
    return stages, resolved


def train_dynamic_curriculum(
    manifest_path: str | Path,
    sim_cfg: SimConfig,
    sonic_cfg: SonicTinyConfig,
    reward_cfg: RewardConfig,
    ppo_cfg: PPOConfig,
    output_dir: str | Path,
    *,
    motion_dir_override: str | Path | None = None,
    validation_fraction: float = 0.15,
    evaluation_chunk_iterations: int = 100,
    minimum_stage_iterations: int = 1000,
    maximum_stage_iterations: int = 20_000,
    promotion_patience: int = 2,
    criteria: PromotionCriteria | None = None,
    randomization_start_stage: int = 3,
    resume_from: str | Path | None = None,
) -> CurriculumTrainState:
    if evaluation_chunk_iterations <= 0 or minimum_stage_iterations < 0:
        raise ValueError("Curriculum iteration intervals must be positive")
    if maximum_stage_iterations < max(1, minimum_stage_iterations):
        raise ValueError("maximum_stage_iterations must cover minimum_stage_iterations")
    if promotion_patience <= 0:
        raise ValueError("promotion_patience must be positive")
    criteria = criteria or PromotionCriteria()

    stages, stage_paths = _resolve_stages(manifest_path, motion_dir_override)
    splits = split_curriculum_motion_paths(stage_paths, validation_fraction, ppo_cfg.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "curriculum_state.json"
    state = _load_state(state_path)
    manifest_digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if state.manifest_sha256 is None:
        state.manifest_sha256 = manifest_digest
    elif state.manifest_sha256 != manifest_digest:
        raise ValueError(
            "Curriculum manifest changed for an existing run; use a new output directory"
        )
    if state.status == "complete" or state.current_stage >= len(stages):
        state.status = "complete"
        _write_state(state_path, state)
        print("Curriculum is already complete.", flush=True)
        return state

    transition_checkpoint = Path(resume_from) if resume_from is not None else None
    if state.latest_checkpoint is not None and Path(state.latest_checkpoint).exists():
        transition_checkpoint = Path(state.latest_checkpoint)

    while state.current_stage < len(stages):
        stage = stages[state.current_stage]
        training_paths, validation_paths = splits[state.current_stage]
        stage_dir = output_dir / f"stage_{stage.index}_{stage.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_checkpoint = latest_numbered_checkpoint(stage_dir)
        checkpoint = stage_checkpoint or transition_checkpoint
        transitioning = stage_checkpoint is None and checkpoint is not None
        if checkpoint is not None:
            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            next_iteration = int(checkpoint_data.get("iteration", -1)) + 1
        else:
            next_iteration = 0
        state.stage_start_iteration = min(state.stage_start_iteration, next_iteration)
        if stage_checkpoint is None and state.stage_start_iteration == 0 and next_iteration > 0:
            state.stage_start_iteration = next_iteration

        completed_in_stage = max(0, next_iteration - state.stage_start_iteration)
        print(
            f"Curriculum stage {stage.index}/{len(stages) - 1}: {stage.name} "
            f"({len(training_paths)} train, {len(validation_paths)} validation, "
            f"{completed_in_stage}/{maximum_stage_iterations} iterations)",
            flush=True,
        )
        if completed_in_stage >= maximum_stage_iterations:
            state.status = "stage_budget_exhausted"
            _write_state(state_path, state)
            print(
                f"Stage {stage.name} did not pass its promotion gate; stopping before the next stage.",
                flush=True,
            )
            return state

        chunk = min(evaluation_chunk_iterations, maximum_stage_iterations - completed_in_stage)
        target_iteration = next_iteration + chunk
        stage_sim = replace(
            sim_cfg,
            enable_randomization=(
                sim_cfg.enable_randomization and stage.index >= randomization_start_stage
            ),
        )
        train_body_controller(
            training_paths,
            stage_sim,
            sonic_cfg,
            reward_cfg,
            ppo_cfg,
            target_iteration,
            stage_dir,
            validation_paths=validation_paths,
            resume_from=checkpoint,
            reset_best_on_resume=transitioning,
        )
        checkpoint = latest_numbered_checkpoint(stage_dir)
        if checkpoint is None:
            raise RuntimeError(f"Training did not create a checkpoint in {stage_dir}")
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metrics = {
            key: float(value)
            for key, value in checkpoint_data.get("validation", {}).items()
        }
        completed_in_stage = int(checkpoint_data["iteration"]) + 1 - state.stage_start_iteration
        passed, failures = promotion_decision(metrics, criteria)
        eligible_to_promote = completed_in_stage >= minimum_stage_iterations and passed
        state.consecutive_passes = state.consecutive_passes + 1 if eligible_to_promote else 0
        state.latest_checkpoint = str(checkpoint)
        state.status = "training"
        state.history.append(
            {
                "stage": stage.index,
                "name": stage.name,
                "iteration": int(checkpoint_data["iteration"]),
                "stage_iterations": completed_in_stage,
                "metrics": metrics,
                "criteria": asdict(criteria),
                "training_motions": len(training_paths),
                "validation_motions": len(validation_paths),
                "randomization": stage_sim.enable_randomization,
                "gate_passed": passed,
                "eligible_to_promote": eligible_to_promote,
                "failures": failures,
                "consecutive_passes": state.consecutive_passes,
            }
        )
        _write_state(state_path, state)
        if failures:
            print("promotion_gate " + "; ".join(failures), flush=True)
        elif completed_in_stage < minimum_stage_iterations:
            print(
                f"promotion_gate metrics pass; waiting for {minimum_stage_iterations} minimum iterations",
                flush=True,
            )
        else:
            print(
                f"promotion_gate pass {state.consecutive_passes}/{promotion_patience}",
                flush=True,
            )

        if state.consecutive_passes >= promotion_patience:
            print(f"Promoting from {stage.name}.", flush=True)
            transition_checkpoint = checkpoint
            state.current_stage += 1
            state.stage_start_iteration = int(checkpoint_data["iteration"]) + 1
            state.consecutive_passes = 0
            state.status = "complete" if state.current_stage >= len(stages) else "training"
            _write_state(state_path, state)

    return state
