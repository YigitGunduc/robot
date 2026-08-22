from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SonicRuntimeConfig:
    root: Path
    expected_ref: str
    input_type: str = "keyboard"
    target: str = "sim"


class SonicProcessAdapter:
    """Boundary around the official process-based SONIC deployment.

    SONIC is intentionally not reimplemented as a Python controller. Its C++ loop,
    TensorRT engines, checkpoint configuration, and transport remain one pinned unit.
    """

    def __init__(self, config: SonicRuntimeConfig) -> None:
        self.config = config

    @property
    def deploy_root(self) -> Path:
        return self.config.root / "gear_sonic_deploy"

    def validate(self, *, check_git_ref: bool = True) -> None:
        required = (
            self.config.root / "gear_sonic" / "scripts" / "run_sim_loop.py",
            self.deploy_root / "deploy.sh",
            self.deploy_root / "scripts" / "setup_env.sh",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("SONIC checkout is incomplete: " + ", ".join(missing))

        if check_git_ref:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.config.root,
                check=True,
                capture_output=True,
                text=True,
            )
            actual_ref = result.stdout.strip()
            if actual_ref != self.config.expected_ref:
                raise RuntimeError(
                    f"SONIC ref mismatch: expected {self.config.expected_ref}, got {actual_ref}"
                )

    def simulator_command(self) -> tuple[str, ...]:
        python = self.config.root / ".venv_sim" / "bin" / "python"
        return (
            str(python),
            str(self.config.root / "gear_sonic" / "scripts" / "run_sim_loop.py"),
        )

    def deployment_command(self, *extra_args: str) -> tuple[str, ...]:
        return (
            str(self.deploy_root / "deploy.sh"),
            "--input-type",
            self.config.input_type,
            *extra_args,
            self.config.target,
        )

    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.setdefault("TensorRT_ROOT", "/opt/TensorRT")
        return environment

