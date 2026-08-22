#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python -m pip install -e "$PROJECT_ROOT[sim]"
"$PROJECT_ROOT/scripts/fetch_mujoco_assets.sh"
cd "$PROJECT_ROOT"
MUJOCO_GL=egl g1-stack sim-smoke --steps 1000

