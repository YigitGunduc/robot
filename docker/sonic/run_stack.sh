#!/usr/bin/env bash
set -euo pipefail

SONIC_ROOT="${SONIC_ROOT:-/opt/GR00T-WholeBodyControl}"
DEPLOY_ROOT="$SONIC_ROOT/gear_sonic_deploy"
SIM_PYTHON="$SONIC_ROOT/.venv_sim/bin/python"
SIM_SCRIPT="$SONIC_ROOT/gear_sonic/scripts/run_sim_loop.py"
STARTUP_SECONDS="${SONIC_SIM_STARTUP_SECONDS:-5}"
INPUT_TYPE="${SONIC_INPUT_TYPE:-keyboard}"
ZMQ_HOST="${SONIC_ZMQ_HOST:-localhost}"

sim_pid=""

cleanup() {
  if [[ -n "$sim_pid" ]] && kill -0 "$sim_pid" 2>/dev/null; then
    kill -TERM -- "-$sim_pid" 2>/dev/null || true
    wait "$sim_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

sim_args=()
if [[ "${SONIC_HEADLESS:-0}" == "1" ]]; then
  sim_args+=(--no-enable-onscreen --enable-offscreen)
fi

echo "starting_sonic_sim headless=${SONIC_HEADLESS:-0}"
setsid "$SIM_PYTHON" "$SIM_SCRIPT" "${sim_args[@]}" &
sim_pid=$!

for ((second = 1; second <= STARTUP_SECONDS; second++)); do
  if ! kill -0 "$sim_pid" 2>/dev/null; then
    wait "$sim_pid"
    echo "SONIC simulator exited during startup" >&2
    exit 4
  fi
  sleep 1
done

echo "starting_sonic_controller input_type=$INPUT_TYPE zmq_host=$ZMQ_HOST"
echo "The pinned official launcher asks for confirmation; press Enter to continue."
cd "$DEPLOY_ROOT"
./deploy.sh --input-type "$INPUT_TYPE" --zmq-host "$ZMQ_HOST" "$@" sim
