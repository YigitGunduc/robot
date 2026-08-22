#!/usr/bin/env bash
set -euo pipefail

SONIC_ROOT="${SONIC_ROOT:-/opt/GR00T-WholeBodyControl}"
DEPLOY_ROOT="$SONIC_ROOT/gear_sonic_deploy"
APP_ROOT="${APP_ROOT:-/workspace/g1-stack}"

source "$DEPLOY_ROOT/scripts/setup_env.sh" >/dev/null

preflight() {
  local actual_ref
  actual_ref="$(git -C "$SONIC_ROOT" rev-parse HEAD)"
  test -n "$actual_ref"
  if [[ -n "${SONIC_REF:-}" && "$actual_ref" != "$SONIC_REF" ]]; then
    echo "Expected SONIC ref $SONIC_REF, found $actual_ref" >&2
    exit 3
  fi
  test -x "$DEPLOY_ROOT/deploy.sh"
  test -f "$TensorRT_ROOT/include/NvInferVersion.h"
  test -x /usr/local/bin/sonic-stack
  command -v g1-stack >/dev/null
  test -f "$DEPLOY_ROOT/policy/release/model_decoder.onnx"
  test -f "$DEPLOY_ROOT/policy/release/model_encoder.onnx"
  test -f "$DEPLOY_ROOT/policy/release/observation_config.yaml"
  test -f "$DEPLOY_ROOT/planner/target_vel/V2/planner_sonic.onnx"
  test -s /opt/sonic-model.sha256

  local trt_major trt_minor
  trt_major="$(awk '/#define NV_TENSORRT_MAJOR/ {print $3}' "$TensorRT_ROOT/include/NvInferVersion.h")"
  trt_minor="$(awk '/#define NV_TENSORRT_MINOR/ {print $3}' "$TensorRT_ROOT/include/NvInferVersion.h")"
  if [[ "$trt_major.$trt_minor" != "10.13" ]]; then
    echo "Expected TensorRT 10.13, found $trt_major.$trt_minor" >&2
    exit 3
  fi

  echo "sonic_preflight_ok"
  echo "sonic_ref=$actual_ref"
  echo "tensorrt=$trt_major.$trt_minor"
  echo "model_manifest=/opt/sonic-model.sha256"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  elif [[ "${SONIC_REQUIRE_GPU:-1}" == "1" ]]; then
    echo "NVIDIA GPU runtime is required but was not attached" >&2
    exit 3
  else
    echo "warning: GPU check skipped; inference cannot run" >&2
  fi

  python -c "import g1_stack; print('g1_stack=' + g1_stack.__version__)"
}

command="${1:-preflight}"
shift || true

case "$command" in
  app|stack)
    preflight
    exec /usr/local/bin/sonic-stack "$@"
    ;;
  preflight)
    preflight
    ;;
  shell)
    preflight
    exec bash "$@"
    ;;
  official-sim)
    exec "$SONIC_ROOT/.venv_sim/bin/python" \
      "$SONIC_ROOT/gear_sonic/scripts/run_sim_loop.py" "$@"
    ;;
  sonic-deploy)
    preflight
    cd "$DEPLOY_ROOT"
    exec ./deploy.sh "$@"
    ;;
  local-sim)
    cd "$APP_ROOT"
    exec g1-stack sim-run "$@"
    ;;
  g1-stack)
    cd "$APP_ROOT"
    exec g1-stack "$@"
    ;;
  *)
    exec "$command" "$@"
    ;;
esac
