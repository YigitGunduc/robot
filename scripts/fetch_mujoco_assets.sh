#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/upstream.env"

ASSET_ROOT="$PROJECT_ROOT/third_party/mujoco_menagerie"
REPOSITORY_URL="https://github.com/google-deepmind/mujoco_menagerie.git"

if [[ -d "$ASSET_ROOT/.git" ]]; then
  ACTUAL_REF="$(git -C "$ASSET_ROOT" rev-parse HEAD)"
  if [[ "$ACTUAL_REF" != "$MUJOCO_MENAGERIE_REF" ]]; then
    echo "Asset checkout exists at unexpected ref: $ACTUAL_REF" >&2
    echo "Expected: $MUJOCO_MENAGERIE_REF" >&2
    exit 2
  fi
  echo "MuJoCo Menagerie assets already pinned at $ACTUAL_REF"
  exit 0
fi

if [[ -e "$ASSET_ROOT" ]]; then
  echo "Refusing to overwrite non-git path: $ASSET_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$ASSET_ROOT")"
git clone --filter=blob:none --no-checkout "$REPOSITORY_URL" "$ASSET_ROOT"
git -C "$ASSET_ROOT" sparse-checkout set unitree_g1
git -C "$ASSET_ROOT" checkout "$MUJOCO_MENAGERIE_REF"
echo "Fetched Unitree G1 assets at $MUJOCO_MENAGERIE_REF"

