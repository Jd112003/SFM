#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-config.a100.yaml}"
OBJECT_ID="${OBJECT_ID:-leon}"
SERVICE_NAME="${SERVICE_NAME:-sfm-gpu}"
export SFM_UID="${SFM_UID:-$(id -u)}"
export SFM_GID="${SFM_GID:-$(id -g)}"

run_step() {
  local step="$1"
  echo
  echo "==> Running ${step} for ${OBJECT_ID} with ${CONFIG_PATH}"
  docker compose run --rm "${SERVICE_NAME}" "${step}" --object "${OBJECT_ID}" --config "${CONFIG_PATH}"
}

echo "==> Verifying Docker GPU access"
docker compose run --rm --entrypoint bash colmap-gpu -lc 'nvidia-smi'

echo
echo "==> Building pipeline image"
docker compose build sfm

run_step prepare
run_step extract-match
run_step reconstruct
run_step analyze-graph
run_step detect-doppelgangers
run_step run-ablation
run_step report

echo
echo "==> Pipeline completed"
echo "Outputs available under outputs/${OBJECT_ID}/"
