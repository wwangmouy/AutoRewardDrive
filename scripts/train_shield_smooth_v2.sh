#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-4}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
DEVICE="${DEVICE:-cuda:0}"
HOST="${HOST:-localhost}"
PORT="${PORT:-2000}"
FPS="${FPS:-15}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-50}"
LATEST_BUNDLE_FREQ="${LATEST_BUNDLE_FREQ:-50000}"
RUN_NAME="${RUN_NAME:-shield_smooth_v2_clean}"

cd "$(dirname "$0")/.."

python train.py \
  --config "${CONFIG}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --device "${DEVICE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --fps "${FPS}" \
  --no_render \
  --start_carla \
  --num_checkpoints "${NUM_CHECKPOINTS}" \
  --latest_bundle_freq "${LATEST_BUNDLE_FREQ}" \
  --run_name "${RUN_NAME}"
