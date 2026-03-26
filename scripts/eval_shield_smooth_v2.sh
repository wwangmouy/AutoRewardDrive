#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_shield_smooth_v2.sh <model_dir> [checkpoint ...]"
  exit 1
fi

MODEL_DIR="$1"
shift

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(
    "model_150000_steps.zip"
    "model_200000_steps.zip"
    "model_300000_steps.zip"
    "model_500000_steps.zip"
  )
fi

CONFIG="${CONFIG:-4}"
DEVICE="${DEVICE:-cuda:0}"
TOWN="${TOWN:-Town02}"
DENSITY="${DENSITY:-regular}"
PORT="${PORT:-2020}"
SEED="${SEED:-101}"
EPISODES="${EPISODES:-10}"

cd "$(dirname "$0")/.."

python run_eval.py \
  --model_dir "${MODEL_DIR}" \
  --models "${MODELS[@]}" \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --town "${TOWN}" \
  --density "${DENSITY}" \
  --port "${PORT}" \
  --seed "${SEED}" \
  --episodes "${EPISODES}" \
  --inference_mode step \
  --eval_tag step_raw

python run_eval.py \
  --model_dir "${MODEL_DIR}" \
  --models "${MODELS[@]}" \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --town "${TOWN}" \
  --density "${DENSITY}" \
  --port "${PORT}" \
  --seed "${SEED}" \
  --episodes "${EPISODES}" \
  --use_shield \
  --inference_mode step \
  --eval_tag step_shielded
