#!/usr/bin/env bash
set -euo pipefail

# whb 账户：已确认平台路径，checkpoint/配对清单/标定文件由本次实验明确指定。
# DRY_RUN=1 仅打印实际 argv，不检查远程文件是否存在，不运行模型。
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo 'Required: CHECKPOINT_PATH, PAIR_MANIFEST; enabled DPS also requires OBSERVATION_CALIBRATION.'
  echo 'Optional: TEST_DATA_ROOT, OUTPUT_DIR, GENERATOR_TIMESTEP, UNET_LORA_PARTS, DPS_STEPS=1, DPS_STEP_REL=0.01, DPS_RADIUS_REL=0.05.'
  echo 'DRY_RUN=1 prints argv; PREFLIGHT_ONLY=1 checks manifest/calibration without loading OSEDiff.'
  exit 0
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/public/home/whb/Chen_Project/PretrainedModel/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/test}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/test_results/idea7_dps}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the exact trained checkpoint.}"
PAIR_MANIFEST="${PAIR_MANIFEST:?Set PAIR_MANIFEST to the validated explicit pair manifest.}"
OBSERVATION_CALIBRATION="${OBSERVATION_CALIBRATION:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DPS_STEPS="${DPS_STEPS:-1}"
DPS_UP_BLOCK="${DPS_UP_BLOCK:-penultimate}"
DPS_STEP_REL="${DPS_STEP_REL:-0.01}"
DPS_RADIUS_REL="${DPS_RADIUS_REL:-0.05}"
GENERATOR_TIMESTEP="${GENERATOR_TIMESTEP:-}"
UNET_LORA_PARTS="${UNET_LORA_PARTS:-}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"
FEEDBACK_MODE="${FEEDBACK_MODE:-correct}"
FEEDBACK_SHIFT="${FEEDBACK_SHIFT:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
export CUDA_VISIBLE_DEVICES
case "$DPS_STEPS" in 0|1|2) ;; *) echo 'DPS_STEPS must be 0, 1 or 2.' >&2; exit 1 ;; esac
case "$MIXED_PRECISION" in fp16|fp32) ;; *) echo 'MIXED_PRECISION must be fp16 or fp32.' >&2; exit 1 ;; esac
CMD=("$PYTHON_BIN" "$REPO_ROOT/test_idea7_dps.py"
  --data-root "$TEST_DATA_ROOT" --pair-manifest "$PAIR_MANIFEST"
  --output-dir "$OUTPUT_DIR" --pretrained-model-name-or-path "$PRETRAINED_MODEL_NAME_OR_PATH"
  --event-lora-checkpoint "$CHECKPOINT_PATH" --dps-steps "$DPS_STEPS"
  --dps-up-block "$DPS_UP_BLOCK" --dps-step-rel "$DPS_STEP_REL" --dps-radius-rel "$DPS_RADIUS_REL"
  --mixed-precision "$MIXED_PRECISION" --max-samples "$MAX_SAMPLES" --seed "$SEED"
  --feedback-mode "$FEEDBACK_MODE" --feedback-shift "$FEEDBACK_SHIFT")
[[ -z "$OBSERVATION_CALIBRATION" ]] || CMD+=(--observation-calibration "$OBSERVATION_CALIBRATION")
[[ -z "$GENERATOR_TIMESTEP" ]] || CMD+=(--generator-timestep "$GENERATOR_TIMESTEP")
[[ -z "$UNET_LORA_PARTS" ]] || CMD+=(--unet-lora-parts "$UNET_LORA_PARTS")
[[ -z "${RGB_EVENT_PRETRAINED_ROOT:-}" ]] || CMD+=(--rgb-event-pretrained-root "$RGB_EVENT_PRETRAINED_ROOT")
[[ -z "${EVALUATION_MANIFEST:-}" ]] || CMD+=(--evaluation-manifest "$EVALUATION_MANIFEST")
[[ "$PREFLIGHT_ONLY" != 1 ]] || CMD+=(--preflight-only)
CMD+=("$@")
printf '[idea7/whb] '
printf '%q ' "${CMD[@]}"
printf '\n'
[[ "$DRY_RUN" != 1 ]] || exit 0

# 时间/极性/区间/标定状态由 Python 的预检统一验证；不在 shell 猜测。
[[ -d "$TEST_DATA_ROOT" ]] || { echo "Missing data directory: $TEST_DATA_ROOT" >&2; exit 1; }
[[ -f "$PAIR_MANIFEST" ]] || { echo "Missing pair manifest: $PAIR_MANIFEST" >&2; exit 1; }
if [[ "$PREFLIGHT_ONLY" != 1 ]]; then
  [[ -d "$PRETRAINED_MODEL_NAME_OR_PATH" ]] || { echo 'Missing pretrained model directory.' >&2; exit 1; }
  [[ -f "$CHECKPOINT_PATH" ]] || { echo 'Missing trained checkpoint.' >&2; exit 1; }
fi
mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/run.log"
