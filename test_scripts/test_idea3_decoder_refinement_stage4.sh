#!/usr/bin/env bash
set -euo pipefail

# Stage4 A-F 评估：A/B 使用同一 correct-event z0，只比较 decoder fusion 开关。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"

PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/public/home/slpchen/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/OSEDiff_cbam_lowrank_residual_stage4/checkpoints/model_final.pkl}"
RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/public/home/slpchen/Chen_Project/TrainingWeights/HDRev}"
OUTPUT_DIR="${OUTPUT_DIR:-test_results/OSEDiff_cbam_lowrank_residual_stage4}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-cbam_lowrank_residual}"
DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-auto}"
CGDRS_EVENT_LEVELS="${CGDRS_EVENT_LEVELS:-1,2,3}"
CGDRS_GATE_BIAS="${CGDRS_GATE_BIAS:--2.0}"
CGDRS_ACTIVITY_MODE="${CGDRS_ACTIVITY_MODE:-abs_mean}"

for required_dir in "$PRETRAINED_MODEL_NAME_OR_PATH" "$TEST_DATA_ROOT" "$RGB_EVENT_PRETRAINED_ROOT"; do
  if [ ! -d "$required_dir" ]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done
for required_file in "$CHECKPOINT_PATH" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_I.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_E.pth"; do
  if [ ! -f "$required_file" ]; then
    echo "[ERROR] Required file does not exist: $required_file" >&2
    exit 1
  fi
done

FID_ARGS=()
if [[ "${CALCULATE_FID:-0}" == "1" ]]; then
  FID_ARGS+=(--calculate_fid)
fi
SAVE_ARGS=()
if [[ "${SAVE_IMAGES:-1}" == "0" ]]; then
  SAVE_ARGS+=(--no-save_images)
fi
RESUME_ARGS=()
if [[ "${RESUME_IGNORE_MISMATCH:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume_ignore_mismatch)
fi

mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage4 test \
  "PRETRAINED_MODEL_NAME_OR_PATH=$PRETRAINED_MODEL_NAME_OR_PATH" \
  "TEST_DATA_ROOT=$TEST_DATA_ROOT" "CHECKPOINT_PATH=$CHECKPOINT_PATH" \
  "RGB_EVENT_PRETRAINED_ROOT=$RGB_EVENT_PRETRAINED_ROOT" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" "DECODER_REFINEMENT_VARIANT=$DECODER_REFINEMENT_VARIANT" \
  "CGDRS_EVENT_LEVELS=$CGDRS_EVENT_LEVELS" "CGDRS_GATE_BIAS=$CGDRS_GATE_BIAS" \
  "CGDRS_ACTIVITY_MODE=$CGDRS_ACTIVITY_MODE" \
  "IMAGE_SIZE=${IMAGE_SIZE:-256}" "GENERATOR_TIMESTEP=${GENERATOR_TIMESTEP:-999}"
echo "[Stage4 Test] variant=$EVENT_LORA_VARIANT | decoder_refinement_variant=$DECODER_REFINEMENT_VARIANT | experiments=${EXPERIMENTS:-A,B,C,D,E,F}"

python test_idea3_decoder_refinement.py \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --event_lora_checkpoint "$CHECKPOINT_PATH" \
  --rgb_event_pretrained_root "$RGB_EVENT_PRETRAINED_ROOT" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --decoder_refinement_variant "$DECODER_REFINEMENT_VARIANT" \
  --cgdrs_event_levels "$CGDRS_EVENT_LEVELS" \
  --cgdrs_gate_bias "$CGDRS_GATE_BIAS" \
  --cgdrs_activity_mode "$CGDRS_ACTIVITY_MODE" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "${EVENT_LORA_MOD_CHANNELS:-32}" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --generator_timestep "${GENERATOR_TIMESTEP:-999}" \
  --image_size "${IMAGE_SIZE:-256}" \
  --mixed_precision "${MIXED_PRECISION:-fp16}" \
  --experiments "${EXPERIMENTS:-A,B,C,D,E,F}" \
  "${FID_ARGS[@]}" \
  "${SAVE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
