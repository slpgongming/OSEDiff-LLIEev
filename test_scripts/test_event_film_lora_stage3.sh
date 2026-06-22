#!/usr/bin/env bash
set -euo pipefail

# Stage3 测试：GPU0/GPU1 并行运行 correct/zero，最后自动合并指标。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/path/to/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/path/to/SDE_indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-experience/event_film_lora_stage3/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-output_event_film/stage3_test}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-film}"
FID_ARGS=()
if [[ "${CALCULATE_FID:-1}" == "1" ]]; then
  FID_ARGS+=(--calculate_fid)
fi
MULTI_GPU_ARGS=()
if [[ "${MULTI_GPU:-1}" == "1" ]]; then
  MULTI_GPU_ARGS+=(--multi_gpu)
fi

if [ ! -d "$PRETRAINED_MODEL" ]; then
  echo "[ERROR] PRETRAINED_MODEL does not exist: $PRETRAINED_MODEL" >&2
  exit 1
fi
if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "[ERROR] TEST_DATA_ROOT does not exist: $TEST_DATA_ROOT" >&2
  exit 1
fi
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "[ERROR] CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cp -f "$0" "$OUTPUT_DIR/$(basename "$0")"

python test_event_film_lora_osediff.py \
  --stage 3 \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --osediff_path "$CHECKPOINT_PATH" \
  --generator_timestep "${GENERATOR_TIMESTEP:-999}" \
  --image_size 256 \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --mixed_precision fp16 \
  --ablation_modes "${ABLATION_MODES:-correct,zero}" \
  "${FID_ARGS[@]}" \
  "${MULTI_GPU_ARGS[@]}" \
  --save_debug_vis
