#!/usr/bin/env bash
set -euo pipefail

# Stage2 测试：加载 Stage2 checkpoint，输出 correct/zero/shuffled 事件消融指标。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/path/to/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/path/to/SDE_indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-experience/event_film_lora_stage2/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-output_event_film/stage2_test}"

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

python test_event_film_lora_osediff.py \
  --stage 2 \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --osediff_path "$CHECKPOINT_PATH" \
  --generator_timestep "${GENERATOR_TIMESTEP:-999}" \
  --image_size 256 \
  --mixed_precision fp16 \
  --ablation_modes "${ABLATION_MODES:-correct,zero,shuffled}" \
  --save_debug_vis
