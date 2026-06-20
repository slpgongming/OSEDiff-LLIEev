#!/usr/bin/env bash
set -euo pipefail

# Stage2 测试：加载 Stage2 checkpoint，输出 correct/zero 事件消融指标。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/slpchen/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/OSEDiff_cbam_lowrank_residual_stage2_6-19/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-test_results/OSEDiff_cbam_lowrank_residual_stage2_6-19}"
FID_ARGS=()
if [[ "${CALCULATE_FID:-1}" == "1" ]]; then
  FID_ARGS+=(--calculate_fid)
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

python test_event_film_lora_osediff.py \
  --stage 2 \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --osediff_path "$CHECKPOINT_PATH" \
  --generator_timestep "${GENERATOR_TIMESTEP:-999}" \
  --event_lora_variant "${EVENT_LORA_VARIANT:-cbam_lowrank_residual}" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "${EVENT_LORA_MOD_CHANNELS:-32}" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --image_size 256 \
  --mixed_precision fp16 \
  --ablation_modes "${ABLATION_MODES:-correct,zero}" \
  "${FID_ARGS[@]}" \
  --save_debug_vis
