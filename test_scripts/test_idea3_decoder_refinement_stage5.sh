#!/usr/bin/env bash
set -euo pipefail

# Stage5 A-F 测试：使用完整 Stage5 LoRA 与 refinement decoder 权重。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/public/home/slpchen/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/public/home/slpchen/OSEDiff_project/TrainingWeights/OSEDiff_stage5_8-17-1100/checkpoints/model_final.pkl}"
RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/public/home/slpchen/OSEDiff_project/TrainingWeights/HDRev}"
OUTPUT_DIR="${OUTPUT_DIR:-/public/home/slpchen/OSEDiff_project/test_results/clora/idea3_test_3}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"

if [[ "$EVENT_LORA_VARIANT" == residual_cross_attn* ]]; then
  EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-64}"
else
  EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-32}"
fi

for required_dir in "$PRETRAINED_MODEL_NAME_OR_PATH" "$TEST_DATA_ROOT" "$RGB_EVENT_PRETRAINED_ROOT"; do
  if [ ! -d "$required_dir" ]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done

for required_file in "$CHECKPOINT_PATH" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_I.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_E.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Merge.pth"; do
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
cp -f "$0" "$OUTPUT_DIR/$(basename "$0")"

echo "[Stage5 Test] checkpoint=$CHECKPOINT_PATH"
echo "[Stage5 Test] variant=$EVENT_LORA_VARIANT | mod_channels=$EVENT_LORA_MOD_CHANNELS"
echo "[Stage5 Test] experiments=${EXPERIMENTS:-A,B,E}"
echo "[Stage5 Test] output=$OUTPUT_DIR"

python test_idea3_decoder_refinement_stage5.py \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --event_lora_checkpoint "$CHECKPOINT_PATH" \
  --rgb_event_pretrained_root "$RGB_EVENT_PRETRAINED_ROOT" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "$EVENT_LORA_MOD_CHANNELS" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --generator_timestep "${GENERATOR_TIMESTEP:-500}" \
  --image_size "${IMAGE_SIZE:-256}" \
  --mixed_precision "${MIXED_PRECISION:-fp16}" \
  --experiments "${EXPERIMENTS:-A,B,E}" \
  "${FID_ARGS[@]}" \
  "${SAVE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
