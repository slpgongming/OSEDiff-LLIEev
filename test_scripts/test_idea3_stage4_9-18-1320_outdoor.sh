#!/usr/bin/env bash
set -euo pipefail

# Stage4 9-18-1140（SDE outdoor）评估：A/B 共享同一 correct-event z0，
# 仅比较 legacy_fused decoder refinement 开关；E 额外评估 RGB-only refinement。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_out_release/sde_out_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage4_9-18-1320/checkpoints/model_final.pkl}"
RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/REencoder/SDE_outdoor}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data1/cjl/OSEDiff_Project/Test_results/idea3_test_15}"

EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
# 必须与本次 Stage4 训练的旧版 MultiScaleFusion decoder 保持一致。
DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-legacy_fused}"
# 复用训练阶段的双路 event voxel 布局：cLoRA 为完整 5-bin，RGB-event encoder 为 bin 0。
CLORA_EVENT_ACTIVE_BINS="${CLORA_EVENT_ACTIVE_BINS:-5}"
RGB_EVENT_ACTIVE_BINS="${RGB_EVENT_ACTIVE_BINS:-1}"

for active_bin_setting in "$CLORA_EVENT_ACTIVE_BINS" "$RGB_EVENT_ACTIVE_BINS"; do
  if [[ ! "$active_bin_setting" =~ ^[1-5]$ ]]; then
    echo "[ERROR] Active event bins must be an integer from 1 to 5, got: $active_bin_setting" >&2
    exit 1
  fi
done
for required_dir in "$PRETRAINED_MODEL_NAME_OR_PATH" "$TEST_DATA_ROOT" "$RGB_EVENT_PRETRAINED_ROOT"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done
# legacy_fused 推理需要 Merge.pth；缺失时在模型构造前明确报错。
for required_file in "$CHECKPOINT_PATH" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_I.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_E.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Merge.pth"; do
  if [[ ! -f "$required_file" ]]; then
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
  "RGB_EVENT_PRETRAINED_ROOT=$RGB_EVENT_PRETRAINED_ROOT" "OUTPUT_DIR=$OUTPUT_DIR" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" \
  "DECODER_REFINEMENT_VARIANT=$DECODER_REFINEMENT_VARIANT" \
  "CLORA_EVENT_ACTIVE_BINS=$CLORA_EVENT_ACTIVE_BINS" \
  "RGB_EVENT_ACTIVE_BINS=$RGB_EVENT_ACTIVE_BINS" \
  "IMAGE_SIZE=${IMAGE_SIZE:-256}" "GENERATOR_TIMESTEP=${GENERATOR_TIMESTEP:-500}" \
  "EXPERIMENTS=${EXPERIMENTS:-A,B,E}"

echo "[Stage4 Outdoor Test] checkpoint=$CHECKPOINT_PATH"
echo "[Stage4 Outdoor Test] output=$OUTPUT_DIR"
echo "[Stage4 Outdoor Test] active event bins: cLoRA=$CLORA_EVENT_ACTIVE_BINS, RGB-event=$RGB_EVENT_ACTIVE_BINS (total channels=5)"
echo "[Stage4 Outdoor Test] experiments=${EXPERIMENTS:-A,B,E}"

python "$REPO_ROOT/test_idea3_decoder_refinement.py" \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --event_lora_checkpoint "$CHECKPOINT_PATH" \
  --rgb_event_pretrained_root "$RGB_EVENT_PRETRAINED_ROOT" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --decoder_refinement_variant "$DECODER_REFINEMENT_VARIANT" \
  --clora_event_active_bins "$CLORA_EVENT_ACTIVE_BINS" \
  --rgb_event_active_bins "$RGB_EVENT_ACTIVE_BINS" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "${EVENT_LORA_MOD_CHANNELS:-64}" \
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
