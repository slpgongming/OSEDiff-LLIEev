#!/usr/bin/env bash
set -euo pipefail

# EventOneChannel Stage2 测试：以 correct/zero 事件消融验证单通道 cLoRA。
# 测试严格固定 cLoRA=1 通道，RE encoder=5 bins，并复用训练时的 LoRA 配置。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

# slpchen 账号的室内测试数据和 Stable Diffusion 2.1 基座。
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_EventOneChannel_stage2_9-19-2120/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data1/cjl/OSEDiff_Project/Test_results/OSEDiff_EventOneChannel_stage2_9-19-2120}"

EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
LORA_RANK="${LORA_RANK:-4}"
EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-64}"
EVENT_BINS="${EVENT_BINS:-5}"
EVENT_CHANNELS="${EVENT_CHANNELS:-1}"
RGB_EVENT_CHANNELS="${RGB_EVENT_CHANNELS:-5}"
ABLATION_MODES="${ABLATION_MODES:-correct,zero}"

if [[ "$EVENT_BINS" != "5" || "$EVENT_CHANNELS" != "1" || "$RGB_EVENT_CHANNELS" != "5" ]]; then
  echo "[ERROR] EventOneChannel requires EVENT_BINS=5, EVENT_CHANNELS=1, and RGB_EVENT_CHANNELS=5." >&2
  exit 1
fi
if [[ "$EVENT_LORA_VARIANT" != "residual_cross_attn" ]]; then
  echo "[ERROR] This Stage2 test must match training EVENT_LORA_VARIANT=residual_cross_attn." >&2
  exit 1
fi
for required_dir in "$PRETRAINED_MODEL" "$TEST_DATA_ROOT"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH" >&2
  exit 1
fi

FID_ARGS=()
if [[ "${CALCULATE_FID:-1}" == "1" ]]; then
  FID_ARGS+=(--calculate_fid)
fi
MULTI_GPU_ARGS=()
if [[ "${MULTI_GPU:-1}" == "1" ]]; then
  MULTI_GPU_ARGS+=(--multi_gpu)
fi

mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage2 test \
  "PRETRAINED_MODEL=$PRETRAINED_MODEL" "TEST_DATA_ROOT=$TEST_DATA_ROOT" \
  "CHECKPOINT_PATH=$CHECKPOINT_PATH" "OUTPUT_DIR=$OUTPUT_DIR" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" "LORA_RANK=$LORA_RANK" \
  "EVENT_LORA_MOD_CHANNELS=$EVENT_LORA_MOD_CHANNELS" \
  "EVENT_BINS=$EVENT_BINS" "EVENT_CHANNELS=$EVENT_CHANNELS" \
  "RGB_EVENT_CHANNELS=$RGB_EVENT_CHANNELS" \
  "ABLATION_MODES=$ABLATION_MODES" "IMAGE_SIZE=256" \
  "GENERATOR_TIMESTEP=${GENERATOR_TIMESTEP:-500}"

python test_event_film_lora_osediff.py \
  --stage 2 \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --osediff_path "$CHECKPOINT_PATH" \
  --generator_timestep "${GENERATOR_TIMESTEP:-500}" \
  --event_bins "$EVENT_BINS" \
  --event_channels "$EVENT_CHANNELS" \
  --rgb_event_channels "$RGB_EVENT_CHANNELS" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --lora_rank "$LORA_RANK" \
  --mod_channels "$EVENT_LORA_MOD_CHANNELS" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --image_size 256 \
  --mixed_precision fp16 \
  --ablation_modes "$ABLATION_MODES" \
  "${FID_ARGS[@]}" \
  "${MULTI_GPU_ARGS[@]}" \
  --save_debug_vis
