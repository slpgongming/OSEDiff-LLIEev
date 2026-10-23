#!/usr/bin/env bash
set -euo pipefail

# slpchen / SDE indoor：测试由 10-1-1440 重命名的 EventOneChannel Stage3 权重。
# 与训练前向保持一致：只激活普通 default_others；独立 Event-LoRA 始终参与。
# 默认 GPU 0、1 并行评估 correct/zero，generator timestep=500。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/slpchen/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/public/home/slpchen/OSEDiff_project/TrainingWeights/EventOneChannel_stage3_10-1-1550/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/public/home/slpchen/OSEDiff_project/Test_results/EventOneChannel_indoor_stage3_test3}"
UNET_ADAPTER_MODE="${UNET_ADAPTER_MODE:-others_only}"

for switch in CALCULATE_FID MULTI_GPU; do
  value="${!switch:-1}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "[ERROR] $switch must be 0 or 1." >&2
    exit 1
  fi
done

FID_ARGS=()
if [[ "${CALCULATE_FID:-1}" == "1" ]]; then
  FID_ARGS+=(--calculate_fid)
fi
MULTI_GPU_ARGS=()
if [[ "${MULTI_GPU:-1}" == "1" ]]; then
  MULTI_GPU_ARGS+=(--multi_gpu)
fi

COMMAND=(python test_event_film_lora_osediff.py \
  --stage 3 \
  --unet_adapter_mode "$UNET_ADAPTER_MODE" \
  --event_bins 5 --event_channels 1 --rgb_event_channels 5 \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --osediff_path "$CHECKPOINT_PATH" \
  --generator_timestep "${GENERATOR_TIMESTEP:-500}" \
  --event_lora_variant "${EVENT_LORA_VARIANT:-residual_cross_attn}" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "${EVENT_LORA_MOD_CHANNELS:-64}" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --mod_num_heads "${MOD_NUM_HEADS:-4}" --mod_window_size "${MOD_WINDOW_SIZE:-8}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --image_size 256 \
  --mixed_precision fp16 \
  --ablation_modes "${ABLATION_MODES:-correct,zero}" \
  "${FID_ARGS[@]}" \
  "${MULTI_GPU_ARGS[@]}" \
  --save_debug_vis)

# dry-run 只打印命令，不读取服务器数据或权重，也不执行推理。
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
elif [[ "${DRY_RUN:-0}" != "0" ]]; then
  echo "[ERROR] DRY_RUN must be 0 or 1." >&2
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
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] OUTPUT_DIR must not be empty." >&2
  exit 1
fi
command -v python >/dev/null || { echo "[ERROR] Activate the OSEDiff Python environment." >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
printf -v RESOLVED_COMMAND '%q ' "${COMMAND[@]}"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage3 test \
  "ACCOUNT=slpchen" "DATASET=SDE_indoor" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "CHECKPOINT_PATH=$CHECKPOINT_PATH" "OUTPUT_DIR=$OUTPUT_DIR" \
  "UNET_ADAPTER_MODE=$UNET_ADAPTER_MODE" "GENERATOR_TIMESTEP=${GENERATOR_TIMESTEP:-500}" \
  "RESOLVED_COMMAND=$RESOLVED_COMMAND"
exec "${COMMAND[@]}"
