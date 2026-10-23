#!/usr/bin/env bash
set -euo pipefail

# zwy / idea4 Stage 2：室内测试；普通 UNet LoRA 不激活，Event-LoRA 始终激活。
# 将脚本放入服务器代码仓库的 test_scripts/ 后运行；也可显式设置 REPO_ROOT。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "$SCRIPT_PATH")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/test_event_film_lora_osediff.py" || ! -f "$REPO_ROOT/scripts/archive_run_context.sh" ]]; then
  echo "[ERROR] REPO_ROOT is not the idea4 code directory: $REPO_ROOT" >&2
  exit 1
fi
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/zhouweiyu/Chen_Project/PretrainedModel/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/public/home/zhouweiyu/Chen_Project/dataset/SDE/indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/public/home/zhouweiyu/Chen_Project/OSEDiff_project/TrainingWeights/OSEDiff_stage2_9-30-1300/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/test_results/OSEDiff_stage2_9-30-1300_indoor}"
GENERATOR_TIMESTEP="${GENERATOR_TIMESTEP:-500}"
ABLATION_MODES="${ABLATION_MODES:-correct,zero}"
CALCULATE_FID="${CALCULATE_FID:-1}"
MULTI_GPU="${MULTI_GPU:-1}"
SAVE_DEBUG_VIS="${SAVE_DEBUG_VIS:-1}"

for switch in CALCULATE_FID MULTI_GPU SAVE_DEBUG_VIS; do
  value="${!switch}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "[ERROR] $switch must be 0 or 1." >&2
    exit 1
  fi
done

EXTRA_ARGS=()
if [[ "$CALCULATE_FID" == "1" ]]; then EXTRA_ARGS+=(--calculate_fid); fi
if [[ "$MULTI_GPU" == "1" ]]; then EXTRA_ARGS+=(--multi_gpu); fi
if [[ "$SAVE_DEBUG_VIS" == "1" ]]; then EXTRA_ARGS+=(--save_debug_vis); fi

COMMAND=(python test_event_film_lora_osediff.py
  --stage 2
  --input_image "$TEST_DATA_ROOT"
  --output_dir "$OUTPUT_DIR"
  --pretrained_model_name_or_path "$PRETRAINED_MODEL"
  --osediff_path "$CHECKPOINT_PATH"
  --generator_timestep "$GENERATOR_TIMESTEP"
  --unet_lora_parts none
  --event_bins 5
  --event_channels 5
  --clora_event_active_bins 5
  --rgb_event_active_bins 5
  --event_lora_variant residual_cross_attn
  --lora_rank 4
  --mod_channels 64
  --modulation_scale 0.1
  --image_size 256
  --mixed_precision fp16
  --ablation_modes "$ABLATION_MODES"
  "${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
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
command -v python >/dev/null || { echo "[ERROR] Activate the OSEDiff Python environment." >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
printf -v RESOLVED_COMMAND '%q ' "${COMMAND[@]}"
archive_run_context "$OUTPUT_DIR" "$SCRIPT_PATH" stage2 test \
  "ACCOUNT=zwy" "DATASET=SDE_indoor" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "CHECKPOINT_PATH=$CHECKPOINT_PATH" "OUTPUT_DIR=$OUTPUT_DIR" \
  "UNET_LORA_PARTS=none" "CLORA_EVENT_ACTIVE_BINS=5" \
  "GENERATOR_TIMESTEP=$GENERATOR_TIMESTEP" \
  "ABLATION_MODES=$ABLATION_MODES" "CALCULATE_FID=$CALCULATE_FID" \
  "MULTI_GPU=$MULTI_GPU" "SAVE_DEBUG_VIS=$SAVE_DEBUG_VIS" \
  "RESOLVED_COMMAND=$RESOLVED_COMMAND"

"${COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/test.log"
