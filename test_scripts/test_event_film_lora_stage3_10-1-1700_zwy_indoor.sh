#!/usr/bin/env bash
set -euo pipefail

# zwy / Stage 3 indoor test: disable every ordinary UNet LoRA adapter;
# the separate Event-LoRA remains active.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "$SCRIPT_PATH")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/test_event_film_lora_osediff.py" || ! -f "$REPO_ROOT/scripts/archive_run_context.sh" ]]; then
  echo "[ERROR] REPO_ROOT is not the OSEDiff code directory: $REPO_ROOT" >&2
  exit 1
fi
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/zhouweiyu/Chen_Project/PretrainedModel/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/public/home/zhouweiyu/Chen_Project/dataset/SDE/indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/public/home/zhouweiyu/Chen_Project/OSEDiff_project/TrainingWeights/OSEDiff_stage3_10-1-1700/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/public/home/zhouweiyu/Chen_Project/OSEDiff_project/test_results/indoor_stage3_test0}"
GENERATOR_TIMESTEP="${GENERATOR_TIMESTEP:-500}"
ABLATION_MODES="${ABLATION_MODES:-correct,zero}"
CALCULATE_FID="${CALCULATE_FID:-1}"
MULTI_GPU="${MULTI_GPU:-1}"
SAVE_DEBUG_VIS="${SAVE_DEBUG_VIS:-1}"

# Support both the test CLI shipped with the Stage 3 run and the newer adapter-mode CLI.
if grep -q -- '--unet_lora_parts' "$REPO_ROOT/test_event_film_lora_osediff.py"; then
  UNET_ADAPTER_ARGS=(--unet_lora_parts none)
elif grep -q -- '--unet_adapter_mode' "$REPO_ROOT/test_event_film_lora_osediff.py"; then
  UNET_ADAPTER_ARGS=(--unet_adapter_mode none)
else
  echo "[ERROR] Test entry point has no supported ordinary UNet LoRA selector." >&2
  exit 1
fi

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
  --stage 3
  "${UNET_ADAPTER_ARGS[@]}"
  --input_image "$TEST_DATA_ROOT"
  --output_dir "$OUTPUT_DIR"
  --pretrained_model_name_or_path "$PRETRAINED_MODEL"
  --osediff_path "$CHECKPOINT_PATH"
  --generator_timestep "$GENERATOR_TIMESTEP"
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
archive_run_context "$OUTPUT_DIR" "$SCRIPT_PATH" stage3 test \
  "ACCOUNT=zwy" "DATASET=SDE_indoor" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "CHECKPOINT_PATH=$CHECKPOINT_PATH" "OUTPUT_DIR=$OUTPUT_DIR" \
  "ORDINARY_UNET_LORA=none" "EVENT_LORA_VARIANT=residual_cross_attn" \
  "EVENT_CHANNELS=5" "GENERATOR_TIMESTEP=$GENERATOR_TIMESTEP" \
  "ABLATION_MODES=$ABLATION_MODES" "CALCULATE_FID=$CALCULATE_FID" \
  "MULTI_GPU=$MULTI_GPU" "SAVE_DEBUG_VIS=$SAVE_DEBUG_VIS" \
  "RESOLVED_COMMAND=$RESOLVED_COMMAND"

"${COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/test.log"
