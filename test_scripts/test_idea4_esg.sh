#!/usr/bin/env bash
set -euo pipefail

# idea4 ESG 仅做推理。路径可以按以下任一方式传入（优先级从高到低）：
#   bash test_idea4_esg.sh <sd2.1目录> <测试集目录> <checkpoint> <event encoder目录> <输出目录>
#   通过同名环境变量传入；也可以直接修改下面的默认值。
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '4,12p' "$0"
  exit 0
fi

PRETRAINED_MODEL_NAME_OR_PATH="${1:-${PRETRAINED_MODEL_NAME_OR_PATH:-/public/home/slpchen/stable-diffusion-2-1-base}}"
TEST_DATA_ROOT="${2:-${TEST_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/test}}"
CHECKPOINT_PATH="${3:-${CHECKPOINT_PATH:-/public/home/slpchen/OSEDiff_project/TrainingWeights/OSEDiff_stage3_8-25-1350/checkpoints/model_70000.pkl}}"
RGB_EVENT_PRETRAINED_ROOT="${4:-${RGB_EVENT_PRETRAINED_ROOT:-/public/home/slpchen/OSEDiff_project/TrainingWeights/REencoder/SDE/indoor/joint/weights/best}}"
OUTPUT_DIR="${5:-${OUTPUT_DIR:-/public/home/slpchen/OSEDiff_project/test_results/clora/idea4_stage3_test_0}}"
export PRETRAINED_MODEL_NAME_OR_PATH TEST_DATA_ROOT CHECKPOINT_PATH RGB_EVENT_PRETRAINED_ROOT OUTPUT_DIR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 先创建结果目录，便于记录后续的路径校验或 Python 启动错误。
mkdir -p "$OUTPUT_DIR"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"

EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-auto}"
DECODER_REFINEMENT_ENABLED="${DECODER_REFINEMENT_ENABLED:-1}"
if [[ "$EVENT_LORA_VARIANT" == residual_cross_attn* ]]; then
  EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-64}"
else
  EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-32}"
fi

REQUIRED_DIRS=("$PRETRAINED_MODEL_NAME_OR_PATH" "$TEST_DATA_ROOT")
REQUIRED_FILES=("$CHECKPOINT_PATH")
DECODER_ARGS=()
case "$DECODER_REFINEMENT_ENABLED" in
  1|true|TRUE|yes|YES)
    DECODER_REFINEMENT_ENABLED=1
    REQUIRED_DIRS+=("$RGB_EVENT_PRETRAINED_ROOT")
    REQUIRED_FILES+=(
      "$RGB_EVENT_PRETRAINED_ROOT/Encoder_I.pth"
      "$RGB_EVENT_PRETRAINED_ROOT/Encoder_E.pth"
    )
    DECODER_ARGS=(
      --decoder_refinement_enabled
      --decoder_refinement_variant "$DECODER_REFINEMENT_VARIANT"
      --rgb_event_pretrained_root "$RGB_EVENT_PRETRAINED_ROOT"
    )
    ;;
  0|false|FALSE|no|NO)
    DECODER_REFINEMENT_ENABLED=0
    DECODER_ARGS=(--no-decoder_refinement_enabled)
    ;;
  *)
    echo "[ERROR] DECODER_REFINEMENT_ENABLED must be 0 or 1, got: $DECODER_REFINEMENT_ENABLED" >&2
    exit 1
    ;;
esac

for required_dir in "${REQUIRED_DIRS[@]}"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done
for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[ERROR] Required file does not exist: $required_file" >&2
    exit 1
  fi
done

OPTIONAL_ARGS=()
# 默认计算 FID；设置 CALCULATE_FID=0 可关闭。
if [[ "${CALCULATE_FID:-1}" == "1" ]]; then
  OPTIONAL_ARGS+=(--calculate_fid)
fi
if [[ "${SAVE_IMAGES:-1}" == "0" ]]; then
  OPTIONAL_ARGS+=(--no-save_images)
fi
if [[ "${RESUME_IGNORE_MISMATCH:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--resume_ignore_mismatch)
fi
if [[ "${MAX_SAMPLES:-0}" != "0" ]]; then
  OPTIONAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" idea4 esg_test \
  "PRETRAINED_MODEL_NAME_OR_PATH=$PRETRAINED_MODEL_NAME_OR_PATH" \
  "TEST_DATA_ROOT=$TEST_DATA_ROOT" \
  "CHECKPOINT_PATH=$CHECKPOINT_PATH" \
  "RGB_EVENT_PRETRAINED_ROOT=$RGB_EVENT_PRETRAINED_ROOT" \
  "DECODER_REFINEMENT_ENABLED=$DECODER_REFINEMENT_ENABLED" \
  "CALCULATE_FID=${CALCULATE_FID:-1}" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" \
  "DECODER_REFINEMENT_VARIANT=$DECODER_REFINEMENT_VARIANT" \
  "ESG_GUIDANCE_SCALES=${ESG_GUIDANCE_SCALES:-0,0.1,0.25,0.5,0.75}" \
  "ESG_SWAP_RATIOS=${ESG_SWAP_RATIOS:-0.05,0.10}" \
  "ESG_SWAP_MODES=${ESG_SWAP_MODES:-spatial}" \
  "ESG_PAIR_POLICIES=${ESG_PAIR_POLICIES:-dissimilar}" \
  "ESG_FEATURE_SCALES=${ESG_FEATURE_SCALES:-32,16,8}" \
  "ESG_EXECUTION_MODE=${ESG_EXECUTION_MODE:-batched}"

echo "[Idea4 ESG] checkpoint=$CHECKPOINT_PATH"
echo "[Idea4 ESG] output=$OUTPUT_DIR"
echo "[Idea4 ESG] calculate_fid=${CALCULATE_FID:-1}"
echo "[Idea4 ESG] decoder_refinement_enabled=$DECODER_REFINEMENT_ENABLED"
echo "[Idea4 ESG] guidance=${ESG_GUIDANCE_SCALES:-0,0.1,0.25,0.5,0.75} ratio=${ESG_SWAP_RATIOS:-0.05,0.10}"
echo "[Idea4 ESG] mode=${ESG_SWAP_MODES:-spatial} policy=${ESG_PAIR_POLICIES:-dissimilar} scales=${ESG_FEATURE_SCALES:-32,16,8}"

python "$REPO_ROOT/test_idea4_esg.py" \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --event_lora_checkpoint "$CHECKPOINT_PATH" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  "${DECODER_ARGS[@]}" \
  --cgdrs_event_levels "${CGDRS_EVENT_LEVELS:-1,2,3}" \
  --cgdrs_gate_bias "${CGDRS_GATE_BIAS:--2.0}" \
  --cgdrs_activity_mode "${CGDRS_ACTIVITY_MODE:-abs_mean}" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "$EVENT_LORA_MOD_CHANNELS" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --mod_num_heads "${MOD_NUM_HEADS:-4}" \
  --mod_window_size "${MOD_WINDOW_SIZE:-8}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --generator_timestep "${GENERATOR_TIMESTEP:-500}" \
  --image_size "${IMAGE_SIZE:-256}" \
  --mixed_precision "${MIXED_PRECISION:-fp16}" \
  --esg_guidance_scales "${ESG_GUIDANCE_SCALES:-0,0.1,0.25,0.5,0.75}" \
  --esg_swap_ratios "${ESG_SWAP_RATIOS:-0.05,0.10}" \
  --esg_swap_modes "${ESG_SWAP_MODES:-spatial}" \
  --esg_pair_policies "${ESG_PAIR_POLICIES:-dissimilar}" \
  --esg_feature_scales "${ESG_FEATURE_SCALES:-32,16,8}" \
  --esg_execution_mode "${ESG_EXECUTION_MODE:-batched}" \
  --esg_seed "${ESG_SEED:-42}" \
  "${OPTIONAL_ARGS[@]}"
