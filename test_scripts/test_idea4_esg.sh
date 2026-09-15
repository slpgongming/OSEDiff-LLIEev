#!/usr/bin/env bash
set -euo pipefail

# 可直接修改下方默认值，也可由外层实验脚本通过同名环境变量覆盖。
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '4,45p' "$0"
  exit 0
fi

# ===== 用户配置区域 =====
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/public/home/slpchen/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/public/home/slpchen/OSEDiff_project/TrainingWeights/OSEDiff_stage5_8-17-1100/checkpoints/model_final.pkl}"
RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/public/home/slpchen/OSEDiff_project/TrainingWeights/REencoder/SDE/indoor/joint/weights/best}"
OUTPUT_DIR="${OUTPUT_DIR:-/public/home/slpchen/OSEDiff_project/test_results/idea4_esg}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
CALCULATE_FID="${CALCULATE_FID:-1}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
RESUME_IGNORE_MISMATCH="${RESUME_IGNORE_MISMATCH:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-auto}"
DECODER_REFINEMENT_ENABLED="${DECODER_REFINEMENT_ENABLED:-1}"
ESG_GUIDANCE_SCALES="${ESG_GUIDANCE_SCALES:-0,0.1,0.25,0.5,0.75}"
ESG_SWAP_RATIOS="${ESG_SWAP_RATIOS:-0.05,0.10}"
ESG_SWAP_MODES="${ESG_SWAP_MODES:-spatial}"
ESG_PAIR_POLICIES="${ESG_PAIR_POLICIES:-dissimilar}"
ESG_FEATURE_SCALES="${ESG_FEATURE_SCALES:-32,16,8}"
ESG_EXECUTION_MODE="${ESG_EXECUTION_MODE:-batched}"
ESG_SEED="${ESG_SEED:-42}"
GENERATOR_TIMESTEP="${GENERATOR_TIMESTEP:-500}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
LORA_RANK="${LORA_RANK:-4}"
EVENT_LORA_MODULATION_SCALE="${EVENT_LORA_MODULATION_SCALE:-0.1}"
MOD_NUM_HEADS="${MOD_NUM_HEADS:-4}"
MOD_WINDOW_SIZE="${MOD_WINDOW_SIZE:-8}"
EVENT_LORA_CBAM_REDUCTION="${EVENT_LORA_CBAM_REDUCTION:-8}"
EVENT_LORA_CBAM_SPATIAL_KERNEL="${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}"
CLORA_EVENT_ACTIVE_BINS="${CLORA_EVENT_ACTIVE_BINS:-5}"
RGB_EVENT_ACTIVE_BINS="${RGB_EVENT_ACTIVE_BINS:-5}"
CGDRS_EVENT_LEVELS="${CGDRS_EVENT_LEVELS:-1,2,3}"
CGDRS_GATE_BIAS="${CGDRS_GATE_BIAS:--2.0}"
CGDRS_ACTIVITY_MODE="${CGDRS_ACTIVITY_MODE:-abs_mean}"
# ===== 配置区域结束 =====

export PRETRAINED_MODEL_NAME_OR_PATH TEST_DATA_ROOT CHECKPOINT_PATH RGB_EVENT_PRETRAINED_ROOT OUTPUT_DIR CUDA_VISIBLE_DEVICES

# 先创建结果目录，便于记录后续的路径校验或 Python 启动错误。
mkdir -p "$OUTPUT_DIR"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"

if [[ "$EVENT_LORA_VARIANT" == residual_cross_attn* ]]; then
  EVENT_LORA_MOD_CHANNELS=64
else
  EVENT_LORA_MOD_CHANNELS=32
fi

for active_bin_setting in "$CLORA_EVENT_ACTIVE_BINS" "$RGB_EVENT_ACTIVE_BINS"; do
  if [[ ! "$active_bin_setting" =~ ^[1-5]$ ]]; then
    echo "[ERROR] Active event bins must be an integer from 1 to 5, got: $active_bin_setting" >&2
    exit 1
  fi
done

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
    # 旧版 legacy_fused decoder 还会使用 RGB-event encoder 的 Merge 权重。
    if [[ "$DECODER_REFINEMENT_VARIANT" == "legacy_fused" ]]; then
      REQUIRED_FILES+=("$RGB_EVENT_PRETRAINED_ROOT/Merge.pth")
    fi
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
if [[ "$CALCULATE_FID" == "1" ]]; then
  OPTIONAL_ARGS+=(--calculate_fid)
fi
if [[ "$SAVE_IMAGES" == "0" ]]; then
  OPTIONAL_ARGS+=(--no-save_images)
fi
if [[ "$RESUME_IGNORE_MISMATCH" == "1" ]]; then
  OPTIONAL_ARGS+=(--resume_ignore_mismatch)
fi
if [[ "$MAX_SAMPLES" != "0" ]]; then
  OPTIONAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" idea4 esg_test \
  "PRETRAINED_MODEL_NAME_OR_PATH=$PRETRAINED_MODEL_NAME_OR_PATH" \
  "TEST_DATA_ROOT=$TEST_DATA_ROOT" \
  "CHECKPOINT_PATH=$CHECKPOINT_PATH" \
  "RGB_EVENT_PRETRAINED_ROOT=$RGB_EVENT_PRETRAINED_ROOT" \
  "DECODER_REFINEMENT_ENABLED=$DECODER_REFINEMENT_ENABLED" \
  "CALCULATE_FID=$CALCULATE_FID" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" \
  "DECODER_REFINEMENT_VARIANT=$DECODER_REFINEMENT_VARIANT" \
  "CLORA_EVENT_ACTIVE_BINS=$CLORA_EVENT_ACTIVE_BINS" \
  "RGB_EVENT_ACTIVE_BINS=$RGB_EVENT_ACTIVE_BINS" \
  "ESG_GUIDANCE_SCALES=$ESG_GUIDANCE_SCALES" \
  "ESG_SWAP_RATIOS=$ESG_SWAP_RATIOS" \
  "ESG_SWAP_MODES=$ESG_SWAP_MODES" \
  "ESG_PAIR_POLICIES=$ESG_PAIR_POLICIES" \
  "ESG_FEATURE_SCALES=$ESG_FEATURE_SCALES" \
  "ESG_EXECUTION_MODE=$ESG_EXECUTION_MODE"

echo "[Idea4 ESG] checkpoint=$CHECKPOINT_PATH"
echo "[Idea4 ESG] output=$OUTPUT_DIR"
echo "[Idea4 ESG] calculate_fid=$CALCULATE_FID"
echo "[Idea4 ESG] decoder_refinement_enabled=$DECODER_REFINEMENT_ENABLED"
echo "[Idea4 ESG] active event bins: cLoRA=$CLORA_EVENT_ACTIVE_BINS, RGB-event=$RGB_EVENT_ACTIVE_BINS (total channels=5)"
echo "[Idea4 ESG] guidance=$ESG_GUIDANCE_SCALES ratio=$ESG_SWAP_RATIOS"
echo "[Idea4 ESG] mode=$ESG_SWAP_MODES policy=$ESG_PAIR_POLICIES scales=$ESG_FEATURE_SCALES"

python "$REPO_ROOT/test_idea4_esg.py" \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --event_lora_checkpoint "$CHECKPOINT_PATH" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  "${DECODER_ARGS[@]}" \
  --clora_event_active_bins "$CLORA_EVENT_ACTIVE_BINS" \
  --rgb_event_active_bins "$RGB_EVENT_ACTIVE_BINS" \
  --cgdrs_event_levels "$CGDRS_EVENT_LEVELS" \
  --cgdrs_gate_bias "$CGDRS_GATE_BIAS" \
  --cgdrs_activity_mode "$CGDRS_ACTIVITY_MODE" \
  --lora_rank "$LORA_RANK" \
  --mod_channels "$EVENT_LORA_MOD_CHANNELS" \
  --modulation_scale "$EVENT_LORA_MODULATION_SCALE" \
  --mod_num_heads "$MOD_NUM_HEADS" \
  --mod_window_size "$MOD_WINDOW_SIZE" \
  --event_lora_cbam_reduction "$EVENT_LORA_CBAM_REDUCTION" \
  --event_lora_cbam_spatial_kernel "$EVENT_LORA_CBAM_SPATIAL_KERNEL" \
  --generator_timestep "$GENERATOR_TIMESTEP" \
  --image_size "$IMAGE_SIZE" \
  --mixed_precision "$MIXED_PRECISION" \
  --esg_guidance_scales "$ESG_GUIDANCE_SCALES" \
  --esg_swap_ratios "$ESG_SWAP_RATIOS" \
  --esg_swap_modes "$ESG_SWAP_MODES" \
  --esg_pair_policies "$ESG_PAIR_POLICIES" \
  --esg_feature_scales "$ESG_FEATURE_SCALES" \
  --esg_execution_mode "$ESG_EXECUTION_MODE" \
  --esg_seed "$ESG_SEED" \
  "${OPTIONAL_ARGS[@]}"
