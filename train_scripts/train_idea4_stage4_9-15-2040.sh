#!/usr/bin/env bash
set -euo pipefail

# idea4 Stage4（SDE indoor）：冻结 Stage1 OSEDiff、Stage3 cLoRA 及 RGB-event
# encoder，仅训练 CGDRS decoder refinement blocks。除本文件明确列出的差异外，
# 超参数沿用 9-15-1920 的 Stage4 复现实验。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_INSECURE="${WANDB_INSECURE:-true}"
export WANDB_INSECURE_DISABLE_SSL="${WANDB_INSECURE_DISABLE_SSL:-true}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

# ===== 八卡服务器默认配置；可由同名环境变量覆盖 =====
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_in_release/train}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
EVENT_LORA_CHECKPOINT="${EVENT_LORA_CHECKPOINT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_clora_stage3_6-16-1050/checkpoints/model_final.pkl}"
# 此目录的 RGB-event encoder 在旧 timestamp bug 条件下训练；本次 decoder 输入也
# 显式压缩至 bin 0，以保持训练/预训练 voxel 布局一致。
RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/HDRev}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage4_9-15-2040}"

EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-cgdrs}"
CGDRS_EVENT_LEVELS="${CGDRS_EVENT_LEVELS:-2,3}"
CGDRS_GATE_BIAS="${CGDRS_GATE_BIAS:--3}"
CGDRS_ACTIVITY_MODE="${CGDRS_ACTIVITY_MODE:-abs_mean}"
# 其余设置照旧：cLoRA 也继续使用 bin-0 voxel，固定总通道数为 5。
CLORA_EVENT_ACTIVE_BINS="${CLORA_EVENT_ACTIVE_BINS:-1}"
RGB_EVENT_ACTIVE_BINS="${RGB_EVENT_ACTIVE_BINS:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
REPORT_TO="${REPORT_TO:-wandb}"
EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-64}"

for active_bin_setting in "$CLORA_EVENT_ACTIVE_BINS" "$RGB_EVENT_ACTIVE_BINS"; do
  if [[ ! "$active_bin_setting" =~ ^[1-5]$ ]]; then
    echo "[ERROR] Active event bins must be an integer from 1 to 5, got: $active_bin_setting" >&2
    exit 1
  fi
done
for required_dir in "$PRETRAINED_MODEL_NAME_OR_PATH" "$TRAIN_DATA_ROOT" "$RGB_EVENT_PRETRAINED_ROOT"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done
# CGDRS 不使用 legacy MultiScaleFusion，因此只需要两个 encoder 权重，不要求 Merge.pth。
for required_file in "$STAGE1_CHECKPOINT" "$EVENT_LORA_CHECKPOINT" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_I.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_E.pth"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[ERROR] Required file does not exist: $required_file" >&2
    exit 1
  fi
done

EXTRA_ARGS=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [[ "${RESUME_IGNORE_MISMATCH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume_ignore_mismatch)
fi
if [[ "${BUNDLE_RGB_EVENT_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--bundle_rgb_event_encoder)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi
TRACKER_ARGS=()
if [[ -n "$REPORT_TO" ]]; then
  TRACKER_ARGS+=(--report_to "$REPORT_TO")
fi

mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage4 train \
  "PRETRAINED_MODEL_NAME_OR_PATH=$PRETRAINED_MODEL_NAME_OR_PATH" \
  "TRAIN_DATA_ROOT=$TRAIN_DATA_ROOT" "STAGE1_CHECKPOINT=$STAGE1_CHECKPOINT" \
  "EVENT_LORA_CHECKPOINT=$EVENT_LORA_CHECKPOINT" \
  "RGB_EVENT_PRETRAINED_ROOT=$RGB_EVENT_PRETRAINED_ROOT" "OUTPUT_DIR=$OUTPUT_DIR" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" \
  "DECODER_REFINEMENT_VARIANT=$DECODER_REFINEMENT_VARIANT" \
  "CGDRS_EVENT_LEVELS=$CGDRS_EVENT_LEVELS" "CGDRS_GATE_BIAS=$CGDRS_GATE_BIAS" \
  "CGDRS_ACTIVITY_MODE=$CGDRS_ACTIVITY_MODE" \
  "CLORA_EVENT_ACTIVE_BINS=$CLORA_EVENT_ACTIVE_BINS" \
  "RGB_EVENT_ACTIVE_BINS=$RGB_EVENT_ACTIVE_BINS" "REPORT_TO=$REPORT_TO" \
  "IMAGE_SIZE=${IMAGE_SIZE:-256}" "GENERATOR_TIMESTEP=${GENERATOR_TIMESTEP:-500}" \
  "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}" "LEARNING_RATE=${LEARNING_RATE:-1e-4}"

echo "[Idea4 Stage4] output=$OUTPUT_DIR"
echo "[Idea4 Stage4] decoder_refinement_variant=$DECODER_REFINEMENT_VARIANT"
echo "[Idea4 Stage4] CGDRS event levels=$CGDRS_EVENT_LEVELS, gate bias=$CGDRS_GATE_BIAS"
echo "[Idea4 Stage4] active event bins: cLoRA=$CLORA_EVENT_ACTIVE_BINS, RGB-event=$RGB_EVENT_ACTIVE_BINS (total channels=5)"
echo "[Idea4 Stage4] freeze scope=vae.decoder.feature_fusion_blocks.* only"
echo "[Idea4 Stage4] losses: l2=${LAMBDA_L2:-1.0}, lpips=${LAMBDA_LPIPS:-0.1}, edge=${LAMBDA_EDGE:-0.1}"

accelerate launch --num_processes "$NPROC_PER_NODE" "$REPO_ROOT/train_idea3_decoder_refinement.py" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --stage1_checkpoint "$STAGE1_CHECKPOINT" \
  --event_lora_checkpoint "$EVENT_LORA_CHECKPOINT" \
  --rgb_event_pretrained_root "$RGB_EVENT_PRETRAINED_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --decoder_refinement_variant "$DECODER_REFINEMENT_VARIANT" \
  --cgdrs_event_levels "$CGDRS_EVENT_LEVELS" \
  --cgdrs_gate_bias "$CGDRS_GATE_BIAS" \
  --cgdrs_activity_mode "$CGDRS_ACTIVITY_MODE" \
  --clora_event_active_bins "$CLORA_EVENT_ACTIVE_BINS" \
  --rgb_event_active_bins "$RGB_EVENT_ACTIVE_BINS" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "$EVENT_LORA_MOD_CHANNELS" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --image_size "${IMAGE_SIZE:-256}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-20000}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-100}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS:-0}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-1000}" \
  --mixed_precision "${MIXED_PRECISION:-fp16}" \
  --tracker_project_name "${TRACKER_PROJECT_NAME:-OSEDiff_idea4}" \
  --wandb_run_name "${WANDB_RUN_NAME:-OSEDiff_stage4_9-15-2040}" \
  --generator_timestep "${GENERATOR_TIMESTEP:-500}" \
  --lambda_l2 "${LAMBDA_L2:-1.0}" \
  --lambda_lpips "${LAMBDA_LPIPS:-0.1}" \
  --lambda_edge "${LAMBDA_EDGE:-0.1}" \
  --lambda_vsd 0 \
  --lambda_vsd_lora 0 \
  --lambda_smooth 0 \
  --lambda_mod 0 \
  "${TRACKER_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
