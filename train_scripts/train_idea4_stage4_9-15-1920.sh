#!/usr/bin/env bash
set -euo pipefail

# idea4 Stage4（SDE indoor）：冻结 Stage1 OSEDiff、Stage3 cLoRA 及修复时间戳
# bug 后训练的 RGB-event encoder，仅训练旧版 decoder feature-fusion blocks。
# 默认超参数沿用 train_idea3_decoder_refinement_stage4.sh（原 6-16 实验配置）。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_INSECURE="${WANDB_INSECURE:-true}"
export WANDB_INSECURE_DISABLE_SSL="${WANDB_INSECURE_DISABLE_SSL:-true}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"

# ===== 八卡服务器默认配置；可由同名环境变量覆盖 =====
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_in_release/train}"
# 此 Stage1 权重应与下方 6-16-1050 Stage3 checkpoint 的训练初始化保持一致。
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
EVENT_LORA_CHECKPOINT="${EVENT_LORA_CHECKPOINT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_clora_stage3_6-16-1050/checkpoints/model_final.pkl}"
# 修复 SDE event timestamp float32 精度问题后训练的室内 RGB-event encoder。
RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/REencoder/SDE_indoor}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage4_9-15-1950}"

EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
# 本次复现实验固定使用旧版 MultiScaleFusion + feature-fusion decoder。
DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-legacy_fused}"
# n=1 复现旧 timestamp bug：事件均集中在 bin 0，但 cLoRA/RGB-event encoder
# 仍分别接收 5 通道 voxel。可分别覆盖，做不对称的输入布局消融。
CLORA_EVENT_ACTIVE_BINS="${CLORA_EVENT_ACTIVE_BINS:-1}"
RGB_EVENT_ACTIVE_BINS="${RGB_EVENT_ACTIVE_BINS:-5}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
REPORT_TO="${REPORT_TO:-wandb}"

# residual_cross_attn Stage3 checkpoint 使用 64 个 modulation channels。
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
for required_file in "$STAGE1_CHECKPOINT" "$EVENT_LORA_CHECKPOINT" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_I.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_E.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Merge.pth"; do
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
  "CLORA_EVENT_ACTIVE_BINS=$CLORA_EVENT_ACTIVE_BINS" \
  "RGB_EVENT_ACTIVE_BINS=$RGB_EVENT_ACTIVE_BINS" "REPORT_TO=$REPORT_TO" \
  "IMAGE_SIZE=${IMAGE_SIZE:-256}" "GENERATOR_TIMESTEP=${GENERATOR_TIMESTEP:-500}" \
  "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}" "LEARNING_RATE=${LEARNING_RATE:-1e-4}"

echo "[Idea4 Stage4] output=$OUTPUT_DIR"
echo "[Idea4 Stage4] decoder_refinement_variant=$DECODER_REFINEMENT_VARIANT"
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
  --wandb_run_name "${WANDB_RUN_NAME:-OSEDiff_stage4_9-15-1950}" \
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
