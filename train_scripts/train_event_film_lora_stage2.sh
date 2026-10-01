#!/usr/bin/env bash
set -euo pipefail

# zwy / Stage 2：普通 UNet LoRA 关闭并冻结，Event-LoRA A/B 冻结；训练 EventEncoder 与事件调制分支。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/zhouweiyu/Chen_Project/PretrainedModel/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/public/home/zhouweiyu/Chen_Project/dataset/SDE/indoor/sde_in_release/train}"
STAGE1_CKPT="${STAGE1_CKPT:-/public/home/zhouweiyu/Chen_Project/OSEDiff_project/TrainingWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/OSEDiff_stage2_9-30-1140}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
REGULARIZER_LORA_MODE="${REGULARIZER_LORA_MODE:-same_as_generator}"
UNET_LORA_ACTIVE_PARTS="${UNET_LORA_ACTIVE_PARTS:-none}"
UNET_LORA_TRAINABLE_PARTS="${UNET_LORA_TRAINABLE_PARTS:-none}"
CLORA_EVENT_ACTIVE_BINS="${CLORA_EVENT_ACTIVE_BINS:-5}"

if [ ! -d "$PRETRAINED_MODEL" ]; then
  echo "[ERROR] PRETRAINED_MODEL does not exist: $PRETRAINED_MODEL" >&2
  exit 1
fi
if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "[ERROR] TRAIN_DATA_ROOT does not exist: $TRAIN_DATA_ROOT" >&2
  exit 1
fi
if [ ! -f "$STAGE1_CKPT" ]; then
  echo "[ERROR] STAGE1_CKPT does not exist: $STAGE1_CKPT" >&2
  exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
  echo "[ERROR] OUTPUT_DIR is empty. Set the Stage2 output directory before training." >&2
  exit 1
fi

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "$SCRIPT_PATH" stage2 train \
  "PRETRAINED_MODEL=$PRETRAINED_MODEL" "TRAIN_DATA_ROOT=$TRAIN_DATA_ROOT" \
  "STAGE1_CKPT=$STAGE1_CKPT" "OUTPUT_DIR=$OUTPUT_DIR" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" "REGULARIZER_LORA_MODE=$REGULARIZER_LORA_MODE" \
  "UNET_LORA_ACTIVE_PARTS=$UNET_LORA_ACTIVE_PARTS" \
  "UNET_LORA_TRAINABLE_PARTS=$UNET_LORA_TRAINABLE_PARTS" \
  "CLORA_EVENT_ACTIVE_BINS=$CLORA_EVENT_ACTIVE_BINS" "IMAGE_SIZE=256" \
  "GENERATOR_TRAIN_TIMESTEP=${GENERATOR_TRAIN_TIMESTEP:-400,600}" \
  "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100000}" \
  "CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-2000}" \
  "LOCAL_LOG_DIR=$OUTPUT_DIR/logs" \
  "LR_EVENT=${LR_EVENT:-5e-5}" "LR_LORA=${LR_LORA:-1e-5}"

accelerate launch --num_processes "$NPROC_PER_NODE" train_event_film_lora_osediff.py \
  --stage 2 \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --image_size 256 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-100000}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-2000}" \
  --mixed_precision fp16 \
  --report_to tensorboard \
  --logging_dir logs \
  --tracker_project_name stage2_9-30-1140 \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --unet_lora_active_parts "$UNET_LORA_ACTIVE_PARTS" \
  --unet_lora_trainable_parts "$UNET_LORA_TRAINABLE_PARTS" \
  --clora_event_active_bins "$CLORA_EVENT_ACTIVE_BINS" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "${EVENT_LORA_MOD_CHANNELS:-64}" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --regularizer_lora_mode "$REGULARIZER_LORA_MODE" \
  --generator_train_timestep "${GENERATOR_TRAIN_TIMESTEP:-400,600}" \
  --regularizer_train_timestep "${REGULARIZER_TRAIN_TIMESTEP:-400,600}" \
  --lambda_vsd "${LAMBDA_VSD:-0.0}" \
  --lambda_vsd_lora "${LAMBDA_VSD_LORA:-0.0}" \
  --learning_rate_event "${LR_EVENT:-5e-5}" \
  --learning_rate_lora "${LR_LORA:-1e-5}" \
  --learning_rate_reg "${LR_REG:-5e-5}" \
  --lambda_lpips "${LAMBDA_LPIPS:-1.0}" \
  --lambda_edge "${LAMBDA_EDGE:-0.02}" \
  --lambda_smooth "${LAMBDA_SMOOTH:-0.005}" \
  --lambda_mod "${LAMBDA_MOD:-0.01}" \
  "${EXTRA_ARGS[@]}"
