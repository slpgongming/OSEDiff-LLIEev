#!/usr/bin/env bash
set -euo pipefail

# EventOneChannel / Stage2：从完整 Stage1 LoRA 组合开始训练。
# cLoRA EventEncoder 使用单通道聚合事件；RE encoder 保留预训练兼容的 5 bins。
# 使用 residual_cross_attn；只训练 EventEncoder 与 Event-LoRA 事件调制分支。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"
export WANDB_INSECURE=true
export WANDB_INSECURE_DISABLE_SSL=true

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

# slpchen 账号的室内训练数据和 Stable Diffusion 2.1 基座。
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_in_release/train}"
STAGE1_CKPT="${STAGE1_CKPT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_EventOneChannel_stage2_9-19-2120}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# 复现 indoor34 配置；显式给出数值，避免 idea4 的新默认 mod_channels=32 改变实验设置。
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
LORA_RANK="${LORA_RANK:-4}"
EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-64}"
# idea4 已将旧 event_lora 模式统一为 same_as_generator。
REGULARIZER_LORA_MODE="${REGULARIZER_LORA_MODE:-same_as_generator}"
EVENT_BINS="${EVENT_BINS:-5}"
EVENT_CHANNELS="${EVENT_CHANNELS:-1}"
RGB_EVENT_CHANNELS="${RGB_EVENT_CHANNELS:-5}"

if [[ "$EVENT_BINS" != "5" || "$EVENT_CHANNELS" != "1" || "$RGB_EVENT_CHANNELS" != "5" ]]; then
  echo "[ERROR] EventOneChannel requires EVENT_BINS=5, EVENT_CHANNELS=1, and RGB_EVENT_CHANNELS=5." >&2
  exit 1
fi

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

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage2 train \
  "PRETRAINED_MODEL=$PRETRAINED_MODEL" "TRAIN_DATA_ROOT=$TRAIN_DATA_ROOT" \
  "STAGE1_CKPT=$STAGE1_CKPT" "OUTPUT_DIR=$OUTPUT_DIR" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" "LORA_RANK=$LORA_RANK" \
  "EVENT_LORA_MOD_CHANNELS=$EVENT_LORA_MOD_CHANNELS" \
  "REGULARIZER_LORA_MODE=$REGULARIZER_LORA_MODE" \
  "EVENT_BINS=$EVENT_BINS" "EVENT_CHANNELS=$EVENT_CHANNELS" \
  "RGB_EVENT_CHANNELS=$RGB_EVENT_CHANNELS" "IMAGE_SIZE=256" \
  "GENERATOR_TRAIN_TIMESTEP=${GENERATOR_TRAIN_TIMESTEP:-400,600}" \
  "REGULARIZER_TRAIN_TIMESTEP=${REGULARIZER_TRAIN_TIMESTEP:-400,600}" \
  "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100000}" \
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
  --checkpointing_steps "${CHECKPOINTING_STEPS:-500}" \
  --mixed_precision fp16 \
  --report_to wandb \
  --tracker_project_name OSEDiff_clora_stage2 \
  --wandb_run_name "${WANDB_RUN_NAME:-EventOneChannel_stage2_9-19-2120}" \
  --event_bins "$EVENT_BINS" \
  --event_channels "$EVENT_CHANNELS" \
  --rgb_event_channels "$RGB_EVENT_CHANNELS" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --lora_rank "$LORA_RANK" \
  --mod_channels "$EVENT_LORA_MOD_CHANNELS" \
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
