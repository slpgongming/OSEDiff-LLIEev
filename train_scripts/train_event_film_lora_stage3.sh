#!/usr/bin/env bash
set -euo pipefail

# whb / Stage3：从同 variant 的 Stage2 checkpoint 初始化，联合训练 Event branch 与 LoRA A/B。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_INSECURE=true
export WANDB_INSECURE_DISABLE_SSL=true

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/whb/Chen_Project/PretrainedModel/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/train}"
STAGE1_CKPT="${STAGE1_CKPT:-/public/home/whb/Chen_Project/OSEDiff_project/TrainingWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
# 请在启动前填写 Stage2 checkpoint；Stage3 将从该 Event-LoRA 权重初始化。
INIT_FROM_EVENT_CHECKPOINT="${INIT_FROM_EVENT_CHECKPOINT:-}"
# 请在启动前填写本次 Stage3 权重输出目录，或通过同名环境变量覆盖。
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
EVENT_LORA_TARGETS="${EVENT_LORA_TARGETS:-auto}"
REGULARIZER_LORA_MODE="${REGULARIZER_LORA_MODE:-same_as_generator}"
UNET_LORA_ACTIVE_PARTS="${UNET_LORA_ACTIVE_PARTS:-none}"
UNET_LORA_TRAINABLE_PARTS="${UNET_LORA_TRAINABLE_PARTS:-none}"

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
  echo "[ERROR] OUTPUT_DIR is empty. Set the Stage3 output directory before training." >&2
  exit 1
fi
if [ -z "$RESUME_FROM_CHECKPOINT" ] && [ -z "$INIT_FROM_EVENT_CHECKPOINT" ]; then
  echo "[ERROR] INIT_FROM_EVENT_CHECKPOINT is empty. Set the Stage2 checkpoint before Stage3 training." >&2
  exit 1
fi
if [ -z "$RESUME_FROM_CHECKPOINT" ] && [ ! -f "$INIT_FROM_EVENT_CHECKPOINT" ]; then
  echo "[ERROR] INIT_FROM_EVENT_CHECKPOINT does not exist: $INIT_FROM_EVENT_CHECKPOINT" >&2
  exit 1
fi

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
else
  EXTRA_ARGS+=(--init_from_event_checkpoint "$INIT_FROM_EVENT_CHECKPOINT")
fi

mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage3 train \
  "PRETRAINED_MODEL=$PRETRAINED_MODEL" "TRAIN_DATA_ROOT=$TRAIN_DATA_ROOT" \
  "STAGE1_CKPT=$STAGE1_CKPT" "INIT_FROM_EVENT_CHECKPOINT=$INIT_FROM_EVENT_CHECKPOINT" \
  "OUTPUT_DIR=$OUTPUT_DIR" "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" "EVENT_LORA_TARGETS=$EVENT_LORA_TARGETS" \
  "UNET_LORA_ACTIVE_PARTS=$UNET_LORA_ACTIVE_PARTS" "UNET_LORA_TRAINABLE_PARTS=$UNET_LORA_TRAINABLE_PARTS" \
  "IMAGE_SIZE=256" "GENERATOR_TRAIN_TIMESTEP=${GENERATOR_TRAIN_TIMESTEP:-400,600}" \
  "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100000}" \
  "LR_EVENT=${LR_EVENT:-5e-5}" "LR_LORA=${LR_LORA:-1e-5}"

accelerate launch --num_processes "$NPROC_PER_NODE" train_event_film_lora_osediff.py \
  --stage 3 \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --image_size 256 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-100000}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-1000}" \
  --mixed_precision fp16 \
  --report_to wandb \
  --tracker_project_name event_film_lora_stage3 \
  --wandb_run_name "${WANDB_RUN_NAME:-cbam_lowrank_residual_stage3_6-19}" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --event_lora_targets "$EVENT_LORA_TARGETS" \
  --unet_lora_active_parts "$UNET_LORA_ACTIVE_PARTS" \
  --unet_lora_trainable_parts "$UNET_LORA_TRAINABLE_PARTS" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "${EVENT_LORA_MOD_CHANNELS:-64}" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --regularizer_lora_mode "$REGULARIZER_LORA_MODE" \
  --generator_train_timestep "${GENERATOR_TRAIN_TIMESTEP:-400,600}" \
  --regularizer_train_timestep "${REGULARIZER_TRAIN_TIMESTEP:-400,600}" \
  --lambda_vsd "${LAMBDA_VSD:-0.5}" \
  --lambda_vsd_lora "${LAMBDA_VSD_LORA:-0.5}" \
  --learning_rate_event "${LR_EVENT:-5e-5}" \
  --learning_rate_lora "${LR_LORA:-1e-5}" \
  --learning_rate_reg "${LR_REG:-5e-5}" \
  --lambda_lpips "${LAMBDA_LPIPS:-1.0}" \
  --lambda_edge "${LAMBDA_EDGE:-0.05}" \
  --lambda_smooth "${LAMBDA_SMOOTH:-0.001}" \
  --lambda_mod "${LAMBDA_MOD:-0.001}" \
  --enable_xformers_memory_efficient_attention \
  "${EXTRA_ARGS[@]}"
