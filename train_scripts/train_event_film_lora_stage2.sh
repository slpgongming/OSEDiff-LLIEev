#!/usr/bin/env bash
set -euo pipefail

# slpchen / idea4 / Stage 2：仅激活 K、V 的 Event-LoRA；冻结 A/B，训练事件编码器和调制头。
# 普通 UNet LoRA 的激活与训练范围均固定为 none。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "$SCRIPT_PATH")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/train_event_film_lora_osediff.py" || ! -f "$REPO_ROOT/scripts/archive_run_context.sh" ]]; then
  echo "[ERROR] REPO_ROOT is not the idea4 code directory: $REPO_ROOT" >&2
  exit 1
fi
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/slpchen/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/train}"
STAGE1_CKPT="${STAGE1_CKPT:-/public/home/slpchen/OSEDiff_project/TrainingWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/public/home/slpchen/OSEDiff_project/TrainingWeights/OSEDiff_idea4_kv_stage2}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
# The supplied bash used event_lora, which is a compatibility alias for same_as_generator.
REGULARIZER_LORA_MODE="${REGULARIZER_LORA_MODE:-same_as_generator}"
UNET_LORA_ACTIVE_PARTS="none"
UNET_LORA_TRAINABLE_PARTS="none"

if [[ ! -d "$PRETRAINED_MODEL" ]]; then
  echo "[ERROR] PRETRAINED_MODEL does not exist: $PRETRAINED_MODEL" >&2
  exit 1
fi
if [[ ! -d "$TRAIN_DATA_ROOT" ]]; then
  echo "[ERROR] TRAIN_DATA_ROOT does not exist: $TRAIN_DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$STAGE1_CKPT" ]]; then
  echo "[ERROR] STAGE1_CKPT does not exist: $STAGE1_CKPT" >&2
  exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] OUTPUT_DIR must not be empty." >&2
  exit 1
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" && "$RESUME_FROM_CHECKPOINT" != latest && ! -f "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "[ERROR] RESUME_FROM_CHECKPOINT does not exist: $RESUME_FROM_CHECKPOINT" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "$SCRIPT_PATH" stage2 train \
  "ACCOUNT=slpchen" "PRETRAINED_MODEL=$PRETRAINED_MODEL" \
  "TRAIN_DATA_ROOT=$TRAIN_DATA_ROOT" "STAGE1_CKPT=$STAGE1_CKPT" \
  "OUTPUT_DIR=$OUTPUT_DIR" \
  "EVENT_LORA_TARGETS=k,v" "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" \
  "REGULARIZER_LORA_MODE=$REGULARIZER_LORA_MODE" \
  "UNET_LORA_ACTIVE_PARTS=$UNET_LORA_ACTIVE_PARTS" \
  "UNET_LORA_TRAINABLE_PARTS=$UNET_LORA_TRAINABLE_PARTS" \
  "IMAGE_SIZE=256" "GENERATOR_TRAIN_TIMESTEP=${GENERATOR_TRAIN_TIMESTEP:-400,600}" \
  "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100000}" \
  "CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-1000}" \
  "LOCAL_LOG_DIR=$OUTPUT_DIR/logs" "TRACKER_RUN_NAME=OSEDiff_idea4_kv_stage2"

accelerate launch --num_processes "$NPROC_PER_NODE" "$REPO_ROOT/train_event_film_lora_osediff.py" \
  --stage 2 \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --image_size 256 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-3}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-100000}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-1000}" \
  --mixed_precision fp16 \
  --report_to tensorboard \
  --logging_dir logs \
  --tracker_project_name OSEDiff_idea4_kv_stage2 \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --event_lora_targets k,v \
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
  "${EXTRA_ARGS[@]}"
