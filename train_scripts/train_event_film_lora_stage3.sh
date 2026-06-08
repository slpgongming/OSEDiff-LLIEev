#!/usr/bin/env bash
set -euo pipefail

# Stage3：从 Stage2 Event-FiLM checkpoint 初始化，联合训练 Event branch 与 attn1.to_v LoRA。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_INSECURE=true
export WANDB_INSECURE_DISABLE_SSL=true

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/path/to/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/path/to/SDE_indoor/sde_in_release/train}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoints/osediff_ConcatEv_5-23-2350/checkpoints/model_93501.pkl}"
INIT_FROM_EVENT_CHECKPOINT="${INIT_FROM_EVENT_CHECKPOINT:-experience/event_film_lora_stage2/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-experience/event_film_lora_stage3}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [ ! -d "$PRETRAINED_MODEL" ]; then
  echo "[ERROR] PRETRAINED_MODEL does not exist: $PRETRAINED_MODEL" >&2
  exit 1
fi
if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "[ERROR] TRAIN_DATA_ROOT does not exist: $TRAIN_DATA_ROOT" >&2
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

accelerate launch --num_processes "$NPROC_PER_NODE" train_event_film_lora_osediff.py \
  --stage 3 \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --image_size 256 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-100000}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-500}" \
  --mixed_precision fp16 \
  --report_to wandb \
  --tracker_project_name event_film_lora_stage3 \
  --wandb_run_name "${WANDB_RUN_NAME:-event_film_lora_stage3}" \
  --generator_train_timestep "${GENERATOR_TRAIN_TIMESTEP:-999}" \
  --regularizer_train_timestep "${REGULARIZER_TRAIN_TIMESTEP:-999}" \
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
