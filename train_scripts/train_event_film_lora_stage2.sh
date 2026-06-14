# clora采用residual_cross_attn方案，不再做FiLM。
#!/usr/bin/env bash
set -euo pipefail

# Stage2：加载完整 Stage1 权重，冻结已有 OSEDiff/LoRA，只训练 EventEncoder 与 Event-LoRA 分支。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"
export WANDB_INSECURE=true
export WANDB_INSECURE_DISABLE_SSL=true

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/slpchen/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/train}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoints/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/OSEDiff_clora_stage2_6-14-2200}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
REGULARIZER_LORA_MODE="${REGULARIZER_LORA_MODE:-event_lora}"

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
  --wandb_run_name "${WANDB_RUN_NAME:-stage2_6-14-2200}" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
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
