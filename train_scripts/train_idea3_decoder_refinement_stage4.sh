#!/usr/bin/env bash
set -euo pipefail

# Stage4（原 idea3 Phase 1B）：冻结已有 OSEDiff、cLoRA 与 RGB-event encoder，
# 仅训练 VAE decoder 的四级 feature-fusion blocks。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_INSECURE="${WANDB_INSECURE:-true}"
export WANDB_INSECURE_DISABLE_SSL="${WANDB_INSECURE_DISABLE_SSL:-true}"

PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/public/home/slpchen/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/train}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/public/home/slpchen/Chen_Project/TrainingWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
EVENT_LORA_CHECKPOINT="${EVENT_LORA_CHECKPOINT:-/public/home/slpchen/Chen_Project/TrainingWeights/OSEDiff_stage3_6-28-1700/checkpoints/model_final.pkl}"
RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/public/home/slpchen/Chen_Project/TrainingWeights/HDRev}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/OSEDiff_cbam_lowrank_residual_stage4}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-cbam_lowrank_residual}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

for required_dir in "$PRETRAINED_MODEL_NAME_OR_PATH" "$TRAIN_DATA_ROOT" "$RGB_EVENT_PRETRAINED_ROOT"; do
  if [ ! -d "$required_dir" ]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done
for required_file in "$STAGE1_CHECKPOINT" "$EVENT_LORA_CHECKPOINT" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_I.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_E.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Merge.pth"; do
  if [ ! -f "$required_file" ]; then
    echo "[ERROR] Required file does not exist: $required_file" >&2
    exit 1
  fi
done

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
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
if [ -n "${REPORT_TO:-wandb}" ]; then
  TRACKER_ARGS+=(--report_to "${REPORT_TO:-wandb}")
fi

echo "[Stage4] variant=$EVENT_LORA_VARIANT"
echo "[Stage4] output=$OUTPUT_DIR"
echo "[Stage4] freeze scope=vae.decoder.feature_fusion_blocks.* only"
echo "[Stage4] losses: l2=${LAMBDA_L2:-1.0}, lpips=${LAMBDA_LPIPS:-1.0}, edge=${LAMBDA_EDGE:-0.1}, vsd/vsd_lora/smooth/mod=0"

accelerate launch --num_processes "$NPROC_PER_NODE" train_idea3_decoder_refinement.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --stage1_checkpoint "$STAGE1_CHECKPOINT" \
  --event_lora_checkpoint "$EVENT_LORA_CHECKPOINT" \
  --rgb_event_pretrained_root "$RGB_EVENT_PRETRAINED_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "${EVENT_LORA_MOD_CHANNELS:-32}" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --image_size "${IMAGE_SIZE:-256}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-1}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-100000}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-100}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS:-0}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-500}" \
  --mixed_precision "${MIXED_PRECISION:-fp16}" \
  --tracker_project_name "${TRACKER_PROJECT_NAME:-OSEDiff_clora_stage3}" \
  --wandb_run_name "${WANDB_RUN_NAME:-stage3_6-16-1050}" \
  --generator_timestep "${GENERATOR_TIMESTEP:-999}" \
  --lambda_l2 "${LAMBDA_L2:-1.0}" \
  --lambda_lpips "${LAMBDA_LPIPS:-1.0}" \
  --lambda_edge "${LAMBDA_EDGE:-0.1}" \
  --lambda_vsd 0 \
  --lambda_vsd_lora 0 \
  --lambda_smooth 0 \
  --lambda_mod 0 \
  "${TRACKER_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
