#!/usr/bin/env bash
set -euo pipefail

# Stage5：从 Stage4 refinement decoder checkpoint 出发做受限全量微调。
# 冻结 UNet 主干、原始 VAE decoder 主体、文本编码器和 RGB/Event 辅助编码器；
# 解冻 UNet/VAE LoRA、EventEncoder、Event-LoRA 调制分支和 decoder fusion blocks。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_INSECURE="${WANDB_INSECURE:-true}"
export WANDB_INSECURE_DISABLE_SSL="${WANDB_INSECURE_DISABLE_SSL:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"

PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/public/home/slpchen/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/train}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/public/home/slpchen/OSEDiff_project/TrainingWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
STAGE4_CHECKPOINT="${STAGE4_CHECKPOINT:-/public/home/slpchen/OSEDiff_project/TrainingWeights/OSEDiff_stage4_8-12-2030/checkpoints/model_final.pkl}"
RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/public/home/slpchen/OSEDiff_project/TrainingWeights/HDRev}"
OUTPUT_DIR="${OUTPUT_DIR:-/public/home/slpchen/OSEDiff_project/TrainingWeights/OSEDiff_stage5_8-17-0800}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-auto}"
CGDRS_EVENT_LEVELS="${CGDRS_EVENT_LEVELS:-1,2,3}"
CGDRS_GATE_BIAS="${CGDRS_GATE_BIAS:--2.0}"
CGDRS_ACTIVITY_MODE="${CGDRS_ACTIVITY_MODE:-abs_mean}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
FIND_UNUSED_PARAMETERS="${FIND_UNUSED_PARAMETERS:-1}"

if [ -z "${EVENT_LORA_MOD_CHANNELS:-}" ]; then
  if [[ "$EVENT_LORA_VARIANT" == residual_cross_attn* ]]; then
    EVENT_LORA_MOD_CHANNELS=64
  else
    EVENT_LORA_MOD_CHANNELS=32
  fi
else
  EVENT_LORA_MOD_CHANNELS="$EVENT_LORA_MOD_CHANNELS"
fi

for required_dir in "$PRETRAINED_MODEL_NAME_OR_PATH" "$TRAIN_DATA_ROOT" "$RGB_EVENT_PRETRAINED_ROOT"; do
  if [ ! -d "$required_dir" ]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done
for required_file in "$STAGE1_CHECKPOINT" "$STAGE4_CHECKPOINT" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_I.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_E.pth"; do
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
if [[ "$FIND_UNUSED_PARAMETERS" == "1" ]]; then
  EXTRA_ARGS+=(--find_unused_parameters)
else
  EXTRA_ARGS+=(--no-find_unused_parameters)
fi

TRACKER_ARGS=()
if [ -n "${REPORT_TO:-wandb}" ]; then
  TRACKER_ARGS+=(--report_to "${REPORT_TO:-wandb}")
fi

mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage5 train \
  "PRETRAINED_MODEL_NAME_OR_PATH=$PRETRAINED_MODEL_NAME_OR_PATH" \
  "TRAIN_DATA_ROOT=$TRAIN_DATA_ROOT" "STAGE1_CHECKPOINT=$STAGE1_CHECKPOINT" \
  "STAGE4_CHECKPOINT=$STAGE4_CHECKPOINT" "RGB_EVENT_PRETRAINED_ROOT=$RGB_EVENT_PRETRAINED_ROOT" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" "DECODER_REFINEMENT_VARIANT=$DECODER_REFINEMENT_VARIANT" \
  "CGDRS_EVENT_LEVELS=$CGDRS_EVENT_LEVELS" "CGDRS_GATE_BIAS=$CGDRS_GATE_BIAS" \
  "CGDRS_ACTIVITY_MODE=$CGDRS_ACTIVITY_MODE" \
  "IMAGE_SIZE=${IMAGE_SIZE:-256}" "GENERATOR_TRAIN_TIMESTEP=${GENERATOR_TRAIN_TIMESTEP:-400,600}" \
  "GENERATOR_TIMESTEP=${GENERATOR_TIMESTEP:-500}" "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}" \
  "LEARNING_RATE=${LEARNING_RATE:-5e-6}"

echo "[Stage5] variant=$EVENT_LORA_VARIANT"
echo "[Stage5] decoder_refinement_variant=$DECODER_REFINEMENT_VARIANT"
echo "[Stage5] mod_channels=$EVENT_LORA_MOD_CHANNELS"
echo "[Stage5] stage4_checkpoint=$STAGE4_CHECKPOINT"
echo "[Stage5] output=$OUTPUT_DIR"
echo "[Stage5] trainable=unet_lora,event_encoder,vae_encoder_lora,decoder_feature_fusion"
echo "[Stage5] frozen=unet_backbone,vae_decoder_backbone,text_encoder,rgb_event_encoder"
echo "[Stage5] ddp_find_unused_parameters=$FIND_UNUSED_PARAMETERS"
echo "[Stage5] batch_size=${TRAIN_BATCH_SIZE:-6}, workers=${DATALOADER_NUM_WORKERS:-2}"
echo "[Stage5] losses: l2=${LAMBDA_L2:-1.0}, lpips=${LAMBDA_LPIPS:-0.1}, edge=${LAMBDA_EDGE:-0.1}"

accelerate launch --num_processes "$NPROC_PER_NODE" train_idea3_decoder_refinement_stage5.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --stage1_checkpoint "$STAGE1_CHECKPOINT" \
  --event_lora_checkpoint "$STAGE4_CHECKPOINT" \
  --rgb_event_pretrained_root "$RGB_EVENT_PRETRAINED_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --decoder_refinement_variant "$DECODER_REFINEMENT_VARIANT" \
  --cgdrs_event_levels "$CGDRS_EVENT_LEVELS" \
  --cgdrs_gate_bias "$CGDRS_GATE_BIAS" \
  --cgdrs_activity_mode "$CGDRS_ACTIVITY_MODE" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "$EVENT_LORA_MOD_CHANNELS" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --image_size "${IMAGE_SIZE:-256}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-6}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-20000}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-100}" \
  --learning_rate "${LEARNING_RATE:-5e-6}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS:-0}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-2000}" \
  --mixed_precision "${MIXED_PRECISION:-fp16}" \
  --tracker_project_name "${TRACKER_PROJECT_NAME:-OSEDiff_clora_stage5}" \
  --wandb_run_name "${WANDB_RUN_NAME:-stage5_8-17-0800}" \
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
