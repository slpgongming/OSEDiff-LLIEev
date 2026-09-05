#!/usr/bin/env bash
set -euo pipefail

# Stage4（原 idea3 Phase 1B）：冻结已有 OSEDiff、cLoRA 与 RGB-event encoder，
# 仅训练 VAE decoder 的四级 feature-fusion blocks。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,6}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"

PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_out_release/train}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_origin_8-21-1100/checkpoints/model_final.pkl}"
EVENT_LORA_CHECKPOINT="${EVENT_LORA_CHECKPOINT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage3_8-25-0020/checkpoints/model_80000.pkl}"
RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/REencoder/SDE_outdoor}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage4_9-5-1100}"
EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-cgdrs}"
CGDRS_EVENT_LEVELS="${CGDRS_EVENT_LEVELS:-1,2,3}"
CGDRS_GATE_BIAS="${CGDRS_GATE_BIAS:--2.0}"
CGDRS_ACTIVITY_MODE="${CGDRS_ACTIVITY_MODE:-abs_mean}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
# 本次运行不启用任何 experiment tracker（既不使用 W&B，也不使用 TensorBoard）。
REPORT_TO=""

# residual_cross_attn 系列 checkpoint 的融合中间通道默认为 64，其他当前
# Stage4 变体沿用 32；仍可通过环境变量显式覆盖，且覆盖优先级最高。
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
for required_file in "$STAGE1_CHECKPOINT" "$EVENT_LORA_CHECKPOINT" \
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
mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage4 train \
  "PRETRAINED_MODEL_NAME_OR_PATH=$PRETRAINED_MODEL_NAME_OR_PATH" \
  "TRAIN_DATA_ROOT=$TRAIN_DATA_ROOT" "STAGE1_CHECKPOINT=$STAGE1_CHECKPOINT" \
  "EVENT_LORA_CHECKPOINT=$EVENT_LORA_CHECKPOINT" "RGB_EVENT_PRETRAINED_ROOT=$RGB_EVENT_PRETRAINED_ROOT" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" "DECODER_REFINEMENT_VARIANT=$DECODER_REFINEMENT_VARIANT" \
  "CGDRS_EVENT_LEVELS=$CGDRS_EVENT_LEVELS" "CGDRS_GATE_BIAS=$CGDRS_GATE_BIAS" \
  "CGDRS_ACTIVITY_MODE=$CGDRS_ACTIVITY_MODE" "REPORT_TO=disabled" \
  "IMAGE_SIZE=${IMAGE_SIZE:-256}" "GENERATOR_TIMESTEP=${GENERATOR_TIMESTEP:-500}" \
  "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-30000}" "LEARNING_RATE=${LEARNING_RATE:-1e-4}"

echo "[Stage4] variant=$EVENT_LORA_VARIANT"
echo "[Stage4] decoder_refinement_variant=$DECODER_REFINEMENT_VARIANT"
echo "[Stage4] mod_channels=$EVENT_LORA_MOD_CHANNELS"
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
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-30000}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-100}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS:-0}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-2000}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --generator_timestep "${GENERATOR_TIMESTEP:-500}" \
  --lambda_l2 "${LAMBDA_L2:-1.0}" \
  --lambda_lpips "${LAMBDA_LPIPS:-1.0}" \
  --lambda_edge "${LAMBDA_EDGE:-0.1}" \
  --lambda_vsd 0 \
  --lambda_vsd_lora 0 \
  --lambda_smooth 0 \
  --lambda_mod 0 \
  "${EXTRA_ARGS[@]}"
