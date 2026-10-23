#!/usr/bin/env bash
set -euo pipefail

# whb / SDE indoor：从 EventOneChannel Stage2 权重初始化新的 Stage3。
# 旧激活方式：普通 down/up LoRA 关闭，default_others 前向激活但冻结；
# 训练所有独立 attn1.to_v Event-LoRA A/B、事件调制/共享 heads 和 EventEncoder。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
# whb 默认实时上传 W&B；仍允许通过 WANDB_MODE 环境变量显式覆盖。
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"
export WANDB_INSECURE=true
export WANDB_INSECURE_DISABLE_SSL=true

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"
cd "$REPO_ROOT"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/whb/Chen_Project/PretrainedModel/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/train}"
# Stage2 初始化时不会重新读取 Stage1；保留本批 Stage2 使用的权重身份与参考脚本的存在性检查。
STAGE1_CKPT="${STAGE1_CKPT:-/public/home/whb/Chen_Project/OSEDiff_project/TrainingWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
INIT_FROM_EVENT_CHECKPOINT="${INIT_FROM_EVENT_CHECKPOINT:-/public/home/whb/Chen_Project/OSEDiff_project/TrainingWeights/EventOneChannel_stage2_9-29-2040/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/public/home/whb/Chen_Project/OSEDiff_project/TrainingWeights/EventOneChannel_stage3_10-1-1440}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR}"

EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
LORA_RANK="${LORA_RANK:-4}"
EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-64}"
REGULARIZER_LORA_MODE="${REGULARIZER_LORA_MODE:-same_as_generator}"
EVENT_REGULARIZER_MODE="${EVENT_REGULARIZER_MODE:-legacy}"
UNET_ADAPTER_MODE="${UNET_ADAPTER_MODE:-others_only}"
EVENT_BINS="${EVENT_BINS:-5}"
EVENT_CHANNELS="${EVENT_CHANNELS:-1}"

if [[ "$EVENT_BINS" != "5" || "$EVENT_CHANNELS" != "1" ]]; then
  echo "[ERROR] EventOneChannel Stage3 requires EVENT_BINS=5 and EVENT_CHANNELS=1." >&2
  exit 1
fi
if [[ "$EVENT_LORA_VARIANT" != "residual_cross_attn" ]]; then
  echo "[ERROR] Stage3 EVENT_LORA_VARIANT must match the Stage2 residual_cross_attn checkpoint." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
else
  EXTRA_ARGS+=(--init_from_event_checkpoint "$INIT_FROM_EVENT_CHECKPOINT")
fi

COMMAND=(accelerate launch --num_processes "$NPROC_PER_NODE" train_event_film_lora_osediff.py \
  --stage 3 \
  --unet_adapter_mode "$UNET_ADAPTER_MODE" \
  --event_regularizer_mode "$EVENT_REGULARIZER_MODE" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --stage1_checkpoint "$STAGE1_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --image_size 256 \
  --event_bins "$EVENT_BINS" --event_channels "$EVENT_CHANNELS" --rgb_event_channels 5 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-100000}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-2000}" \
  --mixed_precision fp16 \
  --report_to wandb \
  --tracker_project_name "${TRACKER_PROJECT_NAME:-OSEDiff_clora_stage3}" \
  --wandb_run_name "${WANDB_RUN_NAME:-stage3_10-1-1440}" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --lora_rank "$LORA_RANK" \
  --mod_channels "$EVENT_LORA_MOD_CHANNELS" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --mod_num_heads "${MOD_NUM_HEADS:-4}" --mod_window_size "${MOD_WINDOW_SIZE:-8}" \
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
  --lambda_l2 "${LAMBDA_L2:-1.0}" \
  --lambda_lpips "${LAMBDA_LPIPS:-1.0}" \
  --lambda_edge "${LAMBDA_EDGE:-0.05}" \
  --lambda_smooth "${LAMBDA_SMOOTH:-0.001}" \
  --lambda_mod "${LAMBDA_MOD:-0.001}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-3}" \
  "${EXTRA_ARGS[@]}")

# dry-run 不检查远程文件、不建目录、不启动训练；供最终参数复核。
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
elif [[ "${DRY_RUN:-0}" != "0" ]]; then
  echo "[ERROR] DRY_RUN must be 0 or 1." >&2
  exit 1
fi

for required_dir in "$PRETRAINED_MODEL" "$TRAIN_DATA_ROOT"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done
if [[ ! -f "$STAGE1_CKPT" ]]; then
  echo "[ERROR] STAGE1_CKPT does not exist: $STAGE1_CKPT" >&2
  exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] OUTPUT_DIR must not be empty." >&2
  exit 1
fi
if [[ -z "$RESUME_FROM_CHECKPOINT" && ! -f "$INIT_FROM_EVENT_CHECKPOINT" ]]; then
  echo "[ERROR] INIT_FROM_EVENT_CHECKPOINT does not exist: $INIT_FROM_EVENT_CHECKPOINT" >&2
  exit 1
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" && "$RESUME_FROM_CHECKPOINT" != "latest" && ! -f "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "[ERROR] RESUME_FROM_CHECKPOINT does not exist: $RESUME_FROM_CHECKPOINT" >&2
  exit 1
fi
command -v accelerate >/dev/null || { echo "[ERROR] Activate the OSEDiff environment with accelerate installed." >&2; exit 1; }

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"
printf -v RESOLVED_COMMAND '%q ' "${COMMAND[@]}"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage3 train \
  "ACCOUNT=whb" "DATASET=SDE_indoor" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "NPROC_PER_NODE=$NPROC_PER_NODE" "STAGE1_CKPT=$STAGE1_CKPT" \
  "INIT_FROM_EVENT_CHECKPOINT=$INIT_FROM_EVENT_CHECKPOINT" \
  "OUTPUT_DIR=$OUTPUT_DIR" "RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT" \
  "WANDB_MODE=$WANDB_MODE" "WANDB_DIR=$WANDB_DIR" \
  "UNET_ADAPTER_MODE=$UNET_ADAPTER_MODE" "EVENT_REGULARIZER_MODE=$EVENT_REGULARIZER_MODE" \
  "WANDB_RUN_NAME=${WANDB_RUN_NAME:-stage3_10-1-1440}" "RESOLVED_COMMAND=$RESOLVED_COMMAND"
echo "[Stage3] Effective command: $RESOLVED_COMMAND"
exec "${COMMAND[@]}"
