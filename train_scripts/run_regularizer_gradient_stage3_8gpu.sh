#!/usr/bin/env bash
set -euo pipefail

# 八卡服务器的正则梯度诊断 / Stage3 短程对照。单卡 batch=1，累积16保持有效batch=16。
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/archive_run_context.sh"
PYTHON="${PYTHON:-/home/cjl/.conda/envs/osediff/bin/python}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_out_release/sde_out_release/train}"
INIT_FROM_EVENT_CHECKPOINT="${INIT_FROM_EVENT_CHECKPOINT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_EventOneChannel_stage2_9-19-2210/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a new experiment subdirectory under Test_results}"
PHASE="${PHASE:-probe}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

case "$CUDA_VISIBLE_DEVICES" in 0|1) ;; *) echo '[ERROR] Use one authorized GPU: 0 or 1.' >&2; exit 1 ;; esac
case "$OUTPUT_DIR" in /home/data1/cjl/OSEDiff_Project/Test_results/*) ;; *) echo '[ERROR] Output must stay in Test_results.' >&2; exit 1 ;; esac
for required_dir in "$PRETRAINED_MODEL" "$TRAIN_DATA_ROOT"; do
  [[ -d "$required_dir" ]] || { echo "[ERROR] Missing directory: $required_dir" >&2; exit 1; }
done
[[ -x "$PYTHON" && -f "$INIT_FROM_EVENT_CHECKPOINT" ]] || { echo '[ERROR] Python/checkpoint missing.' >&2; exit 1; }

PROBE_ARGS=()
case "$PHASE" in
  probe)
    MIXED_PRECISION=no
    ACCUMULATION=1
    LAMBDA_MOD=0.001
    LAMBDA_SMOOTH=0.001
    PROBE_ARGS+=(--regularizer_gradient_probe_batches "${PROBE_BATCHES:-20}")
    ;;
  on)
    MIXED_PRECISION=fp16
    ACCUMULATION="${GRADIENT_ACCUMULATION_STEPS:-16}"
    LAMBDA_MOD=0.001
    LAMBDA_SMOOTH=0.001
    ;;
  off)
    MIXED_PRECISION=fp16
    ACCUMULATION="${GRADIENT_ACCUMULATION_STEPS:-16}"
    LAMBDA_MOD=0
    LAMBDA_SMOOTH=0
    ;;
  *) echo '[ERROR] PHASE must be probe, on or off.' >&2; exit 1 ;;
esac
# 沿用 outdoor31 的结构、学习率、损失和 timestep；仅改变实验正则开关与筛查预算。
COMMAND=("$PYTHON" train_event_film_lora_osediff.py
  --stage 3 --pretrained_model_name_or_path "$PRETRAINED_MODEL"
  --train_data_root "$TRAIN_DATA_ROOT" --output_dir "$OUTPUT_DIR"
  --init_from_event_checkpoint "$INIT_FROM_EVENT_CHECKPOINT"
  --seed "${SEED:-123}" --image_size 256 --event_bins 5 --event_channels 1
  --train_batch_size "${TRAIN_BATCH_SIZE:-1}" --gradient_accumulation_steps "$ACCUMULATION"
  --max_train_steps "${MAX_TRAIN_STEPS:-1000}" --checkpointing_steps "${CHECKPOINTING_STEPS:-250}"
  --mixed_precision "$MIXED_PRECISION" --report_to none --dataloader_num_workers 0
  --event_lora_variant residual_cross_attn --lora_rank 4 --mod_channels 64 --modulation_scale 0.1
  --event_lora_cbam_reduction 8 --event_lora_cbam_spatial_kernel 7
  --regularizer_lora_mode same_as_generator
  --generator_train_timestep 400,600 --regularizer_train_timestep 400,600
  --lambda_vsd 0.5 --lambda_vsd_lora 0.5
  --learning_rate_event 5e-5 --learning_rate_lora 1e-5 --learning_rate_reg 5e-5
  --lambda_l2 1.0 --lambda_lpips 1.0 --lambda_edge 0.05
  --lambda_mod "$LAMBDA_MOD" --lambda_smooth "$LAMBDA_SMOOTH"
  "${PROBE_ARGS[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage3 "$PHASE" \
  "INIT_FROM_EVENT_CHECKPOINT=$INIT_FROM_EVENT_CHECKPOINT" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}" "GRADIENT_ACCUMULATION_STEPS=$ACCUMULATION" \
  "LAMBDA_MOD=$LAMBDA_MOD" "LAMBDA_SMOOTH=$LAMBDA_SMOOTH"
exec "${COMMAND[@]}"
