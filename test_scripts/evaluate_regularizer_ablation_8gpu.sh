#!/usr/bin/env bash
set -euo pipefail

# 正则消融的固定跨场景子集评估；共享 VAE latent，测试 timestep 默认500。
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/archive_run_context.sh"
PYTHON="${PYTHON:-/home/cjl/.conda/envs/osediff/bin/python}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_out_release/sde_out_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_EventOneChannel_stage2_9-19-2210/checkpoints/model_final.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:?Set a new evaluation directory under Test_results}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "$CUDA_VISIBLE_DEVICES" in 0|1) ;; *) echo '[ERROR] Use GPU 0 or 1.' >&2; exit 1 ;; esac
case "$OUTPUT_DIR" in /home/data1/cjl/OSEDiff_Project/Test_results/*) ;; *) echo '[ERROR] Output must stay in Test_results.' >&2; exit 1 ;; esac
[[ -x "$PYTHON" && -f "$CHECKPOINT_PATH" && -d "$PRETRAINED_MODEL" && -d "$TEST_DATA_ROOT" ]] || { echo '[ERROR] Required input missing.' >&2; exit 1; }
COMMAND=("$PYTHON" evaluate_regularizer_ablation.py
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" --test_data_root "$TEST_DATA_ROOT"
  --osediff_path "$CHECKPOINT_PATH" --output_dir "$OUTPUT_DIR"
  --generator_timestep "${GENERATOR_TIMESTEP:-500}" --samples_per_scene "${SAMPLES_PER_SCENE:-16}"
  --event_bins 5 --event_channels 1 --image_size 256 --lora_rank 4
  --event_lora_variant residual_cross_attn --mod_channels 64 --modulation_scale 0.1
  --mixed_precision no --seed 20260928)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" stage3 paired_eval \
  "CHECKPOINT_PATH=$CHECKPOINT_PATH" "GENERATOR_TIMESTEP=${GENERATOR_TIMESTEP:-500}" \
  "SAMPLES_PER_SCENE=${SAMPLES_PER_SCENE:-16}" "SHARED_VAE_LATENT=1" "MIXED_PRECISION=no"
exec "${COMMAND[@]}"
