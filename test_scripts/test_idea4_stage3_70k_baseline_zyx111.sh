#!/usr/bin/env bash
set -euo pipefail

# Stage3 70k checkpoint 的可复现 baseline：仅运行单次 baseline。
# 此脚本面向 zyx111 当前服务器路径；如迁移机器，请修改本配置区域。

PRETRAINED_MODEL_NAME_OR_PATH="/public/home/zyx111/Chen_Project/PretrainedModel/stable-diffusion-2-1-base"
TEST_DATA_ROOT="/public/home/zyx111/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/test"
CHECKPOINT_PATH="/public/home/zyx111/Chen_Project/TrainingWeights/OSEDiff_stage3_8-25-1350/checkpoints/model_70000.pkl"
OUTPUT_DIR="/public/home/zyx111/Chen_Project/test_results/idea4_stage3_70k_seed42_baseline"

CUDA_VISIBLE_DEVICES="0"
CALCULATE_FID="1"
SAVE_IMAGES="0"
EVENT_LORA_VARIANT="residual_cross_attn"
EVENT_LORA_MOD_CHANNELS="64"
DECODER_REFINEMENT_ENABLED="0"
GENERATOR_TIMESTEP="500"

export PRETRAINED_MODEL_NAME_OR_PATH TEST_DATA_ROOT CHECKPOINT_PATH OUTPUT_DIR
export CUDA_VISIBLE_DEVICES CALCULATE_FID SAVE_IMAGES EVENT_LORA_VARIANT
export EVENT_LORA_MOD_CHANNELS DECODER_REFINEMENT_ENABLED GENERATOR_TIMESTEP

# 保留历史结果表名称及列布局；兼容模式不启用额外引导。
export BASELINE_CSV_FORMAT="legacy-baseline"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/test_scripts/test_event_baseline.sh"
