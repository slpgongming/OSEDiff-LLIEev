#!/usr/bin/env bash
set -euo pipefail

# Stage3 70k checkpoint 的可复现 baseline：仅运行 ESG guidance_scale=0。
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
ESG_SEED="42"
ESG_GUIDANCE_SCALES="0"
ESG_SWAP_RATIOS="0.05,0.10"
ESG_SWAP_MODES="spatial"
ESG_PAIR_POLICIES="dissimilar"
ESG_FEATURE_SCALES="32,16,8"
ESG_EXECUTION_MODE="batched"

export PRETRAINED_MODEL_NAME_OR_PATH TEST_DATA_ROOT CHECKPOINT_PATH OUTPUT_DIR
export CUDA_VISIBLE_DEVICES CALCULATE_FID SAVE_IMAGES EVENT_LORA_VARIANT
export EVENT_LORA_MOD_CHANNELS DECODER_REFINEMENT_ENABLED GENERATOR_TIMESTEP
export ESG_SEED ESG_GUIDANCE_SCALES ESG_SWAP_RATIOS ESG_SWAP_MODES
export ESG_PAIR_POLICIES ESG_FEATURE_SCALES ESG_EXECUTION_MODE

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/test_scripts/test_idea4_esg.sh"
