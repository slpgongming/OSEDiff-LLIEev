#!/usr/bin/env bash
set -euo pipefail

# 八卡服务器：评估 Stage4 9-15-2040 的 CGDRS decoder refinement checkpoint。
# 默认仅执行 idea4 ESG baseline（guidance=0）；可覆盖 ESG_GUIDANCE_SCALES 进行 sweep。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
export TEST_DATA_ROOT="${TEST_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_in_release/test}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage4_9-15-2040/checkpoints/model_final.pkl}"
# 训练时使用的旧 timestamp-bug 条件 RGB-event encoder；CGDRS 不需要 Merge.pth。
export RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/HDRev}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/data1/cjl/OSEDiff_Project/Test_results/idea4_stage4_9-15-2040_indoor}"

export EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
export DECODER_REFINEMENT_ENABLED="${DECODER_REFINEMENT_ENABLED:-1}"
# 显式匹配训练 checkpoint，避免测试时使用 legacy_fused 或错误的 CGDRS 配置。
export DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-cgdrs}"
export CGDRS_EVENT_LEVELS="${CGDRS_EVENT_LEVELS:-2,3}"
export CGDRS_GATE_BIAS="${CGDRS_GATE_BIAS:--3}"
export CGDRS_ACTIVITY_MODE="${CGDRS_ACTIVITY_MODE:-abs_mean}"
# 训练时 cLoRA 与 RGB-event encoder 均使用 bin-0 voxel，通道数始终为 5。
export CLORA_EVENT_ACTIVE_BINS="${CLORA_EVENT_ACTIVE_BINS:-1}"
export RGB_EVENT_ACTIVE_BINS="${RGB_EVENT_ACTIVE_BINS:-1}"

export ESG_GUIDANCE_SCALES="${ESG_GUIDANCE_SCALES:-0}"
export CALCULATE_FID="${CALCULATE_FID:-1}"
export SAVE_IMAGES="${SAVE_IMAGES:-1}"
export MIXED_PRECISION="${MIXED_PRECISION:-fp16}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/test_scripts/test_idea4_esg.sh"
