#!/usr/bin/env bash
set -euo pipefail

# 八卡服务器：使用单次 baseline 入口评估本次室内 Stage4 旧版 decoder checkpoint。
# 仅运行标准 baseline；不执行推理期特征纠错。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
export TEST_DATA_ROOT="${TEST_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_in_release/test}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage4_9-15-1920/checkpoints/model_final.pkl}"
# 必须与训练时使用的、时间戳修复后训练的 RGB-event encoder 权重一致。
export RGB_EVENT_PRETRAINED_ROOT="${RGB_EVENT_PRETRAINED_ROOT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/REencoder/SDE_indoor}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/data1/cjl/OSEDiff_Project/Test_results/idea4_stage4_9-15-1920_indoor}"

export EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
# 旧版 legacy_fused 需要 Merge.pth；显式设置可在启动前完成文件存在性校验。
export DECODER_REFINEMENT_ENABLED="${DECODER_REFINEMENT_ENABLED:-1}"
export DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-legacy_fused}"
# n=1 复现旧 timestamp bug 的 voxel 分布；通道数仍固定为 5。
export CLORA_EVENT_ACTIVE_BINS="${CLORA_EVENT_ACTIVE_BINS:-1}"
export RGB_EVENT_ACTIVE_BINS="${RGB_EVENT_ACTIVE_BINS:-1}"
export CALCULATE_FID="${CALCULATE_FID:-1}"
export SAVE_IMAGES="${SAVE_IMAGES:-1}"
export MIXED_PRECISION="${MIXED_PRECISION:-fp16}"

# 保留历史结果表名称及列布局；兼容模式不启用额外引导。
export BASELINE_CSV_FORMAT="legacy-baseline"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/test_scripts/test_event_baseline.sh"
