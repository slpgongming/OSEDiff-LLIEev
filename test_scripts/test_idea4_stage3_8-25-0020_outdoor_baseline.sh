#!/usr/bin/env bash
set -euo pipefail

# 八卡服务器：SDE outdoor，Stage3 8-25-0020 的 80k 权重；仅测原模型 baseline。
export CUDA_VISIBLE_DEVICES="1"
export PRETRAINED_MODEL_NAME_OR_PATH="/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base"
export TEST_DATA_ROOT="/home/data/Chen_data/SDE_dataset/sde_out_release/sde_out_release/test"
export CHECKPOINT_PATH="/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage3_8-25-0020/checkpoints/model_80000.pkl"
export OUTPUT_DIR="/home/data1/cjl/OSEDiff_Project/Test_results/idea4_stage3_8-25-0020_80k_outdoor_baseline"

# Stage3 checkpoint 不包含 decoder refinement；只运行原模型 baseline。
export DECODER_REFINEMENT_ENABLED="0"
export CALCULATE_FID="1"
export SAVE_IMAGES="0"

# 保留历史结果表名称及列布局；兼容模式不启用额外引导。
export BASELINE_CSV_FORMAT="legacy-baseline"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/test_scripts/test_event_baseline.sh"
