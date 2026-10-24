#!/usr/bin/env bash
set -euo pipefail

# 八卡服务器：SDE outdoor，Stage4 9-2-2330 的旧版 decoder refinement；仅测原模型 baseline。
export CUDA_VISIBLE_DEVICES="1"
export PRETRAINED_MODEL_NAME_OR_PATH="/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base"
export TEST_DATA_ROOT="/home/data/Chen_data/SDE_dataset/sde_out_release/sde_out_release/test"
export CHECKPOINT_PATH="/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage4_9-2-2330/checkpoints/model_final.pkl"
export RGB_EVENT_PRETRAINED_ROOT="/home/data1/cjl/OSEDiff_Project/TrainedWeights/REencoder/SDE_outdoor"
export OUTPUT_DIR="/home/data1/cjl/OSEDiff_Project/Test_results/idea4_stage4_9-2-2330_outdoor_baseline"

# 该权重采用旧版 legacy_fused decoder refinement；只运行原模型 baseline。
export DECODER_REFINEMENT_ENABLED="1"
export DECODER_REFINEMENT_VARIANT="legacy_fused"
export CALCULATE_FID="1"
export SAVE_IMAGES="0"

# 保留历史结果表名称及列布局；兼容模式不启用额外引导。
export BASELINE_CSV_FORMAT="legacy-baseline"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/test_scripts/test_event_baseline.sh"
