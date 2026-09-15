#!/usr/bin/env bash
set -euo pipefail

# 八卡服务器：SDE indoor，Stage4 9-2-2250 的旧版 decoder refinement；仅测 idea4 ESG baseline。
export CUDA_VISIBLE_DEVICES="0"
export PRETRAINED_MODEL_NAME_OR_PATH="/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base"
export TEST_DATA_ROOT="/home/data/Chen_data/SDE_dataset/sde_in_release/test"
export CHECKPOINT_PATH="/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage4_9-2-2250/checkpoints/model_final.pkl"
export RGB_EVENT_PRETRAINED_ROOT="/home/data1/cjl/OSEDiff_Project/TrainedWeights/REencoder/SDE_indoor"
export OUTPUT_DIR="/home/data1/cjl/OSEDiff_Project/Test_results/idea4_stage4_9-2-2250_indoor_baseline"

# 该权重采用旧版 legacy_fused decoder refinement；guidance=0 时只运行 ESG baseline。
export DECODER_REFINEMENT_ENABLED="1"
export DECODER_REFINEMENT_VARIANT="legacy_fused"
export ESG_GUIDANCE_SCALES="0"
export CALCULATE_FID="1"
export SAVE_IMAGES="0"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/test_scripts/test_idea4_esg.sh"
