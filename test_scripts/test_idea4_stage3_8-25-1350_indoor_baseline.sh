#!/usr/bin/env bash
set -euo pipefail

# 八卡服务器：SDE indoor，Stage3 8-25-1350 的 70k 权重；仅测 idea4 ESG baseline。
export CUDA_VISIBLE_DEVICES="0"
export PRETRAINED_MODEL_NAME_OR_PATH="/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base"
export TEST_DATA_ROOT="/home/data/Chen_data/SDE_dataset/sde_in_release/test"
export CHECKPOINT_PATH="/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage3_8-25-1350/checkpoints/model_70000.pkl"
export OUTPUT_DIR="/home/data1/cjl/OSEDiff_Project/Test_results/idea4_stage3_8-25-1350_70k_indoor_baseline"

# Stage3 checkpoint 不包含 decoder refinement；guidance=0 时只运行 ESG baseline。
export DECODER_REFINEMENT_ENABLED="0"
export ESG_GUIDANCE_SCALES="0"
export CALCULATE_FID="1"
export SAVE_IMAGES="0"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/test_scripts/test_idea4_esg.sh"
