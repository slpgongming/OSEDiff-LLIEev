#!/usr/bin/env bash
set -euo pipefail

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/path/to/stable-diffusion-2-1-base}"
TEST_ROOT="${TEST_ROOT:-/path/to/SDE_dataset/SDE_indoor/sde_in_release/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/event_guided_stage2/latest_model.pkl}"
OSEDIFF_LORA="${OSEDIFF_LORA:-checkpoints/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/event_guided_stage2}"

python test_event_guided_osediff.py \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --test_root "${TEST_ROOT}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --pretrained_osediff_lora_path "${OSEDIFF_LORA}" \
  --output_dir "${OUTPUT_DIR}" \
  --infer_timestep 500 \
  --event_mode normal \
  --save_visualization \
  --calculate_lpips
