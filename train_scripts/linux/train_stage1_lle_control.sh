#!/usr/bin/env bash
set -euo pipefail

# Keep the torchrun process count no larger than the number of visible GPUs.
# This server has 2 GPUs by default; override NPROC_PER_NODE when the job gets a different allocation.
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/public/home/slpchen/stable-diffusion-2-1-base}"
TRAIN_ROOT="${TRAIN_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/train}"
TEST_ROOT="${TEST_ROOT:-/public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/test}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/event_guided_stage1_6-1-2100}"
OSEDIFF_LORA="${OSEDIFF_LORA:-checkpoints/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
LLE_PRETRAIN="${LLE_PRETRAIN:-checkpoints/HDRev}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" train_stage1_lle_control.py \
  --stage stage1 \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --train_root "${TRAIN_ROOT}" \
  --test_root "${TEST_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained_osediff_lora_path "${OSEDIFF_LORA}" \
  --pretrained_lle_encoder_path "${LLE_PRETRAIN}" \
  --train_timestep 500 \
  --num_bins 5 \
  --control_alpha_init 0.0 \
  --control_alpha_final 0.3 \
  --control_alpha_warmup_steps 10000 \
  --control_condition_scale 0.1 \
  --control_condition_clamp 5.0 \
  --train_batch_size 8 \
  --dataloader_num_workers 6 \
  --max_train_steps 100000 \
  --checkpointing_steps 1000 \
  --log_interval 10 \
  --use_wandb \
  --wandb_project event_guided_osediff \
  --wandb_run_name stage1_lle_control_6-1-2100 \
  --wandb_log_image_interval 5000 \
  --num_wandb_visual_samples 4 \
  --wandb_visual_timestep 500 \
  --auto_resume \
  --lambda_lpips 0.1
