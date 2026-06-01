#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/path/to/stable-diffusion-2-1-base}"
TRAIN_ROOT="${TRAIN_ROOT:-/path/to/SDE_dataset/SDE_indoor/sde_in_release/train}"
TEST_ROOT="${TEST_ROOT:-/path/to/SDE_dataset/SDE_indoor/sde_in_release/test}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/event_guided_stage2}"
OSEDIFF_LORA="${OSEDIFF_LORA:-checkpoints/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoints/event_guided_stage1/latest_model.pkl}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" train_stage2_lle_control.py \
  --stage stage2 \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --train_root "${TRAIN_ROOT}" \
  --test_root "${TEST_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --stage1_checkpoint_path "${STAGE1_CKPT}" \
  --pretrained_osediff_lora_path "${OSEDIFF_LORA}" \
  --train_timestep 500 \
  --num_bins 5 \
  --control_alpha_init 1.0 \
  --control_alpha_final 1.0 \
  --control_alpha_warmup_steps 1 \
  --lr 5e-5 \
  --lr_vae_lora 1e-5 \
  --lr_unet_lora 1e-5 \
  --train_batch_size 1 \
  --dataloader_num_workers 4 \
  --max_train_steps 100000 \
  --checkpointing_steps 1000 \
  --use_wandb \
  --wandb_project event_guided_osediff \
  --wandb_run_name stage2_lle_control \
  --wandb_log_image_interval 5000 \
  --num_wandb_visual_samples 4 \
  --wandb_visual_timestep 500 \
  --auto_resume
