@echo off
set PRETRAINED_MODEL=D:\path\to\stable-diffusion-2-1-base
set TRAIN_ROOT=D:\datasets\SDE_dataset\SDE_indoor\sde_in_release\train
set OUTPUT_DIR=checkpoints\event_guided_stage1_smoke
set OSEDIFF_LORA=checkpoints\osediff_lle_5-14-1935\checkpoints\model_94001.pkl
set LLE_PRETRAIN=checkpoints\HDRev

python train_stage1_lle_control.py ^
  --stage stage1 ^
  --pretrained_model_name_or_path "%PRETRAINED_MODEL%" ^
  --train_root "%TRAIN_ROOT%" ^
  --output_dir "%OUTPUT_DIR%" ^
  --pretrained_osediff_lora_path "%OSEDIFF_LORA%" ^
  --pretrained_lle_encoder_path "%LLE_PRETRAIN%" ^
  --train_timestep 500 ^
  --num_bins 5 ^
  --train_batch_size 1 ^
  --max_train_samples 1 ^
  --dry_run
