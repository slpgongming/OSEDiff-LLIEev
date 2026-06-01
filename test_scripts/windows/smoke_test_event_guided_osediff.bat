@echo off
set PRETRAINED_MODEL=D:\path\to\stable-diffusion-2-1-base
set TEST_ROOT=D:\datasets\SDE_dataset\SDE_indoor\sde_in_release\test
set CHECKPOINT_PATH=checkpoints\event_guided_stage1\latest_model.pkl
set OSEDIFF_LORA=checkpoints\osediff_lle_5-14-1935\checkpoints\model_94001.pkl
set OUTPUT_DIR=results\event_guided_smoke

python test_event_guided_osediff.py ^
  --pretrained_model_name_or_path "%PRETRAINED_MODEL%" ^
  --test_root "%TEST_ROOT%" ^
  --checkpoint_path "%CHECKPOINT_PATH%" ^
  --pretrained_osediff_lora_path "%OSEDIFF_LORA%" ^
  --output_dir "%OUTPUT_DIR%" ^
  --infer_timestep 500 ^
  --event_mode zero ^
  --disable_control ^
  --max_samples 1 ^
  --save_visualization
