#!/usr/bin/env bash
# idea6 首次双卡实验：冻结 6/90 号来源参数，仅训练 mask 和每层强度。
# 在已配置好的 CUDA Python 环境中启动；PYTHON_BIN 可显式指定该环境的解释器。
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/home/data/Chen_data/SDE_dataset/sde_in_release/train}"
RGB_CHECKPOINT="${RGB_CHECKPOINT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/osediff_lle_5-14-1935/checkpoints/model_94001.pkl}"
EVENT_CHECKPOINT="${EVENT_CHECKPOINT:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/OSEDiff_stage3_10-1-1700/checkpoints/model_final.pkl}"
RUN_NAME="${RUN_NAME:-idea6_mask_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/data1/cjl/OSEDiff_Project/TrainedWeights/${RUN_NAME}}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
SMOKE_STEPS="${SMOKE_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29616}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd -- "$CODE_ROOT"
[[ "$SMOKE_STEPS" =~ ^[1-9][0-9]*$ && "$MAX_TRAIN_STEPS" =~ ^[1-9][0-9]*$ && "$CHECKPOINTING_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "step settings must be positive integers" >&2; exit 2;
}
(( SMOKE_STEPS <= MAX_TRAIN_STEPS )) || { echo "SMOKE_STEPS exceeds MAX_TRAIN_STEPS" >&2; exit 2; }
[[ "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "batch size and gradient accumulation must be positive integers" >&2; exit 2;
}
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || { echo "Exactly two GPU indices are required" >&2; exit 2; }

# batch 为每卡 microbatch；默认两卡各 12、累积 1，有效 batch=24。
launch=("$PYTHON_BIN" -m accelerate.commands.launch --multi_gpu --num_processes 2
  --num_machines 1 --machine_rank 0 --main_process_port "$MAIN_PROCESS_PORT"
  --mixed_precision fp16 --num_cpu_threads_per_process 4
  train_idea6_mask_fusion.py)
common=(--pretrained_model_name_or_path "$PRETRAINED_MODEL" --train_data_root "$TRAIN_DATA_ROOT"
  --output_dir "$OUTPUT_DIR" --train_batch_size "$TRAIN_BATCH_SIZE" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --generator_timestep 500 --mixed_precision fp16 --gradient_checkpointing
  --learning_rate_mask 1e-4 --learning_rate_strength 1e-3
  --lambda_l2 1 --lambda_lpips 1 --lambda_edge 0.05
  --mask_channels 16 --max_strength 2 --legacy_event_lora_alpha 8
  --max_grad_norm 1 --logging_steps 10 --dataloader_num_workers 0
  --seed 123 --report_to tensorboard)

run_phase() {
  printf 'COMMAND:'
  printf ' %q' "${launch[@]}" "${common[@]}" "$@"
  printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "${launch[@]}" "${common[@]}" "$@"
  fi
}

if [[ "$DRY_RUN" != 1 ]]; then
  for directory in "$PRETRAINED_MODEL" "$TRAIN_DATA_ROOT"; do
    [[ -d "$directory" ]] || { echo "Missing directory: $directory" >&2; exit 2; }
  done
  if [[ -z "$RESUME_FROM_CHECKPOINT" ]]; then
    for weight in "$RGB_CHECKPOINT" "$EVENT_CHECKPOINT"; do
      [[ -f "$weight" ]] || { echo "Missing checkpoint: $weight" >&2; exit 2; }
    done
    [[ ! -e "$OUTPUT_DIR/run.json" ]] || { echo "Existing run; set RESUME_FROM_CHECKPOINT=latest" >&2; exit 2; }
  fi
  # 启动前再次检查指定 GPU，避免占用其他人的任务；不自动改用别的卡。
  "$PYTHON_BIN" - <<'PY'
import os, subprocess
ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(set(ids)) != 2:
    raise SystemExit('GPU indices must be distinct')
output = subprocess.check_output([
    'nvidia-smi', '--id=' + ','.join(ids),
    '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'
], text=True)
print(output, flush=True)
rows = [row.split(',') for row in output.strip().splitlines()]
if len(rows) != 2 or any(int(row[2].strip()) >= 500 for row in rows):
    raise SystemExit('Requested GPUs are not both free (<500 MiB each); launch stopped')
PY
  mkdir -p -- "$OUTPUT_DIR"
  exec > >(tee -a "$OUTPUT_DIR/launch.log") 2>&1
  trap 'code=$?; printf "%s\n" "$code" > "$OUTPUT_DIR/launcher_exit_code.txt"' EXIT
  printf 'Started (Asia/Shanghai): %s\nCode: %s\nOutput: %s\nGPUs: %s\n' \
    "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$CODE_ROOT" "$OUTPUT_DIR" "$CUDA_VISIBLE_DEVICES"
  printf 'Per-GPU batch: %s; accumulation: %s; effective batch: %s\n' \
    "$TRAIN_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$((2 * TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
  # 使用真实 CUDA 运算验证运行环境；导入成功本身不代表 GPU 内核能执行。
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import torch
import accelerate, diffusers, transformers, peft, lpips, tensorboard
assert torch.cuda.device_count() == 2
for device in range(2):
    torch.manual_seed(123)
    x = torch.randn(16, 16, device=f'cuda:{device}')
    y = x @ x.T
    assert torch.isfinite(y).all()
    print('WITNESS', device, list(y.shape), torch.cuda.get_device_name(device), flush=True)
print({name: md.version(name) for name in (
    'torch', 'accelerate', 'diffusers', 'transformers', 'peft', 'lpips', 'tensorboard'
)}, flush=True)
PY
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  run_phase --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    --max_train_steps "$MAX_TRAIN_STEPS" --checkpointing_steps "$CHECKPOINTING_STEPS"
else
  # 先运行 SMOKE_STEPS 步并保存，检查双卡反传、显存、序列化和恢复。
  run_phase --rgb_checkpoint "$RGB_CHECKPOINT" --event_checkpoint "$EVENT_CHECKPOINT" \
    --max_train_steps "$SMOKE_STEPS" --checkpointing_steps "$SMOKE_STEPS"
  if (( MAX_TRAIN_STEPS > SMOKE_STEPS )); then
    # 相同配置/输出目录继续至总计 5000 步；不重新开始训练。
    run_phase --resume_from_checkpoint latest \
      --max_train_steps "$MAX_TRAIN_STEPS" --checkpointing_steps "$CHECKPOINTING_STEPS"
  fi
fi
