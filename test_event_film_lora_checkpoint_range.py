"""按 10k 整数倍批量评测 checkpoint，仅保留总体四项指标。"""

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


METRICS = ("psnr", "ssim", "lpips", "fid")
FIELDS = ("step", "checkpoint", "mode", "count", *METRICS)


def checkpoint_steps(start_k, end_k, interval_k):
    """k 单位范围包含终点；终点不在等差序列上时不会额外添加。"""
    for name, value in (("start_k", start_k), ("end_k", end_k), ("interval_k", interval_k)):
        if value < 0 or value % 10 or (name == "interval_k" and value == 0):
            raise ValueError(f"{name} must be a nonnegative multiple of 10; interval_k must be positive (unit: k).")
    if end_k < start_k:
        raise ValueError("end_k must be greater than or equal to start_k.")
    return [k * 1000 for k in range(start_k, end_k + 1, interval_k)]


def select_checkpoints(result_dir, steps, skip_missing=False):
    root = Path(result_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Training result directory does not exist: {root}")
    # 可传训练结果目录，也可直接传其 checkpoints 子目录。
    checkpoint_dir = root / "checkpoints" if (root / "checkpoints").is_dir() else root
    selected = [(step, checkpoint_dir / f"model_{step}.pkl") for step in steps]
    missing = [path for _, path in selected if not path.is_file()]
    if missing and not skip_missing:
        raise ValueError("Missing checkpoints:\n" + "\n".join(str(path) for path in missing))
    for path in missing:
        print(f"[Skip] Missing checkpoint: {path}")
    selected = [(step, path) for step, path in selected if path.is_file()]
    if not selected:
        raise ValueError("No checkpoints selected for evaluation.")
    return selected


def read_metrics(path, checkpoint, step, modes):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    if len(source) != len(modes) or {row.get("mode") for row in source} != set(modes):
        raise ValueError(f"Unexpected modes in evaluation CSV: {path}")
    result = []
    by_mode = {row["mode"]: row for row in source}
    for mode in modes:
        row = by_mode[mode]
        values = {name: float(row[name]) for name in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite metrics for {checkpoint}, mode={mode}: {values}")
        count = int(row["count"])
        if count <= 0:
            raise ValueError(f"Empty evaluation for {checkpoint}, mode={mode}.")
        result.append(dict(step=step, checkpoint=str(checkpoint), mode=mode, count=count, **values))
    return result


def save_metrics(path, rows):
    # 原子替换，避免中断时把已有汇总写成半个文件。
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                     dir=path.parent, suffix=".csv", delete=False) as handle:
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="其余参数传给新建的 test_event_film_lora_metrics.py，例如 --stage 3、--input_image、--multi_gpu。",
        allow_abbrev=False,
    )
    parser.add_argument("--training_result_dir", required=True,
                        help="一次训练的结果目录或其 checkpoints 子目录。")
    parser.add_argument("--start_k", type=int, required=True, help="起点，单位 k，须为 10 的非负整数倍。")
    parser.add_argument("--end_k", type=int, required=True, help="终点，单位 k，须为 10 的非负整数倍。")
    parser.add_argument("--interval_k", type=int, default=10, help="间隔，单位 k，默认 10。")
    parser.add_argument("--output_dir", required=True, help="保存 checkpoint_metrics.csv 的目录。")
    parser.add_argument("--ablation_modes", default="correct,zero")
    parser.add_argument("--skip_missing", action="store_true",
                        help="跳过缺失 checkpoint；默认在开始推理前报告全部缺失文件。")
    args, evaluation_args = parser.parse_known_args(argv)
    modes = [mode.strip() for mode in args.ablation_modes.split(",") if mode.strip()]
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"correct", "zero"}:
        parser.error("ablation_modes must be correct, zero, or correct,zero without duplicates.")
    # 以下参数由批量入口控制，防止覆盖权重或重新启用图片保存。
    reserved = {"--osediff_path", "--output_dir", "-o", "--_worker_mode",
                "--save_debug_vis", "--metrics_only", "--calculate_fid"}
    for arg in evaluation_args:
        if arg.split("=", 1)[0] in reserved:
            parser.error(f"{arg} is controlled by the checkpoint-range evaluator.")
    return args, evaluation_args, modes


def main(argv=None):
    args, evaluation_args, modes = parse_args(argv)
    selected = select_checkpoints(
        args.training_result_dir,
        checkpoint_steps(args.start_k, args.end_k, args.interval_k),
        args.skip_missing,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "checkpoint_metrics.csv"
    evaluator = Path(__file__).resolve().with_name("test_event_film_lora_metrics.py")
    rows = []
    for index, (step, checkpoint) in enumerate(selected, 1):
        print(f"[Checkpoint {index}/{len(selected)}] step={step}: {checkpoint}", flush=True)
        # 子进程结束后释放上一权重的显存；临时目录只用于四项指标的交接。
        with tempfile.TemporaryDirectory(prefix="checkpoint_eval_", dir=output_dir) as temporary_dir:
            command = [sys.executable, str(evaluator), *evaluation_args,
                       "--osediff_path", str(checkpoint), "--output_dir", temporary_dir,
                       "--ablation_modes", ",".join(modes), "--calculate_fid"]
            subprocess.run(command, check=True)
            rows.extend(read_metrics(
                Path(temporary_dir) / "metrics_event_film_ablation_overall.csv",
                checkpoint, step, modes,
            ))
        save_metrics(summary_path, rows)
        print(f"[Saved] {summary_path}", flush=True)
    print(f"[Done] Evaluated {len(selected)} checkpoints; metrics: {summary_path}")


if __name__ == "__main__":
    main()
