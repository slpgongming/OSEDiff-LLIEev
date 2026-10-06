"""只计算并保存总体 PSNR、SSIM、LPIPS、FID；不保存输出图像。"""

from pathlib import Path
import os
import shutil
import subprocess
import sys

import pyiqa
import torch

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from osediff_event_film import OSEDiffEventFiLMTest
from test_event_film_lora_osediff import (
    FIDEvaluator, MetricAccumulator, build_ablation_event, compute_quality_metrics,
    parse_ablation_modes, parse_args, read_csv, tensor_to_pil, write_csv,
)


CSV_NAME = "metrics_event_film_ablation_overall.csv"
FIELDS = ("mode", "count", "psnr", "ssim", "lpips", "fid")


def worker_command(mode):
    """每个 GPU 执行一个事件模式，始终调用本指标入口。"""
    forwarded = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--ablation_modes", "--_worker_mode"}:
            skip_next = True
        elif arg == "--multi_gpu":
            continue
        elif arg.startswith(("--ablation_modes=", "--_worker_mode=")):
            continue
        else:
            forwarded.append(arg)
    return [sys.executable, str(Path(__file__).resolve()), *forwarded,
            "--ablation_modes", mode, "--_worker_mode", mode]


def run_multi_gpu(args, modes):
    if len(modes) < 2:
        raise ValueError("--multi_gpu requires both correct and zero modes.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < len(modes):
        raise RuntimeError("Not enough visible CUDA GPUs for the requested modes.")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if item.strip()]
    if not visible:
        visible = [str(index) for index in range(torch.cuda.device_count())]
    if len(visible) < len(modes):
        raise RuntimeError("CUDA_VISIBLE_DEVICES has too few devices.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 双进程启动前预加载一次 FID 权重，避免并发下载同一缓存。
    warmup = FIDEvaluator([], torch.device("cuda"))
    try:
        metric = warmup._create_metric_with_cache_repair()
        del metric
    finally:
        warmup.cleanup()
    torch.cuda.empty_cache()
    worker_root = output_dir / ".multi_gpu_metrics"
    processes = []
    try:
        for mode, gpu_id in zip(modes, visible):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu_id, PYTHONUNBUFFERED="1")
            processes.append(subprocess.Popen(worker_command(mode), env=env))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Metric workers failed: {codes}")
        rows = []
        for mode in modes:
            mode_rows, fields = read_csv(worker_root / mode / CSV_NAME)
            if fields != list(FIELDS):
                raise ValueError(f"Unexpected metric fields: {fields}")
            rows.extend(mode_rows)
        write_csv(output_dir / CSV_NAME, rows, FIELDS)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        if worker_root.exists():
            shutil.rmtree(worker_root)


def evaluate(args, modes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SDEEventFilmDataset(split="test", args=args, root_dir=args.input_image)
    model = OSEDiffEventFiLMTest(args)
    psnr = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device)
    ssim = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device)
    lpips = pyiqa.create_metric("lpips", device=device).to(device)
    stats = {mode: MetricAccumulator() for mode in modes}
    fid = FIDEvaluator(modes, device)
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            lq = sample["conditioning_pixel_values"].unsqueeze(0).to(device)
            gt = sample["output_pixel_values"].unsqueeze(0).to(device)
            events = sample["events"].unsqueeze(0).to(device)
            fid.add_ground_truth(index, tensor_to_pil(gt[0]))
            for mode in modes:
                event_input = build_ablation_event(events, mode)
                with torch.inference_mode():
                    pred = model(lq, event_input, prompt=args.fixed_prompt)
                    values = compute_quality_metrics(pred, gt, psnr, ssim, lpips)
                stats[mode].update(values)
                # FID 沿用原入口的量化及指标定义，暂存文件由 finally 清理。
                fid.add_prediction(mode, index, tensor_to_pil(pred[0]))
                print(f"[{index + 1}/{len(dataset)}] {mode}: {values}", flush=True)
        fid_scores = fid.compute(modes)
    finally:
        fid.cleanup()
    rows = [dict(mode=mode, count=stats[mode].count,
                 **stats[mode].averages(), fid=fid_scores[mode]) for mode in modes]
    output_dir = Path(args.output_dir)
    if args._worker_mode:
        output_dir = output_dir / ".multi_gpu_metrics" / args._worker_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / CSV_NAME, rows, FIELDS)
    print(f"[Done] Metrics: {output_dir / CSV_NAME}", flush=True)


def main(args):
    if args.save_debug_vis:
        raise ValueError("This metrics-only entry does not support --save_debug_vis.")
    modes = parse_ablation_modes(args.ablation_modes)
    if args.multi_gpu and not args._worker_mode:
        run_multi_gpu(args, modes)
    else:
        evaluate(args, modes)


if __name__ == "__main__":
    main(parse_args())
