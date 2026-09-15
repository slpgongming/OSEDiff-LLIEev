import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

import pyiqa
import torch
from PIL import Image
from torchvision import transforms

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from osediff_event_film import OSEDiffEventFiLMTest


METRIC_NAMES = ("psnr", "ssim", "lpips")
FID_INCEPTION_WEIGHTS_NAME = "pt_inception-2015-12-05-6726825d.pth"


@dataclass
class MetricAccumulator:
    count: int = 0
    sums: dict = field(default_factory=lambda: {name: 0.0 for name in METRIC_NAMES})

    def update(self, metric_values):
        # Each sample contributes one scalar per metric.  We keep raw sums so
        # scene-level and overall averages are computed from the same values
        # that are written to the per-image CSV.
        self.count += 1
        for name in METRIC_NAMES:
            self.sums[name] += float(metric_values[name])

    def averages(self):
        if self.count == 0:
            return {name: 0.0 for name in METRIC_NAMES}
        return {name: self.sums[name] / self.count for name in METRIC_NAMES}


class FIDEvaluator:
    """在临时目录中收集整组图像，并按消融模式计算总体 FID。"""

    def __init__(self, modes, device):
        self._temp_dir = tempfile.TemporaryDirectory(prefix="osediff_fid_")
        self.device = device
        self.gt_dir = os.path.join(self._temp_dir.name, "ground_truth")
        self.pred_dirs = {
            mode: os.path.join(self._temp_dir.name, "generated", mode)
            for mode in modes
        }
        os.makedirs(self.gt_dir, exist_ok=True)
        for pred_dir in self.pred_dirs.values():
            os.makedirs(pred_dir, exist_ok=True)

        self.gt_count = 0
        self.pred_counts = defaultdict(int)

    @staticmethod
    def _file_name(sample_index):
        # 使用全局样本序号保证跨 scene 文件名唯一，并让 GT/预测集合顺序稳定。
        return f"{sample_index:08d}.png"

    def add_ground_truth(self, sample_index, image):
        image.save(os.path.join(self.gt_dir, self._file_name(sample_index)))
        self.gt_count += 1

    def add_prediction(self, mode, sample_index, image):
        image.save(os.path.join(self.pred_dirs[mode], self._file_name(sample_index)))
        self.pred_counts[mode] += 1

    @staticmethod
    def _fid_weights_path():
        # PyIQA 将 FID Inception 权重放在 torch hub 的 pyiqa 子目录中；
        # 使用 torch.hub.get_dir() 可兼容默认 ~/.cache 和自定义 TORCH_HOME。
        return os.path.join(
            torch.hub.get_dir(),
            "pyiqa",
            FID_INCEPTION_WEIGHTS_NAME,
        )

    def _create_metric_with_cache_repair(self):
        try:
            return pyiqa.create_metric("fid", device=self.device)
        except RuntimeError as exc:
            error_text = str(exc).lower()
            corruption_markers = (
                "unexpected eof",
                "pickle data was truncated",
                "failed finding central directory",
                "pytorchstreamreader failed",
                "invalid header",
            )
            if not any(marker in error_text for marker in corruption_markers):
                raise

            weights_path = self._fid_weights_path()
            if not os.path.isfile(weights_path):
                raise RuntimeError(
                    "FID Inception weights appear corrupted, but the cached file "
                    f"was not found at the expected path: {weights_path}"
                ) from exc

            # 下载中断会留下不完整 .pth；删除后让 PyIQA 自动重新下载一次。
            file_size_mb = os.path.getsize(weights_path) / (1024 ** 2)
            print(
                "[FID][Warning] Corrupted Inception cache detected: "
                f"{weights_path} ({file_size_mb:.1f} MB). Removing and retrying once."
            )
            os.remove(weights_path)
            return pyiqa.create_metric("fid", device=self.device)

    def compute(self, modes):
        if self.gt_count < 2:
            raise RuntimeError("FID requires at least two GT images.")

        # 推理全部结束后再创建 Inception 特征网络，避免它长期占用额外显存。
        # FID 是整组图像分布之间的距离，不能逐样本计算后再平均。
        metric = self._create_metric_with_cache_repair()
        scores = {}
        for mode in modes:
            if self.pred_counts[mode] != self.gt_count:
                raise RuntimeError(
                    f"FID image count mismatch for mode={mode}: "
                    f"generated={self.pred_counts[mode]}, gt={self.gt_count}"
                )
            with torch.inference_mode():
                scores[mode] = float(
                    metric(self.pred_dirs[mode], self.gt_dir).item()
                )
        return scores

    def cleanup(self):
        self._temp_dir.cleanup()


def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().float().clamp(-1, 1) * 0.5 + 0.5
    return transforms.ToPILImage()(tensor)


def event_active_l1(pred, gt, events):
    mask = events.abs().sum(dim=1, keepdim=True)
    mask = mask / mask.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (mask * (pred - gt).abs()).mean().item()


def save_event_activity(events, path):
    activity = events.abs().sum(dim=0)
    activity = activity / activity.max().clamp_min(1e-6)
    Image.fromarray((activity.cpu().numpy() * 255).astype("uint8")).save(path)


def build_ablation_event(events, mode):
    # All ablation inputs keep the same [B, event_bins, H, W] shape expected by
    # the event encoder.  Only the event evidence content changes.
    if mode == "correct":
        return events
    if mode == "zero":
        return torch.zeros_like(events)
    raise ValueError(f"Unknown ablation mode: {mode}")


def compute_quality_metrics(pred, gt, psnr_metric, ssim_metric, lpips_metric):
    # pyiqa full-reference metrics expect image tensors in [0, 1].  The model
    # path uses the OSEDiff convention [-1, 1], so clamp before remapping to
    # avoid out-of-range values dominating PSNR/SSIM/LPIPS.
    pred_01 = pred.float().clamp(-1, 1) * 0.5 + 0.5
    gt_01 = gt.float().clamp(-1, 1) * 0.5 + 0.5
    return {
        "psnr": psnr_metric(pred_01, gt_01).item(),
        "ssim": ssim_metric(pred_01, gt_01).item(),
        "lpips": lpips_metric(pred_01, gt_01).item(),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def parse_ablation_modes(value):
    modes = [mode.strip() for mode in value.split(",") if mode.strip()]
    if not modes:
        raise ValueError("--ablation_modes must contain at least one mode.")
    if len(modes) != len(set(modes)):
        raise ValueError(f"--ablation_modes contains duplicate modes: {modes}")
    invalid_modes = sorted(set(modes) - {"correct", "zero"})
    if invalid_modes:
        raise ValueError(f"Unknown ablation modes: {invalid_modes}")
    return modes


def _worker_command(mode):
    """构造单消融子进程命令，移除主进程专用参数并覆盖 mode。"""
    source_args = sys.argv[1:]
    worker_args = []
    skip_next = False
    options_with_value = {"--ablation_modes", "--_worker_mode"}
    for arg in source_args:
        if skip_next:
            skip_next = False
            continue
        if arg in options_with_value:
            skip_next = True
            continue
        if arg == "--multi_gpu":
            continue
        if arg.startswith("--ablation_modes=") or arg.startswith("--_worker_mode="):
            continue
        worker_args.append(arg)
    return [
        sys.executable,
        os.path.abspath(__file__),
        *worker_args,
        "--ablation_modes",
        mode,
        "--_worker_mode",
        mode,
    ]


def _merge_multi_gpu_metrics(output_dir, modes):
    """将两个 GPU 子进程的 mode 独立 CSV 合并回原有标准文件名。"""
    worker_root = os.path.join(output_dir, ".multi_gpu_metrics")
    csv_names = (
        "metrics_event_film_ablation.csv",
        "metrics_event_film_ablation_scene.csv",
        "metrics_event_film_ablation_overall.csv",
    )
    for csv_name in csv_names:
        merged_rows = []
        merged_fields = None
        for mode in modes:
            worker_csv = os.path.join(worker_root, mode, csv_name)
            if not os.path.isfile(worker_csv):
                raise RuntimeError(f"Missing multi-GPU worker metrics: {worker_csv}")
            rows, fieldnames = read_csv(worker_csv)
            if merged_fields is None:
                merged_fields = fieldnames
            elif fieldnames != merged_fields:
                raise RuntimeError(
                    f"CSV fields mismatch while merging {csv_name}: "
                    f"{merged_fields} vs {fieldnames}"
                )
            merged_rows.extend(rows)
        write_csv(os.path.join(output_dir, csv_name), merged_rows, merged_fields)
    shutil.rmtree(worker_root)


def run_multi_gpu(args):
    """GPU0/GPU1 分别运行 correct/zero，结束后由主进程合并指标。"""
    modes = parse_ablation_modes(args.ablation_modes)
    if len(modes) < 2:
        raise ValueError("--multi_gpu requires both correct and zero modes.")
    if not torch.cuda.is_available():
        raise RuntimeError("--multi_gpu requires CUDA, but CUDA is not available.")
    if torch.cuda.device_count() < len(modes):
        raise RuntimeError(
            f"--multi_gpu needs {len(modes)} GPUs, but PyTorch sees "
            f"{torch.cuda.device_count()}."
        )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if not gpu_ids:
        gpu_ids = [str(index) for index in range(torch.cuda.device_count())]
    if len(gpu_ids) < len(modes):
        raise RuntimeError(
            f"--multi_gpu needs {len(modes)} visible GPUs, but found {gpu_ids or 'none'}."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    worker_root = os.path.join(args.output_dir, ".multi_gpu_metrics")
    if os.path.isdir(worker_root):
        shutil.rmtree(worker_root)

    if args.calculate_fid:
        # 在启动子进程前只下载/修复一次 FID 权重，避免双进程并发写同一缓存。
        warmup = FIDEvaluator([], torch.device("cuda"))
        try:
            metric = warmup._create_metric_with_cache_repair()
            del metric
        finally:
            warmup.cleanup()
        torch.cuda.empty_cache()

    processes = []
    for mode, gpu_id in zip(modes, gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONUNBUFFERED"] = "1"
        print(f"[MultiGPU] mode={mode} -> GPU {gpu_id}")
        processes.append(
            (mode, subprocess.Popen(_worker_command(mode), env=env))
        )

    failed = []
    for mode, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((mode, return_code))
    if failed:
        raise RuntimeError(f"Multi-GPU inference worker failed: {failed}")

    _merge_multi_gpu_metrics(args.output_dir, modes)
    print(f"[MultiGPU] Combined metrics saved under: {args.output_dir}")


def build_summary_rows(stats_by_mode, modes, include_scene):
    rows = []
    for mode in modes:
        if mode not in stats_by_mode:
            continue
        if include_scene:
            items = sorted(stats_by_mode[mode].items())
        else:
            items = [(None, stats_by_mode[mode])]

        for scene_name, accumulator in items:
            averages = accumulator.averages()
            row = {"mode": mode, "count": accumulator.count}
            if include_scene:
                row["scene"] = scene_name
            row.update({name: averages[name] for name in METRIC_NAMES})
            rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[2, 3], default=2,
        help="Stage label for logging/output semantics; checkpoint content is selected by --osediff_path.")
    parser.add_argument("--input_image", "-i", type=str, required=True,
        help="SDE test root with scene/low and scene/normal folders.")
    parser.add_argument("--output_dir", "-o", type=str, default="output_event_film/test")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--osediff_path", type=str, required=True)
    parser.add_argument("--generator_timestep", type=int, default=999)
    parser.add_argument("--fixed_prompt", type=str, default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", type=str, default="")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--event_bins", type=int, default=5)
    parser.add_argument("--event_channels", type=int, default=5)
    parser.add_argument(
        "--clora_event_active_bins", type=int, default=5,
        help="cLoRA 输入使用的前置有效 event bin 数；1 将全部事件聚集到 bin 0，通道数仍为 5。",
    )
    parser.add_argument(
        "--rgb_event_active_bins", type=int, default=5,
        help="为统一数据集接口保留；该 Stage2/3 推理入口不使用 decoder refinement。",
    )
    parser.add_argument("--event_base_channels", type=int, default=32)
    parser.add_argument("--mod_channels", type=int, default=32)
    parser.add_argument("--modulation_scale", type=float, default=0.1)
    parser.add_argument(
        "--event_lora_variant",
        type=str,
        default="film_affine",
        choices=[
            "film_affine",
            "film",
            "cbam_lowrank_residual",
            "cbam_output_gate",
            "residual_cross_attn",
            "residual_cross_attn_bax",
        ],
    )
    parser.add_argument("--event_lora_cbam_reduction", type=int, default=8)
    parser.add_argument(
        "--event_lora_cbam_spatial_kernel", type=int, default=7, choices=[3, 7]
    )
    parser.add_argument(
        "--event_lora_output_gate_channels",
        type=str,
        default="spatial",
        choices=["spatial", "channel_spatial"],
    )
    parser.add_argument("--mod_num_heads", type=int, default=4)
    parser.add_argument("--mod_window_size", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--ablation_modes", type=str, default="correct,zero")
    parser.add_argument(
        "--calculate_fid",
        action="store_true",
        help="Compute dataset-level FID between generated images and GT for each ablation mode.",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Run correct and zero concurrently on two visible GPUs, then merge metrics.",
    )
    parser.add_argument("--_worker_mode", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--save_debug_vis", action="store_true")
    return parser.parse_args()


def main(args):
    if args.multi_gpu and not args._worker_mode:
        run_multi_gpu(args)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = SDEEventFilmDataset(split="test", args=args, root_dir=args.input_image)
    model = OSEDiffEventFiLMTest(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device)
    ssim_metric = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device)
    lpips_metric = pyiqa.create_metric("lpips", device=device).to(device)
    modes = parse_ablation_modes(args.ablation_modes)
    fid_evaluator = FIDEvaluator(modes, device) if args.calculate_fid else None

    scene_counts = defaultdict(int)
    per_image_rows = []
    scene_stats = defaultdict(lambda: defaultdict(MetricAccumulator))
    overall_stats = defaultdict(MetricAccumulator)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        scene_name = sample["scene_name"]
        scene_counts[scene_name] += 1
        prefix = f"{scene_counts[scene_name]:04d}"

        lq = sample["conditioning_pixel_values"].unsqueeze(0).to(device)
        gt = sample["output_pixel_values"].unsqueeze(0).to(device)
        events = sample["events"].unsqueeze(0).to(device)
        low_pil = tensor_to_pil(lq[0])
        gt_pil = tensor_to_pil(gt[0])
        if fid_evaluator is not None:
            fid_evaluator.add_ground_truth(idx, gt_pil)

        for mode in modes:
            event_input = build_ablation_event(events, mode)

            with torch.inference_mode():
                pred = model(lq, event_input, prompt=args.fixed_prompt)

            out_dir = os.path.join(args.output_dir, mode, scene_name)
            os.makedirs(out_dir, exist_ok=True)
            low_path = os.path.join(out_dir, f"{prefix}_lowlight.png")
            result_path = os.path.join(out_dir, f"{prefix}_result.png")
            gt_path = os.path.join(out_dir, f"{prefix}_gt.png")
            pred_pil = tensor_to_pil(pred[0])
            low_pil.save(low_path)
            pred_pil.save(result_path)
            gt_pil.save(gt_path)
            if fid_evaluator is not None:
                fid_evaluator.add_prediction(mode, idx, pred_pil)

            with torch.inference_mode():
                metric_values = compute_quality_metrics(pred, gt, psnr_metric, ssim_metric, lpips_metric)
            active_l1 = event_active_l1(pred.float(), gt.float(), event_input.float())
            debug_stats = model.debug_stats()
            scene_stats[mode][scene_name].update(metric_values)
            overall_stats[mode].update(metric_values)

            if args.save_debug_vis:
                save_event_activity(event_input[0].float(), os.path.join(out_dir, f"{prefix}_event_activity.png"))

            per_image_rows.append({
                "mode": mode,
                "scene": scene_name,
                "prefix": prefix,
                "frame_id": sample["frame_id"],
                "psnr": metric_values["psnr"],
                "ssim": metric_values["ssim"],
                "lpips": metric_values["lpips"],
                "event_active_l1": active_l1,
                "dgamma_abs_mean": debug_stats.get("dgamma_abs_mean", 0.0),
                "dbeta_abs_mean": debug_stats.get("dbeta_abs_mean", 0.0),
                "delta_z_abs_mean": debug_stats.get("delta_z_abs_mean", 0.0),
                "gate_mean": debug_stats.get("gate_mean", 0.0),
                "result_path": result_path,
            })
            print(
                f"[{idx + 1}/{len(dataset)}] {mode}/{scene_name}/{prefix} | "
                f"PSNR={metric_values['psnr']:.4f} | SSIM={metric_values['ssim']:.4f} | "
                f"LPIPS={metric_values['lpips']:.4f} | event_L1={active_l1:.6f}"
            )

    metrics_dir = args.output_dir
    if args._worker_mode:
        metrics_dir = os.path.join(
            args.output_dir, ".multi_gpu_metrics", args._worker_mode
        )
        os.makedirs(metrics_dir, exist_ok=True)
    per_image_csv = os.path.join(metrics_dir, "metrics_event_film_ablation.csv")
    scene_csv = os.path.join(metrics_dir, "metrics_event_film_ablation_scene.csv")
    overall_csv = os.path.join(metrics_dir, "metrics_event_film_ablation_overall.csv")
    per_image_fields = [
        "mode", "scene", "prefix", "frame_id", "psnr", "ssim", "lpips",
        "event_active_l1", "dgamma_abs_mean", "dbeta_abs_mean",
        "delta_z_abs_mean", "gate_mean", "result_path",
    ]
    scene_fields = ["mode", "scene", "count", *METRIC_NAMES]
    overall_fields = ["mode", "count", *METRIC_NAMES]
    scene_rows = build_summary_rows(scene_stats, modes, include_scene=True)
    overall_rows = build_summary_rows(overall_stats, modes, include_scene=False)

    # 先落盘已有指标；即使随后 FID 权重下载或特征提取失败，也保留本次推理结果。
    write_csv(per_image_csv, per_image_rows, per_image_fields)
    write_csv(scene_csv, scene_rows, scene_fields)
    write_csv(overall_csv, overall_rows, overall_fields)

    if fid_evaluator is not None:
        print("\n===== Computing Dataset-level FID =====")
        try:
            fid_scores = fid_evaluator.compute(modes)
        finally:
            fid_evaluator.cleanup()
        for row in overall_rows:
            row["fid"] = fid_scores[row["mode"]]
        write_csv(overall_csv, overall_rows, [*overall_fields, "fid"])

    print("\n===== Overall Average Metrics by Ablation Mode =====")
    for row in overall_rows:
        fid_text = f" | FID={row['fid']:.4f}" if args.calculate_fid else ""
        print(
            f"{row['mode']} (n={row['count']}): "
            f"PSNR={row['psnr']:.4f} | SSIM={row['ssim']:.4f} | "
            f"LPIPS={row['lpips']:.4f}{fid_text}"
        )

    print(f"[Done] Per-image metrics: {per_image_csv}")
    print(f"[Done] Per-scene metrics: {scene_csv}")
    print(f"[Done] Overall metrics: {overall_csv}")


if __name__ == "__main__":
    main(parse_args())
