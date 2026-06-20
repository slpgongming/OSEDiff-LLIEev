import argparse
import csv
import os
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

    def compute(self, modes):
        if self.gt_count < 2:
            raise RuntimeError("FID requires at least two GT images.")

        # 推理全部结束后再创建 Inception 特征网络，避免它长期占用额外显存。
        # FID 是整组图像分布之间的距离，不能逐样本计算后再平均。
        metric = pyiqa.create_metric("fid", device=self.device)
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
    parser.add_argument("--event_base_channels", type=int, default=32)
    parser.add_argument("--mod_channels", type=int, default=64)
    parser.add_argument("--mod_num_heads", type=int, default=4)
    parser.add_argument("--mod_window_size", type=int, default=8)
    parser.add_argument("--modulation_scale", type=float, default=0.1)
    parser.add_argument(
        "--event_lora_variant",
        type=str,
        default="film",
        choices=["film", "residual_cross_attn", "residual_cross_attn_bax"],
        help="Event-LoRA variant; checkpoint value has priority when present.",
    )
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--ablation_modes", type=str, default="correct,zero")
    parser.add_argument(
        "--calculate_fid",
        action="store_true",
        help="Compute dataset-level FID between generated images and GT for each ablation mode.",
    )
    parser.add_argument("--save_debug_vis", action="store_true")
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = SDEEventFilmDataset(split="test", args=args, root_dir=args.input_image)
    model = OSEDiffEventFiLMTest(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device)
    ssim_metric = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device)
    lpips_metric = pyiqa.create_metric("lpips", device=device).to(device)
    modes = [m.strip() for m in args.ablation_modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("--ablation_modes must contain at least one mode.")
    if len(modes) != len(set(modes)):
        raise ValueError(f"--ablation_modes contains duplicate modes: {modes}")
    invalid_modes = sorted(set(modes) - {"correct", "zero"})
    if invalid_modes:
        raise ValueError(f"Unknown ablation modes: {invalid_modes}")
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
                "z_event_abs_mean": debug_stats.get("z_event_abs_mean", 0.0),
                "z_delta_abs_mean": debug_stats.get("z_delta_abs_mean", 0.0),
                "bax_event_abs_mean": debug_stats.get("bax_event_abs_mean", 0.0),
                "bax_delta_abs_mean": debug_stats.get("bax_delta_abs_mean", 0.0),
                "result_path": result_path,
            })
            print(
                f"[{idx + 1}/{len(dataset)}] {mode}/{scene_name}/{prefix} | "
                f"PSNR={metric_values['psnr']:.4f} | SSIM={metric_values['ssim']:.4f} | "
                f"LPIPS={metric_values['lpips']:.4f} | event_L1={active_l1:.6f}"
            )

    per_image_csv = os.path.join(args.output_dir, "metrics_event_film_ablation.csv")
    scene_csv = os.path.join(args.output_dir, "metrics_event_film_ablation_scene.csv")
    overall_csv = os.path.join(args.output_dir, "metrics_event_film_ablation_overall.csv")
    per_image_fields = [
        "mode", "scene", "prefix", "frame_id", "psnr", "ssim", "lpips",
        "event_active_l1", "dgamma_abs_mean", "dbeta_abs_mean",
        "z_event_abs_mean", "z_delta_abs_mean",
        "bax_event_abs_mean", "bax_delta_abs_mean", "result_path",
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
