"""单次 Event-LoRA 推理评估；历史 CSV 兼容模式不执行任何额外引导。"""

import os
import time
from collections import defaultdict

import pyiqa
import torch

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from osediff_event_film import (
    OSEDiffEventFiLMTest,
    parse_unet_lora_parts,
    resolve_decoder_refinement_args,
)
from test_event_film_lora_osediff import FIDEvaluator, tensor_to_pil
from test_idea3_decoder_refinement import (
    build_parser as _build_model_parser,
    finalize_args,
    quality_metrics,
    write_csv,
)


# 只用于读取旧 baseline 报表的兼容布局，不保留交换事件或双分支运算。
LEGACY_CONFIG = {
    "experiment": "baseline",
    "guidance_scale": 0.0,
    "swap_ratio": 0.0,
    "swap_mode": "spatial",
    "pair_policy": "dissimilar",
    "feature_scales": "32,16,8",
    "execution_mode": "batched",
}


def build_parser(include_experiments=False):
    """复用旧 decoder/事件参数；普通 LoRA 默认完整激活，保持历史评估行为。"""
    if include_experiments:
        raise ValueError("The baseline entry supports only a single baseline experiment.")
    parser = _build_model_parser(include_experiments=False)
    parser.description = "Single-pass Event-LoRA baseline evaluation"
    parser.add_argument(
        "--unet_lora_parts", type=parse_unet_lora_parts,
        default=("down", "mid", "up"),
        help="普通 UNet LoRA 激活组；历史 baseline 默认 down,mid,up，可显式选 none。",
    )
    parser.add_argument(
        "--csv_format", choices=("generic", "legacy-baseline"), default="generic",
        help="legacy-baseline 仅兼容历史 CSV 名称和 baseline 列布局。",
    )
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 评估整个测试集，正整数用于小规模检查。")
    return parser


def parse_args(argv=None):
    args = finalize_args(build_parser().parse_args(argv))
    if args.max_samples < 0:
        raise ValueError("--max_samples must be non-negative.")
    args.event_ablation_mode = "correct"
    return args


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _config_columns(csv_format):
    return dict(LEGACY_CONFIG) if csv_format == "legacy-baseline" else {"experiment": "baseline"}


def _compatible_stats(stats, csv_format):
    """旧列中的差值为零；范数必须来自真实预测，不能用零冒充。"""
    if csv_format != "legacy-baseline":
        return dict(stats)
    prediction_norm = stats.get("prediction_norm", stats.get("positive_prediction_norm"))
    if prediction_norm is None:
        raise RuntimeError("Legacy baseline CSV requires the measured prediction norm.")
    return {
        "mean_abs_delta": 0.0,
        "delta_to_positive_norm": 0.0,
        "positive_prediction_norm": prediction_norm,
        "guided_prediction_norm": prediction_norm,
        "unet_ms": stats["unet_ms"],
    }


def write_baseline_reports(output_dir, csv_format, rows, summary, scenes):
    """保留三级统计及历史列排序；新入口默认使用通用文件名。"""
    config_fields = list(_config_columns(csv_format))
    # 历史名称只出现在这个显式兼容出口，不构成对已删除算法的依赖。
    prefix = "idea4_esg" if csv_format == "legacy-baseline" else "event_baseline"
    for suffix, data, extra, tail in (
        ("per_image", rows, ["scene", "frame_id"], ["result_path"]),
        ("summary", summary, ["count"], []),
        ("scene", scenes, ["scene", "count"], []),
    ):
        fixed = config_fields + extra
        numeric = sorted({key for row in data for key in row} - set(fixed + tail))
        write_csv(os.path.join(output_dir, f"{prefix}_{suffix}.csv"), data, fixed + numeric + tail)


def main(args):
    if not os.path.isdir(args.input_image):
        raise NotADirectoryError(args.input_image)
    if not os.path.isfile(args.event_lora_checkpoint):
        raise FileNotFoundError(args.event_lora_checkpoint)
    resolve_decoder_refinement_args(args, args.event_lora_checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = SDEEventFilmDataset(split="test", args=args, root_dir=args.input_image)
    sample_count = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
    model = OSEDiffEventFiLMTest(args)
    device, dtype = model.device, model.weight_dtype
    print(f"[Baseline] seed={args.seed}; unet_lora_parts={args.unet_lora_parts}; dtype={dtype}")
    metrics = {
        "psnr": pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device),
        "ssim": pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device),
        "lpips": pyiqa.create_metric("lpips", device=device).to(device),
    }
    fid = FIDEvaluator(["baseline"], device) if args.calculate_fid else None
    rows, sums = [], defaultdict(float)
    scene_sums, scene_counts = defaultdict(lambda: defaultdict(float)), defaultdict(int)
    config_values = _config_columns(args.csv_format)
    fid_scores = {}
    try:
        for index in range(sample_count):
            sample = dataset[index]
            rgb = sample["conditioning_pixel_values"].unsqueeze(0).to(device, dtype=dtype)
            target = sample["output_pixel_values"].unsqueeze(0).to(device)
            events = sample["events"].unsqueeze(0).to(device, dtype=dtype)
            refinement_events = sample["events_refinement"].unsqueeze(0).to(device, dtype=dtype)
            batch = {"prompt": [args.fixed_prompt], "neg_prompt": [args.neg_prompt]}
            # 与旧 baseline 一致：按 dataset 索引重置图像 seed，仅采样一次 VAE latent。
            torch.manual_seed(args.seed + index)
            with torch.inference_mode():
                context = model.gen.prepare_denoising_context(rgb, events, batch)
                denoised, stats = model.gen.predict_denoised_latent_from_context(
                    context, model.timesteps, measure_time=True,
                )
                _sync(device)
                decode_start = time.perf_counter()
                prediction = model.gen.decode_denoised_latent(
                    denoised, rgb,
                    events_refinement=refinement_events if args.decoder_refinement_enabled else None,
                    decoder_fusion_enabled=args.decoder_refinement_enabled,
                    refinement_fusion_mode="both" if args.decoder_refinement_enabled else None,
                )
                _sync(device)
                decoder_ms = (time.perf_counter() - decode_start) * 1000.0
                values = quality_metrics(prediction, target, metrics)
            numeric = {
                **values,
                **{key: float(value) for key, value in _compatible_stats(stats, args.csv_format).items()},
                "decoder_ms": float(decoder_ms),
            }
            scene_name = sample["scene_name"]
            scene_counts[scene_name] += 1
            for name, value in numeric.items():
                sums[name] += value
                scene_sums[scene_name][name] += value
            result_path = ""
            if args.save_images:
                result_dir = os.path.join(args.output_dir, scene_name)
                os.makedirs(result_dir, exist_ok=True)
                frame_stem = f"{int(sample['frame_index']) + 1:03d}"
                for suffix, tensor in (("ll", rgb), ("gt", target)):
                    path = os.path.join(result_dir, f"{frame_stem}_{suffix}.png")
                    if not os.path.exists(path):
                        tensor_to_pil(tensor[0]).save(path)
                result_path = os.path.join(result_dir, f"{frame_stem}_baseline_result.png")
                tensor_to_pil(prediction[0]).save(result_path)
            if fid is not None:
                fid.add_ground_truth(index, tensor_to_pil(target[0]))
                fid.add_prediction("baseline", index, tensor_to_pil(prediction[0]))
            rows.append({**config_values, "scene": scene_name, "frame_id": sample["frame_id"],
                         **numeric, "result_path": result_path})
            print(f"[{index + 1}/{sample_count}] baseline PSNR={values['psnr']:.4f} "
                  f"SSIM={values['ssim']:.4f} LPIPS={values['lpips']:.4f} UNet={stats['unet_ms']:.2f}ms")
        if fid is not None:
            fid_scores = fid.compute(["baseline"])
    finally:
        if fid is not None:
            fid.cleanup()
    averaged = {name: value / max(sample_count, 1) for name, value in sums.items()}
    if fid_scores:
        averaged["fid"] = fid_scores["baseline"]
    summary = [{**config_values, "count": sample_count, **averaged}]
    scenes = [{**config_values, "scene": scene, "count": scene_counts[scene],
               **{name: value / scene_counts[scene] for name, value in scene_sums[scene].items()}}
              for scene in sorted(scene_counts)]
    write_baseline_reports(args.output_dir, args.csv_format, rows, summary, scenes)


if __name__ == "__main__":
    main(parse_args())
