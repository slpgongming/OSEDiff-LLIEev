"""idea4 Event Self-Swap Guidance (ESG) training-free 推理与参数 sweep。"""

import argparse
import os
import time
from collections import defaultdict

import pyiqa
import torch

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from models.event_self_swap_guidance import EventSelfSwapConfig
from osediff_event_film import OSEDiffEventFiLMTest, resolve_decoder_refinement_args
from test_event_film_lora_osediff import FIDEvaluator, tensor_to_pil
from test_idea3_decoder_refinement import (
    build_parser,
    finalize_args,
    quality_metrics,
    write_csv,
)


CONFIG_FIELDS = (
    "experiment",
    "guidance_scale",
    "swap_ratio",
    "swap_mode",
    "pair_policy",
    "feature_scales",
    "execution_mode",
)
METRIC_FIELDS = ("psnr", "ssim", "lpips")


def _parse_csv(value, cast, name):
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate values: {values}")
    return values


def _float_tag(value):
    return format(float(value), "g").replace("-", "m").replace(".", "p")


def build_esg_experiments(args):
    """构造去重后的 baseline + ESG 网格；不同 guidance 复用相同 swap seed。"""
    guidance_scales = _parse_csv(
        args.esg_guidance_scales, float, "--esg_guidance_scales"
    )
    swap_ratios = _parse_csv(args.esg_swap_ratios, float, "--esg_swap_ratios")
    swap_modes = _parse_csv(args.esg_swap_modes, str, "--esg_swap_modes")
    pair_policies = _parse_csv(
        args.esg_pair_policies, str, "--esg_pair_policies"
    )
    feature_scales = tuple(
        _parse_csv(args.esg_feature_scales, str, "--esg_feature_scales")
    )
    if any(scale < 0 for scale in guidance_scales):
        raise ValueError("ESG guidance scales must be non-negative.")

    experiments = []
    if 0.0 in guidance_scales:
        experiments.append({
            "experiment": "baseline",
            "config": EventSelfSwapConfig(
                guidance_scale=0.0,
                swap_ratio=0.0,
                feature_scales=feature_scales,
                execution_mode=args.esg_execution_mode,
            ),
        })
    for ratio_index, ratio in enumerate(swap_ratios):
        for mode_index, mode in enumerate(swap_modes):
            for policy_index, policy in enumerate(pair_policies):
                # seed 不含 guidance scale，保证强度 sweep 使用完全相同的交换索引。
                swap_seed = (
                    args.esg_seed
                    + ratio_index * 1000003
                    + mode_index * 10007
                    + policy_index * 101
                )
                for guidance_scale in guidance_scales:
                    if guidance_scale == 0:
                        continue
                    label = (
                        f"esg_w{_float_tag(guidance_scale)}_r{_float_tag(ratio)}_"
                        f"{mode}_{policy}"
                    )
                    experiments.append({
                        "experiment": label,
                        "config": EventSelfSwapConfig(
                            guidance_scale=guidance_scale,
                            swap_ratio=ratio,
                            swap_mode=mode,
                            pair_policy=policy,
                            feature_scales=feature_scales,
                            seed=swap_seed,
                            execution_mode=args.esg_execution_mode,
                        ),
                    })
    if not experiments:
        raise ValueError("The ESG sweep produced no experiments.")
    return experiments


def parse_args():
    parser = build_parser(include_experiments=False)
    parser.description = "idea4 Event Self-Swap Guidance inference sweep"
    parser.add_argument(
        "--esg_guidance_scales", default="0,0.1,0.25,0.5,0.75"
    )
    parser.add_argument("--esg_swap_ratios", default="0.05,0.10")
    parser.add_argument("--esg_swap_modes", default="spatial")
    parser.add_argument("--esg_pair_policies", default="dissimilar")
    parser.add_argument("--esg_feature_scales", default="32,16,8")
    parser.add_argument(
        "--esg_execution_mode",
        choices=("batched", "sequential"),
        default="batched",
    )
    parser.add_argument("--esg_seed", type=int, default=42)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 表示完整测试集；正整数用于服务器 smoke test。",
    )
    args = finalize_args(parser.parse_args())
    args.event_ablation_mode = "correct"
    return args


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _config_columns(experiment):
    config = experiment["config"]
    return {
        "experiment": experiment["experiment"],
        "guidance_scale": config.guidance_scale,
        "swap_ratio": config.swap_ratio,
        "swap_mode": config.swap_mode,
        "pair_policy": config.pair_policy,
        "feature_scales": ",".join(config.feature_scales),
        "execution_mode": config.execution_mode,
    }


def main(args):
    if not os.path.isdir(args.input_image):
        raise NotADirectoryError(args.input_image)
    if not os.path.isfile(args.event_lora_checkpoint):
        raise FileNotFoundError(args.event_lora_checkpoint)
    resolve_decoder_refinement_args(args, args.event_lora_checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    experiments = build_esg_experiments(args)
    experiment_names = [item["experiment"] for item in experiments]
    dataset = SDEEventFilmDataset(
        split="test", args=args, root_dir=args.input_image
    )
    sample_count = len(dataset)
    if args.max_samples > 0:
        sample_count = min(sample_count, args.max_samples)

    model = OSEDiffEventFiLMTest(args)
    device, dtype = model.device, model.weight_dtype
    metrics = {
        "psnr": pyiqa.create_metric(
            "psnr", test_y_channel=True, color_space="ycbcr"
        ).to(device),
        "ssim": pyiqa.create_metric(
            "ssim", test_y_channel=True, color_space="ycbcr"
        ).to(device),
        "lpips": pyiqa.create_metric("lpips", device=device).to(device),
    }
    fid = FIDEvaluator(experiment_names, device) if args.calculate_fid else None
    rows = []
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    scene_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    scene_counts = defaultdict(lambda: defaultdict(int))

    for index in range(sample_count):
        sample = dataset[index]
        rgb = sample["conditioning_pixel_values"].unsqueeze(0).to(
            device, dtype=dtype
        )
        target = sample["output_pixel_values"].unsqueeze(0).to(device)
        events = sample["events"].unsqueeze(0).to(device, dtype=dtype)
        refinement_events = sample["events_refinement"].unsqueeze(0).to(
            device, dtype=dtype
        )
        batch = {
            "prompt": [args.fixed_prompt],
            "neg_prompt": [args.neg_prompt],
        }
        # 固定一次 VAE posterior sample，并让全部 ESG 参数共享 RGB/event context。
        torch.manual_seed(args.seed + index)
        with torch.inference_mode():
            context = model.gen.prepare_denoising_context(rgb, events, batch)
        if fid is not None:
            fid.add_ground_truth(index, tensor_to_pil(target[0]))

        for experiment in experiments:
            config = experiment["config"]
            with torch.inference_mode():
                denoised, esg_stats = model.gen.predict_denoised_latent_from_context(
                    context,
                    model.timesteps,
                    esg_config=config,
                    measure_time=True,
                )
                _sync(device)
                decode_start = time.perf_counter()
                prediction = model.gen.decode_denoised_latent(
                    denoised,
                    rgb,
                    events_refinement=(
                        refinement_events if args.decoder_refinement_enabled else None
                    ),
                    decoder_fusion_enabled=args.decoder_refinement_enabled,
                    refinement_fusion_mode=(
                        "both" if args.decoder_refinement_enabled else None
                    ),
                )
                _sync(device)
                decoder_ms = (time.perf_counter() - decode_start) * 1000.0
                values = quality_metrics(prediction, target, metrics)

            numeric = {
                **values,
                **{key: float(value) for key, value in esg_stats.items()},
                "decoder_ms": float(decoder_ms),
            }
            config_values = _config_columns(experiment)
            counts[experiment["experiment"]] += 1
            scene_name = sample["scene_name"]
            for name, value in numeric.items():
                sums[experiment["experiment"]][name] += value
                scene_sums[experiment["experiment"]][scene_name][name] += value
            scene_counts[experiment["experiment"]][scene_name] += 1

            result_path = ""
            if args.save_images:
                result_dir = os.path.join(args.output_dir, scene_name)
                os.makedirs(result_dir, exist_ok=True)
                frame_number = int(sample["frame_index"]) + 1
                frame_stem = f"{frame_number:03d}"
                ll_path = os.path.join(result_dir, f"{frame_stem}_ll.png")
                gt_path = os.path.join(result_dir, f"{frame_stem}_gt.png")
                result_path = os.path.join(
                    result_dir,
                    f"{frame_stem}_{experiment['experiment']}_result.png",
                )
                if not os.path.exists(ll_path):
                    tensor_to_pil(rgb[0]).save(ll_path)
                if not os.path.exists(gt_path):
                    tensor_to_pil(target[0]).save(gt_path)
                tensor_to_pil(prediction[0]).save(result_path)
            if fid is not None:
                fid.add_prediction(
                    experiment["experiment"], index, tensor_to_pil(prediction[0])
                )
            rows.append({
                **config_values,
                "scene": scene_name,
                "frame_id": sample["frame_id"],
                **numeric,
                "result_path": result_path,
            })
            print(
                f"[{index + 1}/{sample_count}] {experiment['experiment']} "
                f"PSNR={values['psnr']:.4f} SSIM={values['ssim']:.4f} "
                f"LPIPS={values['lpips']:.4f} UNet={esg_stats['unet_ms']:.2f}ms"
            )

    fid_scores = {}
    if fid is not None:
        try:
            fid_scores = fid.compute(experiment_names)
        finally:
            fid.cleanup()

    summary = []
    scene_summary = []
    experiment_by_name = {item["experiment"]: item for item in experiments}
    for experiment_name in experiment_names:
        count = counts[experiment_name]
        averaged = {
            name: value / max(count, 1)
            for name, value in sums[experiment_name].items()
        }
        if fid_scores:
            averaged["fid"] = fid_scores[experiment_name]
        summary.append({
            **_config_columns(experiment_by_name[experiment_name]),
            "count": count,
            **averaged,
        })
        for scene_name in sorted(scene_counts[experiment_name]):
            scene_count = scene_counts[experiment_name][scene_name]
            scene_summary.append({
                **_config_columns(experiment_by_name[experiment_name]),
                "scene": scene_name,
                "count": scene_count,
                **{
                    name: value / scene_count
                    for name, value in scene_sums[experiment_name][scene_name].items()
                },
            })

    per_image_fields = list(CONFIG_FIELDS) + ["scene", "frame_id"]
    per_image_numeric = sorted(
        {key for row in rows for key in row} - set(per_image_fields) - {"result_path"}
    )
    write_csv(
        os.path.join(args.output_dir, "idea4_esg_per_image.csv"),
        rows,
        per_image_fields + per_image_numeric + ["result_path"],
    )
    summary_numeric = sorted(
        {key for row in summary for key in row} - set(CONFIG_FIELDS) - {"count"}
    )
    write_csv(
        os.path.join(args.output_dir, "idea4_esg_summary.csv"),
        summary,
        list(CONFIG_FIELDS) + ["count"] + summary_numeric,
    )
    scene_numeric = sorted(
        {key for row in scene_summary for key in row}
        - set(CONFIG_FIELDS)
        - {"scene", "count"}
    )
    write_csv(
        os.path.join(args.output_dir, "idea4_esg_scene.csv"),
        scene_summary,
        list(CONFIG_FIELDS) + ["scene", "count"] + scene_numeric,
    )


if __name__ == "__main__":
    main(parse_args())
