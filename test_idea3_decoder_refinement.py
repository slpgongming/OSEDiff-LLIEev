"""idea3 Phase 1 A-F 因果实验；同 event mode 的实验严格复用同一 z0。"""

import argparse
import csv
import os
from collections import defaultdict

import pyiqa
import torch

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from osediff_event_film import (
    OSEDiffEventFiLMTest,
    resolve_decoder_refinement_args,
)
from test_event_film_lora_osediff import FIDEvaluator, tensor_to_pil


EXPERIMENTS = {
    "A": {"fusion": False, "events": "correct", "fusion_mode": "both"},
    "B": {"fusion": True, "events": "correct", "fusion_mode": "both"},
    "C": {"fusion": True, "events": "zero", "fusion_mode": "both"},
    "D": {"fusion": True, "events": "shuffled", "fusion_mode": "both"},
    "E": {"fusion": True, "events": "correct", "fusion_mode": "rgb"},
    "F": {"fusion": True, "events": "correct", "fusion_mode": "event"},
}


def parse_experiments(value):
    experiments = [item.strip().upper() for item in value.split(",") if item.strip()]
    invalid = [item for item in experiments if item not in EXPERIMENTS]
    if not experiments or invalid:
        raise ValueError(f"Invalid experiments: {invalid or experiments}")
    return experiments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--event_lora_checkpoint", required=True)
    parser.add_argument("--rgb_event_pretrained_root", default=None)
    parser.add_argument("--rgb_event_encoder_i_path", default=None)
    parser.add_argument("--rgb_event_encoder_e_path", default=None)
    parser.add_argument("--rgb_event_fusion_path", default=None)
    parser.add_argument(
        "--rgb_event_strict_load",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--event_lora_variant",
        default="cbam_lowrank_residual",
        choices=(
            "film_affine", "film", "cbam_lowrank_residual", "cbam_output_gate",
            "residual_cross_attn", "residual_cross_attn_bax",
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--event_channels", type=int, default=5)
    parser.add_argument("--event_bins", type=int, default=5)
    parser.add_argument("--event_base_channels", type=int, default=32)
    parser.add_argument("--mod_channels", type=int, default=32)
    parser.add_argument("--modulation_scale", type=float, default=0.1)
    parser.add_argument("--mod_num_heads", type=int, default=4)
    parser.add_argument("--mod_window_size", type=int, default=8)
    parser.add_argument("--event_lora_cbam_reduction", type=int, default=8)
    parser.add_argument("--event_lora_cbam_spatial_kernel", type=int, default=7)
    parser.add_argument("--event_lora_output_gate_channels", default="spatial")
    parser.add_argument("--decoder_fusion_weight", type=float, default=1.0)
    parser.add_argument("--decoder_fusion_init", default="near_zero")
    parser.add_argument(
        "--decoder_refinement_variant",
        default="auto",
        choices=("auto", "legacy_fused", "cgdrs"),
        help="Refinement path; auto inherits the checkpoint configuration.",
    )
    parser.add_argument("--cgdrs_event_levels", default="1,2,3")
    parser.add_argument("--cgdrs_gate_bias", type=float, default=-2.0)
    parser.add_argument(
        "--cgdrs_activity_mode", choices=("abs_mean", "abs_sum"), default="abs_mean"
    )
    parser.add_argument("--experiments", default="A,B,C,D,E,F")
    parser.add_argument("--generator_timestep", type=int, default=999)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fixed_prompt", default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", default="")
    parser.add_argument("--mixed_precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calculate_fid", action="store_true")
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_ignore_mismatch", action="store_true")
    args = parser.parse_args()
    args.osediff_path = args.event_lora_checkpoint
    args.decoder_refinement_enabled = True
    args.decoder_fusion_enabled = True
    args.refinement_fusion_mode = "both"
    return args


def quality_metrics(prediction, target, metrics):
    prediction = prediction.float().clamp(-1, 1) * 0.5 + 0.5
    target = target.float().clamp(-1, 1) * 0.5 + 0.5
    return {
        name: float(metric(prediction, target).item())
        for name, metric in metrics.items()
    }


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    experiments = parse_experiments(args.experiments)
    if not os.path.isdir(args.input_image):
        raise NotADirectoryError(args.input_image)
    if not os.path.isfile(args.event_lora_checkpoint):
        raise FileNotFoundError(args.event_lora_checkpoint)
    resolve_decoder_refinement_args(args, args.event_lora_checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    required_event_modes = sorted({EXPERIMENTS[item]["events"] for item in experiments})
    datasets = {}
    for event_mode in required_event_modes:
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.event_ablation_mode = event_mode
        datasets[event_mode] = SDEEventFilmDataset(
            split="test", args=dataset_args, root_dir=args.input_image
        )
    lengths = {len(dataset) for dataset in datasets.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"Ablation dataset length mismatch: {lengths}")

    model = OSEDiffEventFiLMTest(args)
    device = model.device
    dtype = model.weight_dtype
    metrics = {
        "psnr": pyiqa.create_metric(
            "psnr", test_y_channel=True, color_space="ycbcr"
        ).to(device),
        "ssim": pyiqa.create_metric(
            "ssim", test_y_channel=True, color_space="ycbcr"
        ).to(device),
        "lpips": pyiqa.create_metric("lpips", device=device).to(device),
    }
    fid = FIDEvaluator(experiments, device) if args.calculate_fid else None
    rows = []
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    scene_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    scene_counts = defaultdict(lambda: defaultdict(int))

    for index in range(next(iter(lengths))):
        samples = {mode: dataset[index] for mode, dataset in datasets.items()}
        correct = samples.get("correct", next(iter(samples.values())))
        rgb = correct["conditioning_pixel_values"].unsqueeze(0).to(device, dtype=dtype)
        target = correct["output_pixel_values"].unsqueeze(0).to(device)
        if fid is not None:
            fid.add_ground_truth(index, tensor_to_pil(target[0]))

        denoised_by_mode = {}
        for event_mode in required_event_modes:
            sample = samples[event_mode]
            events = sample["events"].unsqueeze(0).to(device, dtype=dtype)
            # 每种事件证据只采样/运行一次 diffusion latent；A/B/E/F 共用 correct z0。
            torch.manual_seed(args.seed + index)
            batch = {
                "prompt": [args.fixed_prompt],
                "neg_prompt": [args.neg_prompt],
            }
            with torch.inference_mode():
                denoised_by_mode[event_mode] = model.gen.predict_denoised_latent(
                    rgb, events, model.timesteps, batch
                )[0]

        for experiment in experiments:
            config = EXPERIMENTS[experiment]
            sample = samples[config["events"]]
            refinement_events = sample["events_refinement"].unsqueeze(0).to(
                device, dtype=dtype
            )
            with torch.inference_mode():
                prediction = model.gen.decode_denoised_latent(
                    denoised_by_mode[config["events"]],
                    rgb,
                    events_refinement=refinement_events,
                    decoder_fusion_enabled=config["fusion"],
                    refinement_fusion_mode=config["fusion_mode"],
                )
                values = quality_metrics(prediction, target, metrics)
            counts[experiment] += 1
            for name, value in values.items():
                sums[experiment][name] += value
                scene_sums[experiment][correct["scene_name"]][name] += value
            scene_counts[experiment][correct["scene_name"]] += 1
            result_path = ""
            if args.save_images:
                # 同一场景共享 low/gt；不同实验只在推理结果文件名中用实验字母区分。
                # frame_index 已按场景内时间戳顺序生成，因此从 1 开始使用三位序号。
                result_dir = os.path.join(
                    args.output_dir, correct["scene_name"]
                )
                os.makedirs(result_dir, exist_ok=True)
                frame_number = int(correct["frame_index"]) + 1
                frame_stem = f"{frame_number:03d}"
                ll_path = os.path.join(result_dir, f"{frame_stem}_ll.png")
                gt_path = os.path.join(result_dir, f"{frame_stem}_gt.png")
                result_path = os.path.join(
                    result_dir, f"{frame_stem}_{experiment}_result.png"
                )
                # low/gt 对同一帧只需保存一次，避免 A/B/E 等实验重复写盘。
                if not os.path.exists(ll_path):
                    tensor_to_pil(rgb[0]).save(ll_path)
                if not os.path.exists(gt_path):
                    tensor_to_pil(target[0]).save(gt_path)
                tensor_to_pil(prediction[0]).save(result_path)
            if fid is not None:
                fid.add_prediction(experiment, index, tensor_to_pil(prediction[0]))
            rows.append({
                "experiment": experiment,
                "scene": correct["scene_name"],
                "frame_id": correct["frame_id"],
                **values,
                "result_path": result_path,
            })
            print(
                f"[{index + 1}/{next(iter(lengths))}] {experiment} "
                f"PSNR={values['psnr']:.4f} SSIM={values['ssim']:.4f} "
                f"LPIPS={values['lpips']:.4f}"
            )

    fid_scores = {}
    if fid is not None:
        try:
            fid_scores = fid.compute(experiments)
        finally:
            fid.cleanup()
    summary = []
    scene_summary = []
    for experiment in experiments:
        row = {
            "experiment": experiment,
            "count": counts[experiment],
            **{
                name: sums[experiment][name] / max(counts[experiment], 1)
                for name in ("psnr", "ssim", "lpips")
            },
        }
        if fid_scores:
            row["fid"] = fid_scores[experiment]
        summary.append(row)
        for scene_name in sorted(scene_counts[experiment]):
            scene_count = scene_counts[experiment][scene_name]
            scene_summary.append({
                "experiment": experiment,
                "scene": scene_name,
                "count": scene_count,
                **{
                    name: scene_sums[experiment][scene_name][name] / scene_count
                    for name in ("psnr", "ssim", "lpips")
                },
            })

    write_csv(
        os.path.join(args.output_dir, "idea3_phase1_per_image.csv"),
        rows,
        ("experiment", "scene", "frame_id", "psnr", "ssim", "lpips", "result_path"),
    )
    summary_fields = ["experiment", "count", "psnr", "ssim", "lpips"]
    if fid_scores:
        summary_fields.append("fid")
    write_csv(
        os.path.join(args.output_dir, "idea3_phase1_scene.csv"),
        scene_summary,
        ("experiment", "scene", "count", "psnr", "ssim", "lpips"),
    )
    write_csv(
        os.path.join(args.output_dir, "idea3_phase1_summary.csv"),
        summary,
        summary_fields,
    )


if __name__ == "__main__":
    main(parse_args())
