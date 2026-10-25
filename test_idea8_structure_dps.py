"""idea8 推理：默认计算 PSNR(Y)、SSIM(Y)、LPIPS、整组 FID。"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from dataloaders.sde_event_structure_dataset import SDEEventStructureDataset, StructureEventConfig
from models.event_structure_head import load_structure_checkpoint
from models.event_structure_guidance import GuidanceConfig, observe_events, restore_sample


METRIC_NAMES = ("psnr", "ssim", "lpips", "fid")


def build_parser():
    # 沿用现有 checkpoint/refinement 配置解析；不复制第二套加载逻辑。
    from test_idea3_decoder_refinement import build_parser as base_parser
    parser = base_parser(include_experiments=False)
    parser.description = __doc__
    parser.set_defaults(calculate_fid=True, save_images=False, event_lora_variant="residual_cross_attn",
                        mod_channels=64, generator_timestep=500, decoder_refinement_enabled=False,
                        unet_lora_parts=())
    parser.add_argument("--no-calculate_fid", dest="calculate_fid", action="store_false",
                        help="显式只算前三项；默认计算全部四项。")
    parser.add_argument("--inference_only", action="store_true", help="无 GT 推理，显式关闭所有指标并保存图像。")
    parser.add_argument("--structure_checkpoint")
    parser.add_argument("--structure_manifest")
    parser.add_argument("--guidance_steps", type=int, default=0)
    parser.add_argument("--guidance_lr", type=float, default=0.01)
    parser.add_argument("--guidance_trust_radius", type=float, default=0.03)
    parser.add_argument("--guidance_lambda_low", type=float, default=2.0)
    parser.add_argument("--guidance_lambda_keep", type=float, default=1.0)
    parser.add_argument("--guidance_lambda_range", type=float, default=1.0)
    parser.add_argument("--guidance_max_low_change", type=float, default=0.02)
    parser.add_argument("--guidance_max_outside_change", type=float, default=0.02)
    parser.add_argument("--guidance_backtracking_steps", type=int, default=3)
    parser.add_argument("--guidance_min_confidence", type=float, default=0.25)
    parser.add_argument("--guidance_save_diagnostics", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser


def guidance_config(args):
    return GuidanceConfig(steps=args.guidance_steps, learning_rate=args.guidance_lr,
                          trust_radius=args.guidance_trust_radius, lambda_low=args.guidance_lambda_low,
                          lambda_keep=args.guidance_lambda_keep, lambda_range=args.guidance_lambda_range,
                          max_low_change=args.guidance_max_low_change,
                          max_outside_change=args.guidance_max_outside_change,
                          backtracking_steps=args.guidance_backtracking_steps,
                          min_confidence=args.guidance_min_confidence)


def parse_args(argv=None):
    from test_idea3_decoder_refinement import finalize_args
    parser = build_parser()
    args = finalize_args(parser.parse_args(argv))
    try:
        guidance_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.guidance_steps and not args.structure_checkpoint:
        parser.error("Positive guidance_steps requires --structure_checkpoint.")
    if args.guidance_steps and (args.decoder_refinement_enabled or args.image_size != 256):
        parser.error("Initial guidance requires refinement off and image_size=256.")
    if args.max_samples < 0:
        parser.error("max_samples must be nonnegative.")
    if args.inference_only:
        args.calculate_fid, args.save_images = False, True
    args.require_ground_truth = not args.inference_only
    args.event_ablation_mode = "correct"
    return args


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(args):
    import pyiqa
    from osediff_event_film import OSEDiffEventFiLMTest, resolve_decoder_refinement_args
    from test_event_film_lora_osediff import FIDEvaluator, tensor_to_pil
    from test_idea3_decoder_refinement import quality_metrics, write_csv

    for path in (args.input_image, args.pretrained_model_name_or_path):
        if not Path(path).is_dir():
            raise NotADirectoryError(path)
    if not Path(args.event_lora_checkpoint).is_file():
        raise FileNotFoundError(args.event_lora_checkpoint)
    resolve_decoder_refinement_args(args, args.event_lora_checkpoint)
    config = guidance_config(args)
    head = checkpoint = operator = None
    if config.steps:
        head, checkpoint, operator = load_structure_checkpoint(args.structure_checkpoint)
        event_config = StructureEventConfig(**checkpoint["event_config"])
        if checkpoint.get("image_size") != args.image_size:
            raise ValueError("Structure checkpoint preprocessing image_size mismatch.")
        if head.config.bins != event_config.bins:
            raise ValueError("Structure checkpoint channel metadata mismatch.")
        dataset = SDEEventStructureDataset("test", args, event_config, args.input_image)
    else:
        # 真正 bypass：steps=0 不需要结构数据/权重，保留旧 checkpoint 的数据行为。
        dataset = SDEEventFilmDataset("test", args, args.input_image)
    count = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    if args.calculate_fid and count < 2:
        raise ValueError("Default four-metric evaluation requires >=2 images for FID; use --no-calculate_fid for a one-image smoke test.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = OSEDiffEventFiLMTest(args)
    if config.steps and getattr(model.gen, "decoder_refinement_enabled", False):
        raise ValueError("Guidance currently requires a generator checkpoint without decoder refinement (Stage3).")
    device, dtype = model.device, model.weight_dtype
    if head is not None:
        head.to(device)
    metrics = {}
    if not args.inference_only:
        metrics = dict(psnr=pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device),
                       ssim=pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device),
                       lpips=pyiqa.create_metric("lpips", device=device).to(device))
    mode = "baseline" if not config.steps else f"structure_dps_{config.steps}"
    fid = FIDEvaluator([mode], device) if args.calculate_fid else None
    manifest = dict(arguments=vars(args), guidance=asdict(config), metrics=list(metrics) + (["fid"] if fid else []),
                    actual_event_lora_targets=list(model.gen.event_lora_config.targets),
                    actual_unet_lora_parts=list(args.unet_lora_parts),
                    structure_event_config=checkpoint["event_config"] if checkpoint else None,
                    scheduler=dict(model.gen.noise_scheduler.config), scheduler_timesteps=model.timesteps.tolist())
    scheduler = model.gen.noise_scheduler
    manifest["scheduler_class"] = type(scheduler).__name__
    manifest["scheduler_internal_timesteps"] = getattr(scheduler, "timesteps", torch.tensor([])).tolist()
    manifest["scheduler_previous_timesteps"] = [int(scheduler.previous_timestep(t)) for t in model.timesteps] if hasattr(scheduler, "previous_timestep") else None
    (output / "run_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    rows, diagnostic_rows = [], []
    totals, scene_totals, scene_counts = defaultdict(float), defaultdict(lambda: defaultdict(float)), defaultdict(int)
    try:
        for index in range(count):
            sample = dataset[index]
            rgb = sample["conditioning_pixel_values"][None].to(device, dtype)
            events = sample["events"][None].to(device, dtype)
            refinement = sample["events_refinement"][None].to(device, dtype)
            # 生成函数只收到白名单输入；GT 不进入 context、结构观测或候选验收。
            torch.manual_seed(args.seed + index)
            _sync(device)
            start = time.perf_counter()
            with torch.no_grad():
                context = model.gen.prepare_denoising_context(
                    rgb, events, dict(prompt=[args.fixed_prompt], neg_prompt=[args.neg_prompt]))
                observation = observe_events(head, sample["events_structure"][None].to(device), config.min_confidence) if head else None
            prediction, stats = restore_sample(model.gen, context, model.timesteps, rgb, config,
                                                observation, operator, refinement, args.decoder_refinement_enabled)
            _sync(device)
            stats["inference_ms"] = (time.perf_counter() - start) * 1000
            values = {}
            if metrics:
                target = sample["output_pixel_values"][None].to(device)
                with torch.no_grad():
                    values = quality_metrics(prediction, target, metrics)
                if fid:
                    fid.add_ground_truth(index, tensor_to_pil(target[0]))
                    fid.add_prediction(mode, index, tensor_to_pil(prediction[0]))
            scene = sample["scene_name"]
            result_path = ""
            if args.save_images:
                destination = output / scene
                destination.mkdir(exist_ok=True)
                stem = f"{int(sample['frame_index'])+1:03d}"
                result_path = str(destination / f"{stem}_{mode}_result.png")
                tensor_to_pil(prediction[0]).save(result_path)
                tensor_to_pil(rgb[0]).save(destination / f"{stem}_ll.png")
                if metrics:
                    tensor_to_pil(target[0]).save(destination / f"{stem}_gt.png")
            for key, value in values.items():
                totals[key] += value
                scene_totals[scene][key] += value
            scene_counts[scene] += 1
            rows.append(dict(scene=scene, frame_id=sample["frame_id"], **values, result_path=result_path))
            diagnostic_rows.append(dict(scene=scene, frame_id=sample["frame_id"], **stats,
                                        structure_meta=sample.get("structure_meta")))
            if args.guidance_save_diagnostics and observation is not None:
                diagnostic_dir = output / "diagnostics" / scene
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
                torch.save(dict(edges=observation.edges.cpu(), confidence=observation.confidence.cpu()),
                           diagnostic_dir / f"{sample['frame_id']}.pt")
            print(f"[{index+1}/{count}] {scene}/{sample['frame_id']} {values} {stats}", flush=True)
        fid_score = fid.compute([mode])[mode] if fid else None
    finally:
        if fid is not None:
            fid.cleanup()
    write_csv(output / "idea8_per_image.csv", rows, ["scene", "frame_id", *metrics, "result_path"])
    summary = dict(mode=mode, count=count, **{key: totals[key]/count for key in metrics})
    if fid_score is not None:
        summary["fid"] = fid_score
    write_csv(output / "idea8_summary.csv", [summary], list(summary))
    scene_rows = [dict(scene=scene, count=n, **{key: scene_totals[scene][key]/n for key in metrics})
                  for scene, n in sorted(scene_counts.items())]
    write_csv(output / "idea8_scene.csv", scene_rows, ["scene", "count", *metrics])
    (output / "idea8_diagnostics.json").write_text(json.dumps(diagnostic_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] {output / 'idea8_summary.csv'}: {summary}", flush=True)


if __name__ == "__main__":
    main(parse_args())
