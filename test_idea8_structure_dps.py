"""idea8 推理：默认计算 PSNR(Y)、SSIM(Y)、LPIPS、整组 FID。"""

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from dataloaders.sde_event_structure_dataset import SDEEventStructureDataset, StructureEventConfig
from models.event_structure_head import load_structure_checkpoint
from models.event_structure_guidance import GuidanceConfig, observe_events, restore_sample, freeze_observation_masks
from models.image_structure import image_structure


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
    parser.add_argument("--guidance_target_source", choices=("event", "gt_oracle"), default="event",
                        help="gt_oracle 仅用于离线目标替换诊断，固定原事件置信度及 M0/K0。")
    parser.add_argument("--save_float_predictions", action="store_true",
                        help="保存未经 PNG 量化的输出与 GT，供结构/纹理配对分析。")
    parser.add_argument("--guidance_steps", type=int, default=0)
    parser.add_argument("--guidance_lr", type=float, default=0.01)
    parser.add_argument("--guidance_trust_radius", type=float, default=0.03)
    parser.add_argument("--guidance_lambda_low", type=float, default=1.0)
    parser.add_argument("--guidance_lambda_keep", type=float, default=0.5)
    parser.add_argument("--guidance_lambda_range", type=float, default=1.0)
    parser.add_argument("--guidance_max_low_change", type=float, default=0.02)
    parser.add_argument("--guidance_max_outside_change", type=float, default=0.02)
    parser.add_argument("--guidance_backtracking_steps", type=int, default=3)
    parser.add_argument("--guidance_min_confidence", type=float, default=0.25)
    parser.add_argument("--guidance_protection_mode", choices=("quadratic", "l1"), default="quadratic")
    parser.add_argument("--guidance_trace_candidates", action=argparse.BooleanOptionalAction, default=True,
                        help="保存每个候选的损失、步长和拒绝条件；默认开启。")
    parser.add_argument("--guidance_save_diagnostics", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_manifest", help="验证子集 CSV，列为 scene,frame_id；保持全数据集索引与随机种子。")
    parser.add_argument("--reproducible_inference", action="store_true",
                        help="关闭 TF32/快速 SDP 并控制 cuDNN，便于零步对照；不保证所有 CUDA 反向严格确定。")
    return parser


def guidance_config(args):
    return GuidanceConfig(steps=args.guidance_steps, learning_rate=args.guidance_lr,
                          trust_radius=args.guidance_trust_radius, lambda_low=args.guidance_lambda_low,
                          lambda_keep=args.guidance_lambda_keep, lambda_range=args.guidance_lambda_range,
                          max_low_change=args.guidance_max_low_change,
                          max_outside_change=args.guidance_max_outside_change,
                          backtracking_steps=args.guidance_backtracking_steps,
                          min_confidence=args.guidance_min_confidence,
                          protection_mode=args.guidance_protection_mode,
                          trace_candidates=args.guidance_trace_candidates)


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
    if args.guidance_target_source == "gt_oracle" and (args.inference_only or args.guidance_steps == 0):
        parser.error("gt_oracle requires positive guidance_steps and ground-truth offline evaluation.")
    if args.inference_only:
        args.calculate_fid, args.save_images = False, True
    args.require_ground_truth = not args.inference_only
    args.event_ablation_mode = "correct"
    return args


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def diagnostic_observation(observation, source, target=None, operator=None):
    """GT 只替换结构目标；W、M0、K0 均由同一事件观测固定，不构造 GT 掩码。"""
    observation = freeze_observation_masks(observation)
    if source == "gt_oracle":
        if target is None:
            raise ValueError("Oracle diagnostic requires an aligned ground-truth tensor.")
        with torch.no_grad():
            edges = image_structure(target.float() * 0.5 + 0.5, operator).detach()
        observation = replace(observation, edges=edges)
    elif source != "event":
        raise ValueError("Unknown guidance target source.")
    observation.validate()
    return observation


def tensor_digest(tensor):
    # 记录未量化张量的哈希，跨组核查 latent、结构权重及掩码完全一致。
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def evaluation_indices(dataset, args):
    """清单只选择样本，不重排全局索引；基线/指导均以 seed+原索引固定随机性。"""
    indices = list(range(len(dataset)))
    if args.sample_manifest:
        with open(args.sample_manifest, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        keys = [(row["scene"], row["frame_id"]) for row in rows]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Sample manifest must be nonempty and contain unique scene/frame_id pairs.")
        lookup = {(s.scene_name, Path(s.low_path).stem): i for i, s in enumerate(dataset.samples)}
        missing = set(keys) - set(lookup)
        if missing:
            raise ValueError(f"Unknown samples in manifest: {sorted(missing)[:5]}")
        indices = sorted(lookup[key] for key in keys)
    return indices[:args.max_samples] if args.max_samples else indices


def configure_inference_numerics(reproducible):
    if reproducible:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch 2.0.1 的 replicate padding 反向不支持严格确定性，明确记录 warn-only。
        torch.use_deterministic_algorithms(True, warn_only=True)
    return dict(reproducible_requested=reproducible, torch_version=str(torch.__version__),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                determinism_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def main(args):
    numeric_settings = configure_inference_numerics(args.reproducible_inference)
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
        event_config = StructureEventConfig.from_dict(checkpoint["event_config"])
        if checkpoint.get("image_size") != args.image_size:
            raise ValueError("Structure checkpoint preprocessing image_size mismatch.")
        if head.config.bins != event_config.bins:
            raise ValueError("Structure checkpoint channel metadata mismatch.")
        dataset = SDEEventStructureDataset("test", args, event_config, args.input_image)
    else:
        # 真正 bypass：steps=0 不需要结构数据/权重，保留旧 checkpoint 的数据行为。
        dataset = SDEEventFilmDataset("test", args, args.input_image)
    indices = evaluation_indices(dataset, args)
    count = len(indices)
    if not count:
        raise ValueError("Evaluation requires at least one sample.")
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
    if args.guidance_target_source == "gt_oracle":
        mode += "_gt_oracle"
    fid = FIDEvaluator([mode], device) if args.calculate_fid else None
    manifest = dict(arguments=vars(args), guidance=asdict(config), metrics=list(metrics) + (["fid"] if fid else []),
                    actual_event_lora_targets=list(model.gen.event_lora_config.targets),
                    actual_unet_lora_parts=list(args.unet_lora_parts),
                    structure_event_config=checkpoint["event_config"] if checkpoint else None,
                    selected_indices=indices, numeric_settings=numeric_settings,
                    uses_gt_for_guidance=args.guidance_target_source == "gt_oracle",
                    guidance_target_source=args.guidance_target_source,
                    mask_policy="original_event_confidence_M0_K0",
                    scheduler=dict(model.gen.noise_scheduler.config), scheduler_timesteps=model.timesteps.tolist())
    scheduler = model.gen.noise_scheduler
    manifest["scheduler_class"] = type(scheduler).__name__
    manifest["scheduler_internal_timesteps"] = getattr(scheduler, "timesteps", torch.tensor([])).tolist()
    manifest["scheduler_previous_timesteps"] = [int(scheduler.previous_timestep(t)) for t in model.timesteps] if hasattr(scheduler, "previous_timestep") else None
    (output / "run_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    rows, diagnostic_rows = [], []
    totals, scene_totals, scene_counts = defaultdict(float), defaultdict(lambda: defaultdict(float)), defaultdict(int)
    diagnostic_log = (output / "idea8_diagnostics.jsonl").open("w", encoding="utf-8")
    try:
        for ordinal, index in enumerate(indices):
            sample = dataset[index]
            rgb = sample["conditioning_pixel_values"][None].to(device, dtype)
            events = sample["events"][None].to(device, dtype)
            refinement = sample["events_refinement"][None].to(device, dtype)
            # 生成器 context 始终只收到 RGB/事件；GT 仅可由显式 oracle 诊断提供结构目标。
            torch.manual_seed(args.seed + index)
            _sync(device)
            start = time.perf_counter()
            with torch.no_grad():
                context = model.gen.prepare_denoising_context(
                    rgb, events, dict(prompt=[args.fixed_prompt], neg_prompt=[args.neg_prompt]))
                observation = observe_events(head, sample["events_structure"][None].to(device), config.min_confidence) if head else None
                if observation is not None:
                    oracle_gt = sample["output_pixel_values"][None].to(device) if args.guidance_target_source == "gt_oracle" else None
                    observation = diagnostic_observation(observation, args.guidance_target_source, oracle_gt, operator)
            prediction, stats = restore_sample(model.gen, context, model.timesteps, rgb, config,
                                                observation, operator, refinement, args.decoder_refinement_enabled)
            _sync(device)
            stats["inference_ms"] = (time.perf_counter() - start) * 1000
            stats.update(sample_index=index, sample_seed=args.seed + index,
                         latent_sha256=tensor_digest(context.encoded_control) if hasattr(context, "encoded_control") else None)
            if observation is not None:
                stats.update(structure_weight_sha256=tensor_digest(observation.confidence),
                             feature_mask_sha256=tensor_digest(observation.feature_mask),
                             protection_mask_sha256=tensor_digest(observation.protection_mask),
                             target_sha256=tensor_digest(observation.edges))
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
            diagnostic_log.write(json.dumps(diagnostic_rows[-1], ensure_ascii=False, allow_nan=False) + "\n")
            diagnostic_log.flush()
            if args.guidance_save_diagnostics and observation is not None:
                diagnostic_dir = output / "diagnostics" / scene
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
                torch.save(dict(edges=observation.edges.cpu(), confidence=observation.confidence.cpu(),
                                feature_mask=observation.feature_mask.cpu(), protection_mask=observation.protection_mask.cpu(),
                                uses_gt=args.guidance_target_source == "gt_oracle"),
                           diagnostic_dir / f"{sample['frame_id']}.pt")
            if args.save_float_predictions:
                float_dir = output / "float_predictions" / scene
                float_dir.mkdir(parents=True, exist_ok=True)
                payload = dict(prediction=prediction.detach().cpu(), sample_index=index,
                               sample_seed=args.seed + index, frame_id=sample["frame_id"])
                if metrics:
                    payload["target"] = target.detach().cpu()
                torch.save(payload, float_dir / f"{sample['frame_id']}.pt")
            print(f"[{ordinal+1}/{count}] {scene}/{sample['frame_id']} {values} "
                  f"accepted={stats['accepted_steps']} reason={stats['reason']} "
                  f"forward={stats['forward_calls']} backward={stats['backward_calls']}", flush=True)
        fid_score = fid.compute([mode])[mode] if fid else None
    finally:
        diagnostic_log.close()
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
