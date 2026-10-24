"""idea7：全通道 UNet 特征的类 DPS 推理；正常光仅由末尾独立评估读取。"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path


def build_parser():
    """轻量 CLI：--help 和配置测试不导入模型、指标或下载权重。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "pair-manifest", "output-dir",
                 "pretrained-model-name-or-path", "event-lora-checkpoint"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--observation-calibration")
    parser.add_argument("--evaluation-manifest", help="可选独立 GT 清单；不参与生成或候选选择")
    parser.add_argument("--dps-steps", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--dps-up-block", default="penultimate")
    parser.add_argument("--dps-step-rel", type=float, default=0.01)
    parser.add_argument("--dps-radius-rel", type=float, default=0.05)
    parser.add_argument("--feedback-mode", choices=("correct", "shifted"), default="correct")
    parser.add_argument("--feedback-shift", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--generator-timestep", type=int)
    parser.add_argument("--unet-lora-parts", help="默认读取 checkpoint；旧权重缺字段时必须显式给出 none 或 down,mid,up")
    parser.add_argument("--fixed-prompt", default="high quality, bright, detailed photo")
    parser.add_argument("--neg-prompt", default="")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--event-bins", type=int)
    parser.add_argument("--clora-event-active-bins", type=int)
    parser.add_argument("--rgb-event-active-bins", type=int)
    parser.add_argument("--event-base-channels", type=int)
    # 结构参数缺省继承 checkpoint，显式冲突时报错，避免加载了另一种网络。
    for name, kind in (("event-lora-variant", str), ("lora-rank", int),
                       ("mod-channels", int), ("modulation-scale", float),
                       ("mod-num-heads", int), ("mod-window-size", int)):
        parser.add_argument("--" + name, type=kind)
    parser.add_argument("--decoder-refinement-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--decoder-refinement-variant", choices=("legacy_fused", "cgdrs"))
    parser.add_argument("--rgb-event-pretrained-root")
    parser.add_argument("--decoder-fusion-weight", type=float)
    return parser


def _choose(args, name, saved, default=None):
    requested = getattr(args, name, None)
    if saved is not None and requested is not None and requested != saved:
        raise ValueError(f"{name} conflicts with checkpoint: requested={requested!r}, saved={saved!r}")
    setattr(args, name, saved if requested is None and saved is not None else
            (default if requested is None else requested))


def resolve_checkpoint_args(args, checkpoint):
    """解析可审计的有效模型配置，不从训练 timestep 区间随意选一个值。"""
    config = dict(checkpoint.get("event_lora_config") or {})
    variant = config.get("variant", checkpoint.get("event_lora_variant", "film_affine"))
    variant = "film_affine" if variant == "film" else variant
    fields = {
        "event_lora_variant": variant,
        "lora_rank": config.get("rank", checkpoint.get("rank_unet", 4)),
        "mod_channels": config.get("mod_channels", 64 if variant.startswith("residual_cross_attn") else 32),
        "modulation_scale": config.get("modulation_scale", 0.1),
        "mod_num_heads": config.get("mod_num_heads", 4),
        "mod_window_size": config.get("mod_window_size", 8),
    }
    for name, value in fields.items():
        _choose(args, name, value)
    if checkpoint.get("rank_vae", args.lora_rank) != args.lora_rank:
        raise ValueError("This entry requires equal VAE/UNet LoRA ranks, as the existing constructor does.")
    args.event_lora_cbam_reduction = config.get("cbam_reduction", 8)
    args.event_lora_cbam_spatial_kernel = config.get("cbam_spatial_kernel", 7)
    args.event_lora_output_gate_channels = config.get("output_gate_channels", "spatial")
    encoder_state = checkpoint.get("state_dict_event_encoder", checkpoint.get("state_dict_existing_event_encoder", {}))
    stem = encoder_state.get("stem.block.0.weight")
    _choose(args, "event_base_channels", int(stem.shape[0]) if stem is not None else None, default=32)
    if stem is not None and int(stem.shape[1]) != args.event_bins:
        raise ValueError("EventEncoder input channels conflict with manifest event_bins.")
    saved_bins = (checkpoint.get("additional_info") or {}).get("event_active_bins") or {}
    for name, key in (("event_bins", "total_channels"),
                      ("clora_event_active_bins", "clora"),
                      ("rgb_event_active_bins", "rgb_event_encoder")):
        if key in saved_bins and getattr(args, name) != saved_bins[key]:
            raise ValueError(f"{name} conflicts with checkpoint event_active_bins: {saved_bins[key]}")
    parts = getattr(args, "unet_lora_parts", None)
    if parts is None:
        if "train_unet_lora_active_parts" not in checkpoint:
            raise ValueError("Checkpoint lacks adapter activation metadata; provide --unet-lora-parts explicitly.")
        parts = checkpoint["train_unet_lora_active_parts"]
    if isinstance(parts, str):
        parts = [] if parts.strip().lower() in ("", "none") else parts.split(",")
    parts = [str(part).strip().lower() for part in parts]
    if len(parts) != len(set(parts)) or set(parts) - {"down", "mid", "up"}:
        raise ValueError("--unet-lora-parts must be none or unique down,mid,up groups.")
    args.unet_lora_parts = tuple(part for part in ("down", "mid", "up") if part in parts)
    spec = (checkpoint.get("timestep_config") or {}).get("generator_train_timestep")
    if isinstance(spec, str):
        spec = [int(value.strip()) for value in spec.split(",")]
    if isinstance(spec, (int, float)):
        spec = [int(spec), int(spec)]
    if spec is not None:
        if len(spec) == 1:
            spec = [spec[0], spec[0]]
        if len(spec) != 2 or int(spec[0]) > int(spec[1]):
            raise ValueError("Invalid generator_train_timestep metadata.")
    if args.generator_timestep is None:
        if spec is None or spec[0] != spec[1]:
            raise ValueError("Provide --generator-timestep: checkpoint has no unique inference timestep.")
        args.generator_timestep = int(spec[0])
    if not 0 <= args.generator_timestep <= 999:
        raise ValueError("generator_timestep must be in [0,999].")
    if spec is not None and not spec[0] <= args.generator_timestep <= spec[1]:
        raise ValueError("Inference timestep is outside the checkpoint training interval.")
    has_decoder = "state_dict_decoder_refinement" in checkpoint
    _choose(args, "decoder_refinement_enabled", has_decoder)
    decoder = checkpoint.get("decoder_refinement_config") or {}
    _choose(args, "decoder_refinement_variant", decoder.get("variant", "legacy_fused"))
    _choose(args, "decoder_fusion_weight", decoder.get("fusion_weight", 1.0))
    args.decoder_fusion_enabled = args.decoder_refinement_enabled
    args.refinement_fusion_mode = "both"
    args.decoder_fusion_init = decoder.get("fusion_init", "near_zero")
    args.cgdrs_event_levels = decoder.get("cgdrs_event_levels", (1, 2, 3))
    args.cgdrs_gate_bias = decoder.get("cgdrs_gate_bias", -2.0)
    args.cgdrs_activity_mode = decoder.get("cgdrs_activity_mode", "abs_mean")
    args.rgb_event_strict_load = True
    args.resume_ignore_mismatch = False
    args.osediff_path = args.event_lora_checkpoint
    args.event_channels = args.event_bins
    args.event_ablation_mode = "correct"
    return args


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    """诊断中的非有限数写成 null，保证 JSON 可被标准解析器读取。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value
                          for key, value in row.items()} for row in rows)


def _output_name(scene, frame):
    # ID 仅用于命名，不允许 manifest 中的路径分隔符把输出写到目录外。
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in f"{scene}_{frame}")[:140]
    suffix = hashlib.sha256(f"{scene}\0{frame}".encode()).hexdigest()[:10]
    return f"{safe}_{suffix}.png"


def frame_seed_indices(pairs):
    """可显式保留旧 dataset 全局索引；未提供时使用 manifest 首次出现顺序。"""
    explicit = any("seed_indices" in pair for pair in pairs)
    result = {}
    for pair in pairs:
        for role, frame_id in (("previous", pair.get("prev_id")), ("current", pair["curr_id"])):
            if frame_id is None:
                continue
            key = (pair["scene_id"], frame_id)
            if explicit:
                value = pair.get("seed_indices", {}).get(role)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("Explicit seed_indices must cover every frame with nonnegative integer indices.")
                if key in result and result[key] != value:
                    raise ValueError("Same frame has inconsistent seed_indices across pairs.")
                result[key] = value
            else:
                result.setdefault(key, len(result))
    return result


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _save_rgb(image, path):
    from PIL import Image
    array = (image.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(array).save(path)


def summarize_rows(rows):
    """聚合真实采用率和开销，跳过样本也计入端到端分母。"""
    result = {"count": len(rows), "accepted": sum(row["selected_iteration"] > 0 for row in rows)}
    result["acceptance_rate"] = result["accepted"] / len(rows) if rows else 0.0
    result["skip_reasons"] = {reason: sum(row.get("skip_reason") == reason for row in rows)
                              for reason in sorted({row.get("skip_reason", "") for row in rows}) if reason}
    for key in ("end_to_end_ms", "candidate_ms", "peak_cuda_mb", "reference_ms"):
        values = sorted(float(row[key]) for row in rows if key in row)
        if values:
            result[key + "_mean"] = sum(values) / len(values)
            result[key + "_p95"] = values[max(0, math.ceil(0.95 * len(values)) - 1)]
    for key in ("forward_calls", "backward_calls", "reference_forward_calls"):
        result[key + "_total"] = sum(row.get(key, 0) for row in rows)
    return result


def main(args, model_factory=None):
    import torch
    from dataloaders.sde_event_pair_dataset import SDEEventPairDataset
    from models.event_observation_proxy import EventObservationProxy
    from models.feature_dps_guidance import FeatureDPSConfig, FeatureDPSRunner, SkipFeatureDPS, freeze_for_feature_dps

    if args.image_size != 256 or args.max_samples < 0:
        raise ValueError("MVP requires image_size=256 and max_samples>=0.")
    index = None if args.dps_up_block == "penultimate" else int(args.dps_up_block)
    config = FeatureDPSConfig(steps=args.dps_steps, up_block_index=index,
                              step_rel=args.dps_step_rel, radius_rel=args.dps_radius_rel)
    enabled = args.dps_steps > 0 and args.dps_radius_rel > 0 and args.dps_step_rel > 0
    if enabled and not args.observation_calibration:
        raise ValueError("Enabled guidance requires --observation-calibration.")
    if args.feedback_mode == "shifted" and args.feedback_shift == 0:
        raise ValueError("Shifted feedback needs a nonzero --feedback-shift.")
    raw_manifest = json.loads(Path(args.pair_manifest).read_text(encoding="utf-8"))
    for name in ("event_bins", "clora_event_active_bins", "rgb_event_active_bins"):
        if getattr(args, name) is None:
            setattr(args, name, raw_manifest["settings"][name])
    args.event_channels = args.event_bins
    dataset = SDEEventPairDataset(args.pair_manifest, data_root=args.data_root, args=args,
                                  validate_files=True, require_measurement=enabled)
    proxy = EventObservationProxy.load(args.observation_calibration) if enabled else None
    if proxy is not None:
        proxy.validate_manifest(dataset.manifest)
    frame_indices = frame_seed_indices(dataset.manifest["pairs"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        _write_json(output / "preflight.json", {"status": "passed", "samples": len(dataset),
                    "guidance_enabled": enabled, "manifest_sha256": _hash_file(args.pair_manifest)})
        print("[idea7] Data/calibration preflight passed; no model loaded.")
        return
    if not Path(args.event_lora_checkpoint).is_file():
        raise FileNotFoundError(args.event_lora_checkpoint)
    if not Path(args.pretrained_model_name_or_path).is_dir():
        raise NotADirectoryError(args.pretrained_model_name_or_path)
    if not torch.cuda.is_available() and model_factory is None:
        raise RuntimeError("The existing OSEDiff constructor requires CUDA. CPU unit tests/preflight remain available.")
    # 结构与 timestep 在构造模型前确定；测试集 GT 完全不参与该流程。
    checkpoint = torch.load(args.event_lora_checkpoint, map_location="cpu", weights_only=False)
    resolve_checkpoint_args(args, checkpoint)
    del checkpoint
    if model_factory is None:
        from osediff_event_film import OSEDiffEventFiLMTest
        model_factory = OSEDiffEventFiLMTest
    model = model_factory(args)
    freeze_for_feature_dps(model)
    device, dtype = model.device, model.weight_dtype
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    runner = FeatureDPSRunner(config)
    baseline_runner = FeatureDPSRunner(FeatureDPSConfig(steps=0))
    checkpoint_hash = _hash_file(args.event_lora_checkpoint)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    resolved = {**vars(args), "base_commit": commit, "checkpoint_sha256": checkpoint_hash,
                "manifest_sha256": _hash_file(args.pair_manifest),
                "calibration_sha256": _hash_file(args.observation_calibration) if proxy else None,
                "torch_version": torch.__version__, "up_blocks": len(model.gen.unet.up_blocks),
                "seed_rule": "base_seed_plus_explicit_index" if any("seed_indices" in p for p in dataset.pairs) else "base_seed_plus_manifest_first_occurrence",
                "resolved_up_block": len(model.gen.unet.up_blocks) - 2 if index is None else index}
    source_names = ("test_idea7_dps.py", "osediff_event_film.py", "models/unet_2d_condition.py",
                    "models/feature_dps_guidance.py", "models/event_observation_proxy.py",
                    "dataloaders/sde_event_pair_dataset.py")
    resolved["source_sha256"] = {name: _hash_file(Path(__file__).parent / name) for name in source_names}
    _write_json(output / "resolved_config.json", resolved)
    print(json.dumps(_json_safe(resolved), ensure_ascii=False, indent=2))
    for directory in ("images_baseline", "images_dps"):
        (output / directory).mkdir(exist_ok=True)

    # 一条链只保留最近 baseline；重排/断链会重新生成参考，不读取修正输出。
    cache_key, cache_image = None, None
    groups = {}
    def donor_group(pair):
        return (pair["scene_id"], pair["observation_group"], pair["t_curr"] - pair["t_prev"],
                json.dumps(pair["geometry"], sort_keys=True))
    for pair_index, pair in enumerate(dataset.manifest["pairs"]):
        if pair.get("prev_id") is not None and not pair.get("skip_reason"):
            groups.setdefault(donor_group(pair), []).append(pair_index)

    def generate_for(frame, scene):
        """固定一次编码采样，后续每个候选复用同一 context 和 scheduler 随机量。"""
        seed = args.seed + frame_indices[(scene, frame["frame_id"])]
        rgb = frame["conditioning_pixel_values"].unsqueeze(0).to(device, dtype=dtype)
        events = frame["events"].unsqueeze(0).to(device, dtype=dtype)
        refinement = frame["events_refinement"].unsqueeze(0).to(device, dtype=dtype)
        batch = {"prompt": [args.fixed_prompt], "neg_prompt": [args.neg_prompt]}
        _sync(torch, device)
        begin = time.perf_counter()
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            # manual_seed() 会修改所有 CUDA 卡；这里只播种实际使用的 generator。
            torch.random.default_generator.manual_seed(seed)
            for cuda_index in cuda_devices:
                torch.cuda.default_generators[cuda_index].manual_seed(seed)
            context = model.gen.prepare_fixed_denoising_context(rgb, events, batch)
            # 保留编码后 RNG，复现原 baseline 的“VAE 采样→scheduler”随机序列。
            after_encode_cpu_rng = torch.get_rng_state()
            after_encode_cuda_rng = {cuda_index: torch.cuda.get_rng_state(cuda_index) for cuda_index in cuda_devices}
        _sync(torch, device)
        context_ms = (time.perf_counter() - begin) * 1000

        def generate(state):
            with torch.random.fork_rng(devices=cuda_devices):
                torch.set_rng_state(after_encode_cpu_rng)
                for cuda_index, rng in after_encode_cuda_rng.items():
                    torch.cuda.set_rng_state(rng, cuda_index)
                latent, _ = model.gen.predict_denoised_latent_from_context(
                    context, model.timesteps, feature_guidance=state)
                image = model.gen.decode_denoised_latent(
                    latent, rgb, events_refinement=refinement if args.decoder_refinement_enabled else None,
                    decoder_fusion_enabled=args.decoder_refinement_enabled,
                    refinement_fusion_mode="both" if args.decoder_refinement_enabled else None)
                return image.float() * 0.5 + 0.5
        return generate, context_ms, seed

    rows = []
    limit = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    for sample_index in range(limit):
        _sync(torch, device)
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        item = dataset[sample_index]
        metadata, frame = item["metadata"], item["current"]
        scene, frame_id = metadata["scene_id"], metadata["curr_id"]
        measurement = item["measurement"]
        skip_reason = item.get("skip_reason")
        donor_id = None
        if enabled and not skip_reason and args.feedback_mode == "shifted":
            candidates = groups[donor_group(metadata)]
            shift = args.feedback_shift % len(candidates)
            if not shift:
                skip_reason = "no_distinct_shift_donor"
            else:
                donor = dataset[candidates[(candidates.index(sample_index) + shift) % len(candidates)]]
                if (donor["metadata"]["geometry"] != metadata["geometry"] or
                        donor["metadata"]["observation_group"] != metadata["observation_group"]):
                    raise ValueError("Shifted diagnostic requires matching geometry and observation group.")
                donor_id = donor["metadata"]["curr_id"]
                if donor.get("skip_reason") or donor["measurement"] is None:
                    skip_reason = "unavailable_shift_donor"
                else:
                    measurement = donor["measurement"]
        data_ms = (time.perf_counter() - start) * 1000
        reference, reference_calls, cache_hit = None, 0, False
        _sync(torch, device)
        ref_start = time.perf_counter()
        if enabled and not skip_reason:
            previous = item["previous"]
            wanted = (scene, previous["baseline_cache_key"], checkpoint_hash)
            if wanted == cache_key:
                reference, cache_hit = cache_image, True
            else:
                previous_generate, _, _ = generate_for(previous, scene)
                previous_result = baseline_runner.run(previous_generate,
                    cleanup=model.gen.clear_feature_dps_caches, cuda_devices=cuda_devices)
                reference = previous_result.baseline
                reference_calls = previous_result.stats["forward_calls"]
                del previous_generate, previous_result
        _sync(torch, device)
        reference_ms = (time.perf_counter() - ref_start) * 1000
        generate, context_ms, seed = generate_for(frame, scene)
        energy_holder = {}

        def loss_factory(baseline):
            energy = proxy.prepare_energy(reference, baseline, measurement, metadata)
            energy_holder["energy"] = energy
            if energy.skip_reason:
                raise SkipFeatureDPS(energy.skip_reason)
            return energy

        _sync(torch, device)
        candidate_start = time.perf_counter()
        result = runner.run(generate, loss_factory if enabled and not skip_reason else None,
                            skip_reason=skip_reason, cleanup=model.gen.clear_feature_dps_caches,
                            cuda_devices=cuda_devices)
        _sync(torch, device)
        candidate_ms = (time.perf_counter() - candidate_start) * 1000
        # 缓存的是本帧零修正图，绝不以 result.image 形成反馈递推。
        cache_key = (scene, frame["baseline_cache_key"], checkpoint_hash)
        cache_image = result.baseline.detach()
        filename = _output_name(scene, frame_id)
        _save_rgb(result.baseline, output / "images_baseline" / filename)
        _save_rgb(result.image, output / "images_dps" / filename)
        row = {"scene_id": scene, "frame_id": frame_id, "seed": seed, "filename": filename,
               "geometry": metadata["geometry"], "feedback_mode": args.feedback_mode,
               "feedback_donor_id": donor_id, **result.stats,
               "data_ms": data_ms, "context_ms": context_ms, "reference_ms": reference_ms,
               "reference_forward_calls": reference_calls, "reference_cache_hit": cache_hit,
               "candidate_ms": candidate_ms,
               "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0}
        row["gradient_rms"] = next((entry["gradient_rms"] for entry in reversed(result.history)
                                     if "gradient_rms" in entry), None)
        if "energy" in energy_holder:
            row.update(energy_holder["energy"].diagnostics)
        row["end_to_end_ms"] = (time.perf_counter() - start) * 1000
        rows.append(_json_safe(row))
        _write_json(output / "last_sample.json", {"row": row, "history": result.history})
        if len(rows) % 25 == 0 or sample_index + 1 == limit:
            _write_csv(output / "per_image.csv", rows)
        print(f"[idea7] {sample_index + 1}/{limit} {scene}/{frame_id}: selected={row['selected_iteration']} reason={row.get('skip_reason')}")
        del generate, result, energy_holder, reference

    _write_json(output / "summary.json", summarize_rows(rows))
    _write_json(output / "scene_summary.json", {scene: summarize_rows([row for row in rows if row["scene_id"] == scene])
                for scene in sorted({row["scene_id"] for row in rows})})
    if args.evaluation_manifest:
        evaluate_saved_outputs(output, rows, args.evaluation_manifest, args.data_root, device)


def evaluate_saved_outputs(output, rows, manifest_path, data_root, device):
    """生成结束后单独读取 GT；指标仅评估已保存的 PNG，不反向影响候选。"""
    import numpy as np
    import torch
    import torch.nn.functional as F
    import pyiqa
    from PIL import Image
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    lookup = {(entry["scene_id"], entry["frame_id"]): entry["normal_path"] for entry in manifest["frames"]}
    if len(lookup) != len(manifest["frames"]):
        raise ValueError("Duplicate frame identity in evaluation manifest.")
    metrics = {"psnr": pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device),
               "ssim": pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device),
               "lpips": pyiqa.create_metric("lpips", device=device).to(device)}

    def load(path):
        with Image.open(path) as image:
            return torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0)

    evaluated = []
    with torch.no_grad():
        for row in rows:
            target = load(Path(data_root) / lookup[(row["scene_id"], row["frame_id"])])
            geometry = row["geometry"]
            if list(target.shape[-2:]) != [geometry["sensor_height"], geometry["sensor_width"]]:
                raise ValueError("GT dimensions disagree with aligned sensor geometry.")
            top, left, height, width = geometry["crop"]
            target = F.interpolate(target[:, :, top:top+height, left:left+width], size=(256, 256), mode="bilinear", align_corners=False).to(device)
            for label in ("baseline", "dps"):
                prediction = load(output / f"images_{label}" / row["filename"]).to(device)
                evaluated.append({"scene_id": row["scene_id"], "frame_id": row["frame_id"], "method": label,
                                  **{name: float(metric(prediction, target).item()) for name, metric in metrics.items()}})
    _write_csv(output / "evaluation.csv", evaluated)
    _write_json(output / "evaluation_summary.json", {label: {name: sum(row[name] for row in evaluated if row["method"] == label) / max(len(rows), 1)
                for name in metrics} for label in ("baseline", "dps")})


if __name__ == "__main__":
    main(build_parser().parse_args())
