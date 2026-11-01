"""在相同 RGB/潜变量上逐级比较 Stage 3 的 correct/zero 事件响应。"""

import argparse
import csv
import dataclasses
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


# 直接以 diagnostics/ 下的文件启动时，显式加入仓库根目录。
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset  # noqa: E402
from models.event_lora_utils import iter_event_lora_modules  # noqa: E402
from osediff_event_film import OSEDiffEventFiLMTest  # noqa: E402
from test_event_film_lora_osediff import parse_args as parse_test_args  # noqa: E402


def tensor_stats(tensor):
    """返回与张量大小无关的幅度统计，统一用 FP32 计算。"""
    value = tensor.detach().float()
    absolute = value.abs()
    return {
        "abs_mean": float(absolute.mean().item()),
        "abs_max": float(absolute.max().item()),
        "rms": float(value.square().mean().sqrt().item()),
    }


def paired_stats(correct, zero):
    delta = correct.detach().float() - zero.detach().float()
    result = tensor_stats(delta)
    result["relative_to_correct_abs_mean"] = result["abs_mean"] / max(
        tensor_stats(correct)["abs_mean"], 1e-12
    )
    return result


def output_pixels(tensor):
    """模型输出 [-1,1] -> 可比较的 [0,255] RGB 浮点值。"""
    return ((tensor.detach().float().clamp(-1, 1) + 1.0) * 127.5).cpu()


def save_preview(path, low, correct, zero, gt, activity):
    """横向保存同一张图的输入、correct、zero、差值热图、GT 和事件活动。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    def rgb(tensor):
        data = tensor.squeeze(0).permute(1, 2, 0).numpy().clip(0, 255)
        return Image.fromarray(data.astype("uint8"), mode="RGB")

    difference = (correct - zero).abs().mean(dim=1, keepdim=True)
    high = max(float(torch.quantile(difference.flatten(), 0.99).item()), 1e-6)
    heat = (difference / high).clamp(0, 1) * 255
    activity_image = activity.float().unsqueeze(0).unsqueeze(0) * 255
    panels = [
        rgb(low), rgb(correct), rgb(zero), rgb(heat.expand(-1, 3, -1, -1)),
        rgb(gt), rgb(activity_image.expand(-1, 3, -1, -1)),
    ]
    canvas = Image.new("RGB", (panels[0].width * len(panels), panels[0].height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * panel.width, 0))
    canvas.save(path)


def region_stats(pixel_diff, event_mask):
    """像素差取 RGB 三通道绝对值均值；区域 mask 来自真实事件。"""
    per_pixel = pixel_diff.mean(dim=1)[0]
    mask = event_mask
    if mask.shape != per_pixel.shape:
        mask = F.interpolate(
            mask.float()[None, None], size=per_pixel.shape,
            mode="nearest"
        )[0, 0].bool()

    def selected_stats(selected):
        if not bool(selected.any()):
            return {"count": 0, "mean_abs_255": None, "max_abs_255": None}
        values = per_pixel[selected]
        return {
            "count": int(values.numel()),
            "mean_abs_255": float(values.mean().item()),
            "max_abs_255": float(values.max().item()),
        }

    return {"active": selected_stats(mask), "inactive": selected_stats(~mask)}


def numeric_summary(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    p95 = sorted_values[min(math.ceil(0.95 * len(values)) - 1, len(values) - 1)]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": p95,
        "max": sorted_values[-1],
    }


def aggregate(output_root):
    """合并两个场景的 JSONL；仅使用本实验结果，不读取 checkpoint。"""
    rows = []
    for scene in ("i_3", "i_33"):
        source = output_root / scene / "per_image.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    selected = {
        "event_nonzero_fraction": lambda r: r["event_input"]["nonzero_fraction"],
        "encoder_diff_abs_mean": lambda r: r["encoder_mean_abs_difference"],
        "direct_head_diff_abs_mean": lambda r: r["layers_mean"]["direct_head_diff_abs_mean"],
        "actual_modulation_diff_abs_mean": lambda r: r["layers_mean"]["actual_modulation_diff_abs_mean"],
        "projected_lora_diff_abs_mean": lambda r: r["layers_mean"]["projected_lora_diff_abs_mean"],
        "unet_diff_abs_mean": lambda r: r["unet"]["abs_mean"],
        "latent_diff_abs_mean": lambda r: r["denoised_latent"]["abs_mean"],
        "pixel_diff_mean_abs_255": lambda r: r["output"]["mean_abs_255"],
        "pixel_diff_max_abs_255": lambda r: r["output"]["max_abs_255"],
        "active_pixel_diff_mean_abs_255": lambda r: r["output"]["regions"]["active"]["mean_abs_255"],
        "inactive_pixel_diff_mean_abs_255": lambda r: r["output"]["regions"]["inactive"]["mean_abs_255"],
        "repeat_correct_output_diff_255": lambda r: r.get("repeat_correct_output_diff_255"),
        "repeat_correct_unet_diff_abs_mean": lambda r: r.get("repeat_correct_unet_diff_abs_mean"),
    }
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "scenes": {}}
    for label, subset in [("i_3", [r for r in rows if r["scene"] == "i_3"]),
                          ("i_33", [r for r in rows if r["scene"] == "i_33"]),
                          ("overall", rows)]:
        result["scenes"][label] = {
            "image_count": len(subset),
            "metrics": {
                name: numeric_summary([getter(r) for r in subset])
                for name, getter in selected.items()
            },
        }
    (output_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "per_image_compact.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "frame_id", *selected])
        for row in rows:
            writer.writerow([row["scene"], row["frame_id"], *[f(row) for f in selected.values()]])
    print(json.dumps({"summary": str(output_root / "summary.json"), "images": len(rows)}))


def run(local, test_args):
    if test_args.event_channels != 5 or test_args.event_bins != 5:
        raise ValueError("This checkpoint requires event_channels=event_bins=5.")
    if test_args.clora_event_active_bins != 5:
        raise ValueError("Checkpoint metadata requires clora_event_active_bins=5.")
    if tuple(test_args.unet_lora_parts) != ("down", "mid", "up"):
        raise ValueError("This experiment requires all ordinary UNet LoRA adapters active.")
    if test_args.generator_timestep != 500:
        raise ValueError("This experiment fixes generator_timestep=500.")
    output_dir = Path(test_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "per_image.jsonl"
    if jsonl_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {jsonl_path}")

    # 严格校验单一 checkpoint 的 5-bin 布局及训练时的普通 LoRA 激活状态。
    checkpoint = torch.load(test_args.osediff_path, map_location="cpu")
    stem = checkpoint["state_dict_event_encoder"]["stem.block.0.weight"]
    if tuple(stem.shape[:2]) != (32, 5):
        raise ValueError(f"Unexpected event encoder stem: {tuple(stem.shape)}")
    active_bins = checkpoint.get("additional_info", {}).get("event_active_bins", {})
    if active_bins.get("clora") != 5:
        raise ValueError(f"Unexpected checkpoint cLoRA event layout: {active_bins}")
    if tuple(checkpoint.get("train_unet_lora_active_parts", ())) != ("down", "mid", "up"):
        raise ValueError("Checkpoint was not trained with all ordinary UNet LoRA active.")
    del checkpoint

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(local.seed)
    torch.cuda.manual_seed_all(local.seed)
    dataset = SDEEventFilmDataset(split="test", args=test_args, root_dir=test_args.input_image)
    dataset.samples = [sample for sample in dataset.samples if sample.scene_name == local.scene]
    if not dataset.samples:
        raise RuntimeError(f"No samples found for {local.scene}")
    if local.max_samples:
        dataset.samples = dataset.samples[:local.max_samples]
    model = OSEDiffEventFiLMTest(test_args)
    if tuple(model.gen.active_unet_lora_parts) != ("down", "mid", "up"):
        raise RuntimeError("UNet LoRA activation did not take effect.")
    event_modules = list(iter_event_lora_modules(model.gen.unet))
    if len(event_modules) != 16 or any(module.variant != "residual_cross_attn" for _, module in event_modules):
        raise RuntimeError(f"Expected 16 residual_cross_attn Event-LoRA layers, found {len(event_modules)}")

    capture = {"mode": None, "layers": {}, "correct_delta": {}, "zero_feats": None, "unet": {}}
    handles = []

    def unet_hook(_module, _inputs, output):
        capture["unet"][capture["mode"]] = output.sample.detach().float().cpu()

    handles.append(model.gen.unet.register_forward_hook(unet_hook))

    def make_layer_hook(name):
        def hook(module, inputs, output):
            if capture["mode"] == "repeat":
                return
            if module.last_z_delta is None or "z_event" not in module.last_aux:
                raise RuntimeError(f"Event modulation inactive in {name}")
            record = capture["layers"].setdefault(name, {})
            mode = capture["mode"]
            record[f"{mode}_head_abs_mean"] = tensor_stats(module.last_aux["z_event"])["abs_mean"]
            record[f"{mode}_modulation_abs_mean"] = tensor_stats(module.last_z_delta)["abs_mean"]
            record[f"{mode}_lora_to_base_norm"] = module.last_debug["lora_norm"] / max(
                module.last_debug["base_norm"], 1e-12
            )
            if mode == "correct":
                capture["correct_delta"][name] = module.last_z_delta.detach().clone()
                # 固定本层 RGB/LoRA query，仅将事件特征换为 zero，测头本身是否读事件。
                hidden = inputs[0].detach()
                spatial = math.isqrt(hidden.shape[1])
                z_map = module.lora_down(hidden).transpose(1, 2).reshape(
                    hidden.shape[0], module.rank, spatial, spatial
                )
                zero_head = module.shared_residual_heads(z_map, capture["zero_feats"])
                raw_difference = module.last_aux["z_event"].float() - zero_head.float()
                record["direct_head_diff_abs_mean"] = tensor_stats(raw_difference)["abs_mean"]
                direct_delta = raw_difference * module.config.modulation_scale
                direct_tokens = direct_delta.flatten(2).transpose(1, 2)
                direct_projected = F.linear(
                    direct_tokens, module.lora_up.weight.float()
                ) * module.scaling
                record["direct_projected_lora_diff_abs_mean"] = tensor_stats(direct_projected)["abs_mean"]
            elif mode == "zero":
                actual_difference = capture["correct_delta"][name].float() - module.last_z_delta.float()
                record["actual_modulation_diff_abs_mean"] = tensor_stats(actual_difference)["abs_mean"]
                projected = F.linear(
                    actual_difference.flatten(2).transpose(1, 2),
                    module.lora_up.weight.float(),
                ) * module.scaling
                record["projected_lora_diff_abs_mean"] = tensor_stats(projected)["abs_mean"]
        return hook

    for name, module in event_modules:
        handles.append(module.register_forward_hook(make_layer_hook(name)))

    manifest = {
        "scene": local.scene, "samples": len(dataset), "seed": local.seed,
        "checkpoint": test_args.osediff_path, "data_root": test_args.input_image,
        "pretrained_model": test_args.pretrained_model_name_or_path,
        "generator_timestep": test_args.generator_timestep,
        "unet_lora_parts": list(test_args.unet_lora_parts),
        "event_bins": test_args.event_bins,
        "event_channels": test_args.event_channels,
        "clora_event_active_bins": test_args.clora_event_active_bins,
        "mixed_precision": test_args.mixed_precision,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "method": "One RGB/VAE latent/prompt per image; direct head contrast fixes its query; no ESG or decoder refinement.",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    try:
        with torch.inference_mode(), jsonl_path.open("w", encoding="utf-8") as stream:
            for index in range(len(dataset)):
                sample = dataset[index]
                low = sample["conditioning_pixel_values"].unsqueeze(0).to(model.device)
                gt = sample["output_pixel_values"].unsqueeze(0)
                events = sample["events"].unsqueeze(0).to(model.device)
                if events.shape[1] != 5:
                    raise RuntimeError(f"Event channel mismatch at {sample['frame_id']}: {events.shape}")
                torch.manual_seed(local.seed + index)
                torch.cuda.manual_seed_all(local.seed + index)
                prompt_batch = {"prompt": [test_args.fixed_prompt], "neg_prompt": [test_args.neg_prompt]}
                context = model.gen.prepare_denoising_context(
                    low.to(dtype=model.weight_dtype), events.to(dtype=model.weight_dtype), prompt_batch
                )
                zero_feats = model.gen.event_encoder(torch.zeros_like(events).to(dtype=model.weight_dtype))
                zero_context = dataclasses.replace(context, event_feats=zero_feats)
                capture.update({"mode": None, "layers": {}, "correct_delta": {},
                                "zero_feats": zero_feats, "unet": {}})
                encoder = {
                    scale: {
                        "correct_abs_mean": tensor_stats(feature)["abs_mean"],
                        "zero_abs_mean": tensor_stats(zero_feats[scale])["abs_mean"],
                        "difference": paired_stats(feature, zero_feats[scale]),
                    }
                    for scale, feature in context.event_feats.items()
                }
                predictions = {}
                latents = {}
                for mode, active_context in (("correct", context), ("zero", zero_context)):
                    # Scheduler/decoder 若使用随机数，两种事件必须消费相同的随机序列。
                    torch.manual_seed(local.seed + index + 1_000_000)
                    torch.cuda.manual_seed_all(local.seed + index + 1_000_000)
                    capture["mode"] = mode
                    latents[mode], _ = model.gen.predict_denoised_latent_from_context(
                        active_context, model.timesteps
                    )
                    predictions[mode] = model.gen.decode_denoised_latent(
                        latents[mode], low.to(dtype=model.weight_dtype)
                    )
                capture["mode"] = None
                correct_pixels = output_pixels(predictions["correct"])
                zero_pixels = output_pixels(predictions["zero"])
                difference = (correct_pixels - zero_pixels).abs()
                mask = events.detach().float().abs().sum(dim=1)[0].cpu() > 0
                layer_values = list(capture["layers"].values())
                layer_keys = (
                    "direct_head_diff_abs_mean", "direct_projected_lora_diff_abs_mean",
                    "actual_modulation_diff_abs_mean", "projected_lora_diff_abs_mean",
                )
                row = {
                    "scene": local.scene,
                    "frame_id": sample["frame_id"],
                    "frame_index": sample["frame_index"],
                    "low_path": sample["low_path"],
                    "event_input": {
                        "nonzero_fraction": float((events != 0).float().mean().item()),
                        "active_pixel_fraction": float(mask.float().mean().item()),
                        "per_bin_abs_mean": [
                            float(value) for value in events.detach().float().abs().mean(dim=(0, 2, 3)).cpu()
                        ],
                    },
                    "encoder": encoder,
                    "encoder_mean_abs_difference": statistics.fmean(
                        item["difference"]["abs_mean"] for item in encoder.values()
                    ),
                    "layers": capture["layers"],
                    "layers_mean": {
                        key: statistics.fmean(item[key] for item in layer_values)
                        for key in layer_keys
                    },
                    "unet": paired_stats(capture["unet"]["correct"], capture["unet"]["zero"]),
                    "denoised_latent": paired_stats(latents["correct"], latents["zero"]),
                    "output": {
                        "mean_abs_255": float(difference.mean().item()),
                        "max_abs_255": float(difference.max().item()),
                        "regions": region_stats(difference, mask),
                    },
                }
                # 每张图同条件再跑一次 correct，记录数值非确定性的逐图下限。
                torch.manual_seed(local.seed + index + 1_000_000)
                torch.cuda.manual_seed_all(local.seed + index + 1_000_000)
                capture["mode"] = "repeat"
                repeat_latent, _ = model.gen.predict_denoised_latent_from_context(context, model.timesteps)
                repeat_image = model.gen.decode_denoised_latent(
                    repeat_latent, low.to(dtype=model.weight_dtype)
                )
                row["repeat_correct_output_diff_255"] = float(
                    (correct_pixels - output_pixels(repeat_image)).abs().mean().item()
                )
                row["repeat_correct_unet_diff_abs_mean"] = paired_stats(
                    capture["unet"]["correct"], capture["unet"]["repeat"]
                )["abs_mean"]
                capture["mode"] = None
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if index < local.preview_count:
                    save_preview(
                        output_dir / "previews" / f"{index + 1:04d}_{sample['frame_id']}.png",
                        output_pixels(low.cpu()), correct_pixels, zero_pixels,
                        output_pixels(gt), mask,
                    )
                if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                    print(json.dumps({
                        "scene": local.scene, "completed": index + 1, "total": len(dataset),
                        "last_pixel_diff_mean_abs_255": row["output"]["mean_abs_255"],
                    }), flush=True)
                del context, zero_context, zero_feats, predictions, latents
    finally:
        for handle in handles:
            handle.remove()


def main():
    local_parser = argparse.ArgumentParser(add_help=False)
    local_parser.add_argument("--scene", choices=("i_3", "i_33"))
    local_parser.add_argument("--max_samples", type=int, default=0)
    local_parser.add_argument("--preview_count", type=int, default=2)
    local_parser.add_argument("--seed", type=int, default=42)
    local_parser.add_argument("--aggregate", type=Path)
    local, remaining = local_parser.parse_known_args()
    if local.aggregate is not None:
        aggregate(local.aggregate)
        return
    if local.scene is None:
        raise ValueError("--scene must be i_3 or i_33")
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        test_args = parse_test_args()
    finally:
        sys.argv = original_argv
    run(local, test_args)


if __name__ == "__main__":
    main()
