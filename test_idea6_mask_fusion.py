"""idea6 配对消融：同一 RGB latent 下分别单次 UNet 推理，保存 RGB/correct/zero/shuffled。"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from models.rgb_event_mask_lora import iter_mask_layers
from osediff_mask_fusion import OSEDiffMaskFusion, load_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--checkpoint", required=True, help="idea6 checkpoint-STEP 目录或 fusion.pt")
    parser.add_argument("--test_data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ablation_modes", default="rgb,correct,zero,shuffled")
    parser.add_argument("--mixed_precision", choices=("fp16", "bf16", "no"), default="fp16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_samples", type=int, default=0, help="0 为完整测试集")
    parser.add_argument("--skip_metrics", action="store_true", help="只生成图像及零事件一致性检查")
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    args = parser.parse_args(argv)
    modes = args.ablation_modes.split(",")
    if not modes or set(modes) - {"rgb", "correct", "zero", "shuffled"} or len(modes) != len(set(modes)):
        parser.error("ablation_modes must be unique values from rgb,correct,zero,shuffled")
    if args.max_samples < 0:
        parser.error("max_samples must be nonnegative")
    args.ablation_modes = modes
    return args


def save_image(tensor, path, is_mask=False):
    from PIL import Image
    values = tensor.detach().float().cpu()
    values = values.clamp(0, 1) if is_mask else values.clamp(-1, 1).mul(0.5).add(0.5)
    array = values.mul(255).round().byte().permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)


def main(args):
    from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
    # 配对消融不仅要共享 latent，还需消除 CUDA 后端的数值非确定性。
    # 4090 / torch 2.0 的默认半精度路径曾使同一 RGB 重复输出也不同；
    # 统一使用确定性 cuDNN/cuBLAS 与数学 SDPA，保持零事件检查阈值不变。
    if args.enable_xformers_memory_efficient_attention:
        raise ValueError("Paired identity evaluation requires deterministic math SDPA; omit xformers")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if not torch.cuda.is_available():
        raise RuntimeError("Full SD inference requires CUDA")
    if not Path(args.test_data_root).is_dir():
        raise FileNotFoundError(args.test_data_root)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output_dir to avoid mixing experiments")
    output.mkdir(parents=True, exist_ok=True)
    path = Path(args.checkpoint)
    saved = load_checkpoint(path / "fusion.pt" if path.is_dir() else path)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[args.mixed_precision]
    device = torch.device("cuda")
    model = OSEDiffMaskFusion.from_checkpoint(args.pretrained_model_name_or_path, saved,
                                               device=device, frozen_dtype=dtype).eval()
    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()
    common = dict(image_size=256, event_bins=5, event_channels=5, clora_event_active_bins=5,
                  rgb_event_active_bins=5, fixed_prompt=model.fixed_prompt)
    datasets = {"correct": SDEEventFilmDataset("test", SimpleNamespace(**common, event_ablation_mode="correct"),
                                               root_dir=args.test_data_root)}
    if "shuffled" in args.ablation_modes:
        datasets["shuffled"] = SDEEventFilmDataset("test", SimpleNamespace(**common, event_ablation_mode="shuffled"),
                                                   root_dir=args.test_data_root)
    metrics = {}
    if not args.skip_metrics:
        import pyiqa
        # 与 idea4 现有评估一致：Y-channel PSNR/SSIM，默认 PyIQA LPIPS。
        metrics = {name: pyiqa.create_metric(name, device=device, test_y_channel=True, color_space="ycbcr")
                   for name in ("psnr", "ssim")}
        metrics["lpips"] = pyiqa.create_metric("lpips", device=device)
    count = min(len(datasets["correct"]), args.max_samples or len(datasets["correct"]))
    sums, counts = defaultdict(lambda: defaultdict(float)), defaultdict(int)
    max_zero_difference = 0.0
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mode", "scene", "frame_index", "frame_id", *metrics])
        writer.writeheader()
        with torch.no_grad():
            for index in range(count):
                item = datasets["correct"][index]
                rgb = item["conditioning_pixel_values"].unsqueeze(0).to(device)
                gt = item["output_pixel_values"].unsqueeze(0).to(device)
                events = item["events"].unsqueeze(0).to(device)
                generator = torch.Generator(device=device).manual_seed(args.seed + index)
                predictions = {}
                with torch.autocast("cuda", dtype=dtype if dtype != torch.float32 else torch.float16,
                                    enabled=dtype != torch.float32):
                    latent = model.encode_rgb(rgb, generator=generator)
                    for mode in args.ablation_modes:
                        selected = events
                        if mode == "zero":
                            selected = torch.zeros_like(events)
                        elif mode == "shuffled":
                            selected = datasets["shuffled"][index]["events"].unsqueeze(0).to(device)
                        # 每种模式单独一次 UNet；latent 复用，不受模式顺序或 RNG 消耗影响。
                        pred = model(rgb, selected, latent=latent, rgb_only=mode == "rgb",
                                     collect_stats=mode == "correct")
                        predictions[mode] = pred
                        directory = output / mode / item["scene_name"]
                        directory.mkdir(parents=True, exist_ok=True)
                        prefix = f"{item['frame_index'] + 1:04d}"
                        save_image(pred[0], directory / f"{prefix}_result.png")
                        save_image(rgb[0], directory / f"{prefix}_ll.png")
                        save_image(gt[0], directory / f"{prefix}_gt.png")
                        activity = selected[0].abs().sum(0, keepdim=True)
                        activity = activity / activity.amax().clamp_min(1e-6)
                        save_image(activity, directory / f"{prefix}_event.png", is_mask=True)
                        if args.save_masks and mode == "correct":
                            for name, layer in iter_mask_layers(model.unet):
                                if layer.last_mask is not None:
                                    save_image(layer.last_mask[0], directory / f"{prefix}_mask_{name}.png", is_mask=True)
                        # 指标单独用 FP32 计算，避免 autocast 改变既有评估口径。
                        with torch.autocast("cuda", enabled=False):
                            scores = {name: float(metric(pred.float() * 0.5 + 0.5, gt.float() * 0.5 + 0.5).item())
                                      for name, metric in metrics.items()}
                        writer.writerow(dict(mode=mode, scene=item["scene_name"], frame_index=item["frame_index"],
                                             frame_id=item["frame_id"], **scores))
                        for group in ((mode, "overall"), (mode, item["scene_name"])):
                            counts[group] += 1
                            for key, value in scores.items():
                                sums[group][key] += value
                if "rgb" in predictions and "zero" in predictions:
                    difference = (predictions["rgb"] - predictions["zero"]).abs().max().item()
                    max_zero_difference = max(max_zero_difference, difference)
                if index % 20 == 0:
                    print(f"{index+1}/{count}: {item['scene_name']} {item['frame_id']}")
    summary = [dict(mode=mode, scene=scene, count=n,
                    **{key: value / n for key, value in sums[(mode, scene)].items()})
               for (mode, scene), n in sorted(counts.items())]
    checked_zero = {"rgb", "zero"}.issubset(args.ablation_modes)
    result = {"args": vars(args), "metrics": summary, "provenance": model.provenance,
              "zero_rgb_max_abs": max_zero_difference if checked_zero else None,
              "zero_rgb_check_passed": max_zero_difference <= 1e-5 if checked_zero else None,
              "timestep": model.timestep, "latent_pairing": "one sampled latent per RGB, shared by all modes",
              "numerics": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "attention_backend": "math_sdpa", "torch_version": torch.__version__,
                           "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if checked_zero and max_zero_difference > 1e-5:
        raise RuntimeError(f"Zero-event RGB identity failed: max_abs={max_zero_difference}; inspect summary.json")
    print(f"Saved {count} paired samples/mode to {output}")


if __name__ == "__main__":
    main(parse_args())
