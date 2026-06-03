import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataloaders.sde_indoor_eglol_dataset import SDEIndoorEGLolDataset
from models.event_guided_osediff import EventGuidedOSEDiff


def parse_args():
    parser = argparse.ArgumentParser("Event-guided OSEDiff inference")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--test_root", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pretrained_osediff_lora_path", type=str, default="checkpoints/osediff_lle_5-14-1935/checkpoints/model_94001.pkl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--infer_timestep", type=int, default=500)
    parser.add_argument("--event_mode", type=str, default="normal", choices=["normal", "zero", "shuffle"])
    parser.add_argument("--disable_control", action="store_true")
    parser.add_argument("--save_visualization", action="store_true")
    parser.add_argument("--calculate_lpips", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--event_norm", type=str, default="none", choices=["none", "zscore"])
    parser.add_argument("--split_event", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fixed_prompt", type=str, default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", type=str, default="")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--control_alpha_init", type=float, default=1.0)
    parser.add_argument("--control_alpha_final", type=float, default=1.0)
    parser.add_argument("--control_alpha_warmup_steps", type=int, default=1)
    return parser.parse_args()


def tensor_to_uint8_image(tensor):
    tensor = tensor.detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


def event_to_uint8_image(event_tensor):
    event = event_tensor.detach().float().cpu()
    pos = torch.clamp(event, min=0).sum(dim=0)
    neg = torch.clamp(-event, min=0).sum(dim=0)
    if pos.max() > 0:
        pos = pos / pos.max()
    if neg.max() > 0:
        neg = neg / neg.max()
    img = torch.zeros((3, event.shape[1], event.shape[2]))
    img[0] = pos
    img[2] = neg
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def calculate_psnr(pred_01, gt_01):
    mse = torch.mean((pred_01 - gt_01) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def calculate_ssim(pred_arr, gt_arr):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(gt_arr, pred_arr, channel_axis=2, data_range=255))
    except Exception:
        return float("nan")


def ensure_rank_from_checkpoint(args):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    args.lora_rank = int(checkpoint.get("rank_unet", args.lora_rank))
    return checkpoint


def save_image(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)


def main():
    args = parse_args()
    checkpoint = ensure_rank_from_checkpoint(args)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = SDEIndoorEGLolDataset(
        root=args.test_root,
        patch_size=args.patch_size,
        num_bins=args.num_bins,
        is_train=False,
        full_resolution=True,
        event_norm=args.event_norm,
        split_event=args.split_event,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventGuidedOSEDiff(args, stage="stage2", device=device)
    model.load_event_guided_checkpoint(args.checkpoint_path, load_lora=True)
    model.eval()

    lpips_model = None
    if args.calculate_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    metrics_dir = os.path.join(args.output_dir, "metrics")
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    psnr_values, ssim_values, lpips_values = [], [], []
    scene_sorted_indices = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            outputs = model(
                batch,
                infer_timestep=args.infer_timestep,
                disable_control=args.disable_control,
                event_mode=args.event_mode,
                fixed_prompt=args.fixed_prompt,
                neg_prompt=args.neg_prompt,
                global_step=args.control_alpha_warmup_steps,
            )
            pred = outputs["output"].detach()

            batch_size = pred.shape[0]
            for item_idx in range(batch_size):
                seq_name = batch["seq_name"][item_idx]
                frame_id = batch["frame_id"][item_idx]
                scene_dir = os.path.join(images_dir, seq_name)
                # Match test_osediff_lle.py: save images with the sorted
                # 1-based index inside each scene, while keeping the original
                # timestamp frame_id in the metrics CSV for traceability.
                scene_sorted_indices[seq_name] = scene_sorted_indices.get(seq_name, 0) + 1
                sorted_index = scene_sorted_indices[seq_name]
                prefix = f"{sorted_index:04d}"

                low_arr = tensor_to_uint8_image(batch["conditioning_pixel_values"][item_idx])
                gt_arr = tensor_to_uint8_image(batch["output_pixel_values"][item_idx])
                pred_arr = tensor_to_uint8_image(pred[item_idx])
                event_arr = event_to_uint8_image(batch["event_values"][item_idx])
                compare_arr = np.concatenate([low_arr, gt_arr, pred_arr, event_arr], axis=1)

                if args.save_visualization:
                    save_image(os.path.join(scene_dir, f"{prefix}_low.png"), low_arr)
                    save_image(os.path.join(scene_dir, f"{prefix}_gt.png"), gt_arr)
                    save_image(os.path.join(scene_dir, f"{prefix}_pred.png"), pred_arr)
                    save_image(os.path.join(scene_dir, f"{prefix}_event.png"), event_arr)
                    save_image(os.path.join(scene_dir, f"{prefix}_compare.png"), compare_arr)

                pred_01 = pred[item_idx].detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5
                gt_01 = batch["output_pixel_values"][item_idx].detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5
                psnr = calculate_psnr(pred_01, gt_01)
                ssim = calculate_ssim(pred_arr, gt_arr)
                lpips_score = float("nan")
                if lpips_model is not None:
                    lpips_score = lpips_model(
                        pred[item_idx:item_idx + 1].to(device).float(),
                        batch["output_pixel_values"][item_idx:item_idx + 1].to(device).float(),
                    ).mean().item()

                psnr_values.append(psnr)
                if not math.isnan(ssim):
                    ssim_values.append(ssim)
                if not math.isnan(lpips_score):
                    lpips_values.append(lpips_score)
                rows.append({
                    "seq_name": seq_name,
                    "sorted_index": sorted_index,
                    "frame_id": frame_id,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips_score,
                    "event_mode": args.event_mode,
                    "disable_control": args.disable_control,
                })
                print(f"[{batch_idx + 1}/{len(loader)}] {seq_name}/{prefix} frame_id={frame_id} PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS={lpips_score:.4f}", flush=True)

    per_image_path = os.path.join(metrics_dir, "metrics_per_image.csv")
    with open(per_image_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_name", "sorted_index", "frame_id", "psnr", "ssim", "lpips", "event_mode", "disable_control"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(metrics_dir, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint_path: {args.checkpoint_path}\n")
        f.write(f"event_mode: {args.event_mode}\n")
        f.write(f"disable_control: {args.disable_control}\n")
        f.write(f"num_images: {len(rows)}\n")
        f.write(f"PSNR: {float(np.mean(psnr_values)) if psnr_values else float('nan'):.6f}\n")
        f.write(f"SSIM: {float(np.mean(ssim_values)) if ssim_values else float('nan'):.6f}\n")
        f.write(f"LPIPS: {float(np.mean(lpips_values)) if lpips_values else float('nan'):.6f}\n")

    print(f"[Done] metrics: {per_image_path}")
    print(f"[Done] summary: {summary_path}")


if __name__ == "__main__":
    main()
