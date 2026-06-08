import argparse
import csv
import os
from collections import defaultdict

import pyiqa
import torch
from PIL import Image
from torchvision import transforms

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from osediff_event_film import OSEDiffEventFiLMTest


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
    parser.add_argument("--mod_channels", type=int, default=32)
    parser.add_argument("--modulation_scale", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--ablation_modes", type=str, default="correct,zero,shuffled")
    parser.add_argument("--save_debug_vis", action="store_true")
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = SDEEventFilmDataset(split="test", args=args, root_dir=args.input_image)
    model = OSEDiffEventFiLMTest(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device)
    ssim_metric = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device)
    modes = [m.strip() for m in args.ablation_modes.split(",") if m.strip()]

    scene_counts = defaultdict(int)
    rows = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        scene_name = sample["scene_name"]
        scene_counts[scene_name] += 1
        prefix = f"{scene_counts[scene_name]:04d}"

        lq = sample["conditioning_pixel_values"].unsqueeze(0).to(device)
        gt = sample["output_pixel_values"].unsqueeze(0).to(device)
        events = sample["events"].unsqueeze(0).to(device)
        shuffled_events = dataset[(idx + 1) % len(dataset)]["events"].unsqueeze(0).to(device)

        for mode in modes:
            if mode == "correct":
                event_input = events
            elif mode == "zero":
                event_input = torch.zeros_like(events)
            elif mode == "shuffled":
                event_input = shuffled_events
            else:
                raise ValueError(f"Unknown ablation mode: {mode}")

            with torch.no_grad():
                pred = model(lq, event_input, prompt=args.fixed_prompt)

            out_dir = os.path.join(args.output_dir, mode, scene_name)
            os.makedirs(out_dir, exist_ok=True)
            low_path = os.path.join(out_dir, f"{prefix}_lowlight.png")
            result_path = os.path.join(out_dir, f"{prefix}_result.png")
            gt_path = os.path.join(out_dir, f"{prefix}_gt.png")
            tensor_to_pil(lq[0]).save(low_path)
            tensor_to_pil(pred[0]).save(result_path)
            tensor_to_pil(gt[0]).save(gt_path)

            pred_01 = pred.float().clamp(-1, 1) * 0.5 + 0.5
            gt_01 = gt.float().clamp(-1, 1) * 0.5 + 0.5
            psnr = psnr_metric(pred_01, gt_01).item()
            ssim = ssim_metric(pred_01, gt_01).item()
            active_l1 = event_active_l1(pred.float(), gt.float(), event_input.float())
            debug_stats = model.debug_stats()

            if args.save_debug_vis:
                save_event_activity(event_input[0].float(), os.path.join(out_dir, f"{prefix}_event_activity.png"))

            rows.append({
                "mode": mode,
                "scene": scene_name,
                "prefix": prefix,
                "frame_id": sample["frame_id"],
                "psnr": psnr,
                "ssim": ssim,
                "event_active_l1": active_l1,
                "dgamma_abs_mean": debug_stats.get("dgamma_abs_mean", 0.0),
                "dbeta_abs_mean": debug_stats.get("dbeta_abs_mean", 0.0),
                "result_path": result_path,
            })
            print(
                f"[{idx + 1}/{len(dataset)}] {mode}/{scene_name}/{prefix} | "
                f"PSNR={psnr:.4f} | SSIM={ssim:.4f} | event_L1={active_l1:.6f}"
            )

    csv_path = os.path.join(args.output_dir, "metrics_event_film_ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Done] Metrics: {csv_path}")


if __name__ == "__main__":
    main(parse_args())
