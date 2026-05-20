"""
Debug script: visualize training inputs and GT from LowLightDataset.

This script uses LowLightDataset (the same one used in train_osediff_lle.py)
to ensure the visualized samples exactly match what the model sees during training.

Saves to output_dir/:
    sample_XXXX_input.png          low-light input   (denormalized from [-1,1] to [0,255])
    sample_XXXX_gt.png             normal-light GT  (denormalized from [-1,1] to [0,255])
    sample_XXXX_side_by_side.png   input | GT horizontally

Usage:
    python debug_visualize_dataset.py \
        --train_data_root D:/data/SDE_indoor/train \
        --output_dir debug_dataset_vis \
        --num_samples 20 \
        --image_size 512

The dataset structure LowLightDataset expects:
    train_data_root/
        i_0/
            low/       <- low-light images
            normal/    <- normal-light images
        i_6/
            low/
            normal/
        ...
Pass --low_subdir and --normal_subdir if your subfolder names differ.
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image

from dataloaders.lle_dataset import LowLightDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def denorm_to_pil(tensor):
    """
    Reverse LowLightDataset normalization:
        [-1, 1]  --(+1)/2-->  [0, 1]  --*255-->  [0, 255]  -->  PIL RGB
    """
    t = tensor.detach().cpu().float()
    t = torch.clamp(t, -1, 1)
    t = (t + 1) / 2
    t = t * 255.0
    t = t.numpy().transpose(1, 2, 0)
    t = np.clip(t, 0, 255).astype(np.uint8)
    return Image.fromarray(t)


def make_side_by_side(img1, img2, gap=4, bg=(255, 255, 255)):
    """Concatenate two PIL images horizontally with a white gap."""
    w1, h1 = img1.size
    w2, h2 = img2.size
    assert h1 == h2, f"Height mismatch: {h1} vs {h2}"
    canvas = Image.new("RGB", (w1 + gap + w2, h1), bg)
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1 + gap, 0))
    return canvas


# ---------------------------------------------------------------------------
# Minimal args object to satisfy LowLightDataset's constructor
# ---------------------------------------------------------------------------
class _MockArgs:
    pass


def build_mock_args(args):
    mock = _MockArgs()
    mock.train_data_root  = args.train_data_root
    mock.low_light_subdir = args.low_subdir
    # NOTE: LowLightDataset derives normal_subdir by replacing "low" in low_light_subdir.
    # Your low_subdir MUST contain the word "low" (e.g. "low", "i_n/low") so that
    #   low_light_subdir.replace("low", "normal")  =>  "normal"
    # works correctly.  If your folder has a different name, adjust --low_subdir accordingly.
    mock.normal_subdir    = args.normal_subdir  # not actually used by LowLightDataset
    mock.image_size       = args.image_size
    return mock


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_root", type=str, required=True,
        help="Root directory of the SDE dataset (contains i_0/, i_6/, ... scene folders)")
    parser.add_argument("--low_subdir", type=str, default="low",
        help="Subdirectory name for low-light images inside each scene folder")
    parser.add_argument("--normal_subdir", type=str, default="normal",
        help="Subdirectory name for normal-light images inside each scene folder")
    parser.add_argument("--output_dir", type=str, default="debug_dataset_vis",
        help="Folder to save visualizations")
    parser.add_argument("--num_samples", type=int, default=20,
        help="Number of samples to visualize")
    parser.add_argument("--image_size", type=int, default=512,
        help="Crop/resize size (must match training)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--split",  type=str, default="test",
        choices=["train", "test"],
        help="'test' = center-crop + resize (like val split); "
             "'train' = random crop + flip (like training)")
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 62)
    print("  SDE Indoor Dataset Visualizer  (via LowLightDataset)")
    print("=" * 62)
    print(f"  data_root     : {args.train_data_root}")
    print(f"  low_subdir    : {args.low_subdir}")
    print(f"  normal_subdir : {args.normal_subdir}")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  image_size    : {args.image_size}")
    print(f"  split         : {args.split}")
    print(f"  num_samples   : {args.num_samples}")
    print()

    # Instantiate LowLightDataset (same as train_osediff_lle.py line ~284)
    mock_args = build_mock_args(args)
    dataset   = LowLightDataset(split=args.split, args=mock_args)

    print(f"[LowLightDataset] Loaded {len(dataset)} image pairs")
    if len(dataset) == 0:
        print("[ERROR] No samples found!")
        print(f"  Check: train_data_root = {args.train_data_root}  (exists: {os.path.exists(args.train_data_root)})")
        print(f"  Each scene folder should contain:")
        print(f"    <scene>/{args.low_subdir}/<images.png>")
        print(f"    <scene>/{args.normal_subdir}/<images.png>")
        return

    # Source file paths from the dataset (for reference)
    low_paths  = dataset.low_list
    norm_paths = dataset.normal_list

    indices = list(range(len(dataset)))
    if args.shuffle:
        np.random.shuffle(indices)

    num_save = min(args.num_samples, len(dataset))

    stats_in = {"min": [], "max": [], "mean": []}
    stats_gt = {"min": [], "max": [], "mean": []}

    print(f"\n[Saving {num_save} pairs]")
    print("-" * 72)

    for i in range(num_save):
        idx = indices[i]
        sample = dataset[idx]

        # Keys match LowLightDataset.__getitem__ output
        cond_t = sample["conditioning_pixel_values"]   # low-light   [-1, 1]
        gt_t   = sample["output_pixel_values"]         # normal GT   [-1, 1]

        stats_in["min"].append(cond_t.min().item())
        stats_in["max"].append(cond_t.max().item())
        stats_in["mean"].append(cond_t.mean().item())
        stats_gt["min"].append(gt_t.min().item())
        stats_gt["max"].append(gt_t.max().item())
        stats_gt["mean"].append(gt_t.mean().item())

        cond_img = denorm_to_pil(cond_t)
        gt_img   = denorm_to_pil(gt_t)
        sbs_img  = make_side_by_side(cond_img, gt_img)

        name     = f"sample_{i:04d}_idx{idx:04d}"
        base_out = os.path.join(args.output_dir, name)

        cond_img.save(f"{base_out}_input.png")
        gt_img.save(  f"{base_out}_gt.png")
        sbs_img.save( f"{base_out}_side_by_side.png")

        rel_low  = os.path.relpath(low_paths[idx],  args.train_data_root)
        rel_norm = os.path.relpath(norm_paths[idx], args.train_data_root)
        print(f"  [{i+1:3d}/{num_save}] idx={idx:4d}  "
              f"shape={cond_t.shape[-2:]}  "
              f"input=[{cond_t.min():.2f}, {cond_t.max():.2f}]  "
              f"gt=[{gt_t.min():.2f}, {gt_t.max():.2f}]")
        print(f"             low : {rel_low}")
        print(f"             norm: {rel_norm}")

    print("-" * 72)
    print(f"\n[Done] {num_save} pairs saved to: {os.path.abspath(args.output_dir)}")

    print(f"\n=== Tensor Statistics (normalized [-1, 1]) ===")
    print(f"  INPUT  (low-light):  "
          f"mean={np.mean(stats_in['mean']):.4f}  "
          f"global_range=[{min(stats_in['min']):.3f}, {max(stats_in['max']):.3f}]")
    print(f"  GT     (norm-light):  "
          f"mean={np.mean(stats_gt['mean']):.4f}  "
          f"global_range=[{min(stats_gt['min']):.3f}, {max(stats_gt['max']):.3f}]")

    # Write info.txt
    info_path = os.path.join(args.output_dir, "info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("SDE Indoor Dataset Visualization Info\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"train_data_root : {args.train_data_root}\n")
        f.write(f"low_subdir     : {args.low_subdir}\n")
        f.write(f"normal_subdir  : {args.normal_subdir}\n")
        f.write(f"image_size     : {args.image_size}\n")
        f.write(f"split          : {args.split}\n")
        f.write(f"num_samples    : {num_save}  (total: {len(dataset)})\n\n")
        f.write("LowLightDataset normalization: mean=[0.5], std=[0.5]  =>  [-1, 1]\n")
        f.write("  raw [0,255]  --ToTensor-->  [0,1]  --normalize-->  [-1,1]  (model input)\n")
        f.write("  [-1,1]  --denormalize-->  [0,255]  --save-->  PNG\n\n")
        f.write("Image pairs (from LowLightDataset):\n")
        for i in range(num_save):
            idx = indices[i]
            f.write(f"  [{i:4d}] idx={idx:4d}  low={os.path.relpath(low_paths[idx],  args.train_data_root)}\n")
            f.write(f"          idx={idx:4d}  norm={os.path.relpath(norm_paths[idx], args.train_data_root)}\n")

    print(f"\nInfo file: {info_path}")


if __name__ == "__main__":
    main(parse_args())
