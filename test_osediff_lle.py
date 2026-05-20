import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import logging
import time
from datetime import datetime
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

import pyiqa
from basicsr.utils import img2tensor

from osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])


def get_validation_prompt(args):
    """
    Returns the fixed prompt for low-light enhancement.
    No RAM/caption model needed.
    """
    return args.fixed_prompt


def preprocess_image(input_image, args):
    """
    Preprocess input image for low-light enhancement inference.

    For LLE: no upscaling needed (input and output are same resolution).
    Only ensures H/8 and W/8 are integers (divisible by 8 for VAE).

    Args:
        input_image: PIL Image
        args: argparse namespace

    Returns:
        PIL Image ready for model input
    """
    input_image = input_image.convert('RGB')

    # Ensure dimensions are divisible by 8 (required for VAE latent space)
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8

    if new_width != input_image.width or new_height != input_image.height:
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    return input_image


def scan_scene_pairs(root_dir):
    """
    Scans the dataset directory structure and returns low/normal image pairs.

    Expected structure (same as SDEIndoorDataset):
        root_dir/
            scene_id_1/
                low/
                    1690097848450715.png
                    1690097848529056.png
                normal/
                    1690097848450715.png   <- same filename
                    1690097848529056.png   <- same filename
            scene_id_2/
                ...

    Pairing: sort both lists, pair by index (zip), exclude event .npz files.
    Returns:
        List of (low_path, normal_path, scene_name) tuples.
    """
    pairs = []
    if not os.path.isdir(root_dir):
        return pairs

    # Find all scene directories
    for scene_name in sorted(os.listdir(root_dir)):
        scene_dir = os.path.join(root_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        low_dir = os.path.join(scene_dir, 'low')
        normal_dir = os.path.join(scene_dir, 'normal')

        if not (os.path.isdir(low_dir) and os.path.isdir(normal_dir)):
            continue

        # Get all PNG files in low/ (exclude event npz), sorted
        low_files = sorted([
            f for f in os.listdir(low_dir)
            if f.endswith('.png') and not f.endswith('.npz')
        ])
        # Get all PNG files in normal/, sorted
        normal_files = sorted([
            f for f in os.listdir(normal_dir)
            if f.endswith('.png')
        ])

        # Pair by index: low_files[i] <-> normal_files[i] (same as SDEIndoorDataset)
        min_len = min(len(low_files), len(normal_files))
        for i in range(min_len):
            low_path = os.path.join(low_dir, low_files[i])
            normal_path = os.path.join(normal_dir, normal_files[i])
            pairs.append((low_path, normal_path, scene_name))

        if len(low_files) == 0:
            continue

        # Warn if lengths differ
        if len(low_files) != len(normal_files):
            print(f"[WARN] Scene {scene_name}: {len(low_files)} low vs "
                  f"{len(normal_files)} normal frames — paired {min_len}")

    return pairs


def setup_logger(output_dir, phase='test'):
    """Set up logger for metric output."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    log_file = os.path.join(output_dir, f'{phase}_{timestamp}.log')

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger('OSEDiffLLE')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, required=True,
        help='Path to the dataset root directory (with scene/low and scene/normal subdirs)')
    parser.add_argument('--output_dir', '-o', type=str, default='output_lle/test',
        help='Directory to save the output images')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None,
        help='Path to pretrained Stable Diffusion 2.1 model')
    parser.add_argument('--osediff_path', type=str, required=True,
        help='Path to trained OSEDiff-LLE checkpoint (.pkl)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'],
        default='nofix', help='Color correction method')
    parser.add_argument('--fixed_prompt', type=str,
        default='high quality, bright, detailed photo',
        help='Fixed text prompt for generation')
    parser.add_argument('--device', type=str, default='cuda',
        help='Device to run inference on')

    # Precision
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")

    # Merge LoRA
    parser.add_argument("--merge_and_unload_lora", default=False,
        help="Merge LoRA weights before inference (slower but no need for LoRA runtime)")

    # Tiling
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    # Timestep (must match training max_train_timestep)
    parser.add_argument("--inference_timestep", type=int, default=999,
        help="Timestep for single-step denoising during inference. "
             "Should match the max_train_timestep used during training. "
             "E.g. 999 for full-range, 200 for small-timestep training.")

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Initialize the model
    print(f"[INFO] Loading OSEDiff-LLE model from: {args.osediff_path}")
    model = OSEDiff_test(args)

    # Scan for low/normal image pairs across all scenes
    scene_pairs = scan_scene_pairs(args.input_image)
    print(f"[INFO] Found {len(scene_pairs)} image pair(s) across all scenes")

    if len(scene_pairs) == 0:
        print("[ERROR] No valid low/normal image pairs found. Check dataset structure.")
        sys.exit(1)

    # Make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logger for metrics
    logger = setup_logger(args.output_dir, phase='test_metrics')
    logger.info("===== Configuration =====")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("==========================\n")

    # Initialize metrics
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)

    # Accumulators for per-scene and overall metrics
    scene_psnr = {}
    scene_ssim = {}
    scene_counts = {}
    overall_psnr = 0.0
    overall_ssim = 0.0
    total_count = 0

    # Get prompt
    prompt = get_validation_prompt(args)
    print(f"[INFO] Using prompt: {prompt}")
    logger.info(f"[INFO] Using prompt: {prompt}")

    # Group pairs by scene for sequential naming
    scene_to_pairs = {}
    for low_path, normal_path, scene_name in scene_pairs:
        scene_to_pairs.setdefault(scene_name, []).append((low_path, normal_path))

    for scene_name in sorted(scene_to_pairs.keys()):
        pairs_in_scene = scene_to_pairs[scene_name]
        scene_output_dir = os.path.join(args.output_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        for k, (low_path, normal_path) in enumerate(pairs_in_scene, start=1):
            print(f"[{total_count+1}/{len(scene_pairs)}] Processing: {scene_name}/{k}")

            # Load and preprocess input (low-light)
            input_image = Image.open(low_path).convert('RGB')
            input_image_processed = preprocess_image(input_image, args)

            # Load GT (normal) for metrics later
            gt_image = Image.open(normal_path).convert('RGB')
            gt_image_processed = preprocess_image(gt_image, args)

            # Convert to tensor and normalize to [-1, 1]
            lq = tensor_transforms(input_image_processed).unsqueeze(0) * 2 - 1

            # Inference
            start_time = time.time()
            with torch.no_grad():
                output_tensor = model(lq, prompt=prompt)
            inference_time = time.time() - start_time

            # Convert back to PIL
            output_pil = transforms.ToPILImage()(
                output_tensor[0].cpu().float().clamp(-1, 1) * 0.5 + 0.5
            )

            # Color fix (optional)
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image_processed)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=input_image_processed)

            # Save input, result, GT with sequential names
            prefix = f"{k:04d}"
            lowlight_path = os.path.join(scene_output_dir, f"{prefix}_lowlight.png")
            result_path = os.path.join(scene_output_dir, f"{prefix}_result.png")
            gt_path = os.path.join(scene_output_dir, f"{prefix}_gt.png")

            input_image_processed.save(lowlight_path)
            output_pil.save(result_path)
            gt_image_processed.save(gt_path)

            # Compute PSNR and SSIM
            if gt_image_processed.size != output_pil.size:
                gt_resized = gt_image_processed.resize(output_pil.size, Image.LANCZOS)
            else:
                gt_resized = gt_image_processed

            sr_img = cv2.imread(result_path, cv2.IMREAD_COLOR)
            gt_img_cv = np.array(gt_resized)[:, :, ::-1]  # RGB -> BGR for pyiqa

            sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
            gt_tensor = img2tensor(gt_img_cv, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

            with torch.no_grad():
                psnr_val = psnr_metric(sr_tensor, gt_tensor).item()
                ssim_val = ssim_metric(sr_tensor, gt_tensor).item()

            # Accumulate per-scene
            if scene_name not in scene_psnr:
                scene_psnr[scene_name] = 0.0
                scene_ssim[scene_name] = 0.0
                scene_counts[scene_name] = 0
            scene_psnr[scene_name] += psnr_val
            scene_ssim[scene_name] += ssim_val
            scene_counts[scene_name] += 1

            # Accumulate overall
            overall_psnr += psnr_val
            overall_ssim += ssim_val
            total_count += 1

            logger.info(f"{scene_name}/{prefix} | PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f} | Time: {inference_time:.2f}s")
            print(f"  PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f} | Saved: {lowlight_path}, {result_path}, {gt_path}")

    # Log per-scene averages
    logger.info("\n===== Per-Scene Average Metrics =====")
    for scene_name in sorted(scene_psnr.keys()):
        cnt = scene_counts[scene_name]
        avg_psnr = scene_psnr[scene_name] / cnt
        avg_ssim = scene_ssim[scene_name] / cnt
        logger.info(f"  {scene_name}: PSNR={avg_psnr:.4f} | SSIM={avg_ssim:.4f} (n={cnt})")

    # Log overall average
    avg_psnr_all = overall_psnr / total_count
    avg_ssim_all = overall_ssim / total_count
    logger.info(f"\n===== Overall Average Metrics =====")
    logger.info(f"  PSNR: {avg_psnr_all:.4f} | SSIM: {avg_ssim_all:.4f} (n={total_count})")

    print(f"\n[Done] Processed {total_count} image(s).")
    print(f"  Overall PSNR: {avg_psnr_all:.4f} | SSIM: {avg_ssim_all:.4f}")
    print(f"  Output: {os.path.abspath(args.output_dir)}")
    print(f"  Log: {os.path.abspath(args.output_dir)}")
