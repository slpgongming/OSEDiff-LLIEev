"""
OSEDiff + Event (Low-Light Enhancement) 推理脚本
在 SDE 室内测试集上逐帧处理，计算 PSNR / SSIM 指标。
"""
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
from dataloaders.sde_event_utils import load_npz_event, generate_voxel_grid


tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])


def get_validation_prompt(args):
    return getattr(args, 'fixed_prompt', 'high quality, bright, detailed photo')


def preprocess_image(input_image, args):
    """
    确保图像尺寸是 8 的倍数（VAE latent 空间要求）。
    同时支持 voxel grid resize 到相同尺寸。
    """
    input_image = input_image.convert('RGB')

    # 确保 H/8, W/8 为整数
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8

    if new_width != input_image.width or new_height != input_image.height:
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    return input_image


def resize_voxel_to_image(voxel, target_h, target_w):
    """
    将 voxel grid resize 到与目标图像相同的 HxW。
    voxel: [C, H, W] tensor
    target_h, target_w: 目标尺寸（图像尺寸）
    """
    C = voxel.shape[0]
    if voxel.shape[1] == target_h and voxel.shape[2] == target_w:
        return voxel

    # 使用双线性插值 resize（voxel grid 的每个通道独立 resize）
    voxel_batch = voxel.unsqueeze(0)  # [1, C, H, W]
    voxel_resized = torch.nn.functional.interpolate(
        voxel_batch,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False,
    )
    return voxel_resized.squeeze(0)  # [C, target_h, target_w]


def build_input_tensor(low_pil, event_npz_path, num_bins, is_split_event, frame_timestamps=None, frame_idx=None):
    """
    为单张低光图像构建模型输入 tensor。

    Args:
        low_pil: PIL Image (RGB)
        event_npz_path: 事件 npz 文件路径（逐帧文件或整序列文件）
        num_bins: voxel grid 通道数
        is_split_event: 是否为逐帧分割的事件文件
        frame_timestamps: 帧时间戳列表（整序列 npz 模式用）
        frame_idx: 当前帧索引（整序列 npz 模式用）
    Returns:
        input_tensor: [3+num_bins, H, W] tensor, normalized to [-1, 1]
    """
    # 1. 预处理图像
    low_pil = preprocess_image(low_pil, None)
    H, W = low_pil.height, low_pil.width

    # 2. 归一化低光图像到 [-1, 1]
    lq_tensor = transforms.ToTensor()(low_pil)  # [3, H, W], [0, 1]
    lq_tensor = lq_tensor * 2.0 - 1.0  # [0,1] → [-1,1]

    # 3. 加载并提取事件
    if is_split_event:
        events = load_npz_event(event_npz_path)
        event_window = events
    else:
        all_events = load_npz_event(event_npz_path)
        event_window = _extract_event_window(all_events, frame_timestamps, frame_idx)

    # 4. 生成 voxel grid（先按图像原始尺寸生成，再 resize 到裁剪后尺寸）
    # 为了避免处理复杂边界，直接在预处理后的尺寸上生成
    voxel_raw = generate_voxel_grid(
        event_window,
        num_bins=num_bins,
        H=H,
        W=W,
        normalize=True,
    )  # [num_bins, H, W]

    # 5. Concat RGB + Voxel
    input_tensor = torch.cat([lq_tensor, voxel_raw], dim=0)  # [3+num_bins, H, W]
    return input_tensor


def _extract_event_window(all_events, frame_timestamps, idx):
    """从整序列事件中提取当前帧对应的时间窗口"""
    if len(all_events) == 0:
        return np.empty((0, 4), dtype=np.float32)

    current_t = frame_timestamps[idx]

    if idx == 0:
        start_t = all_events[0, 0]
    else:
        start_t = float(frame_timestamps[idx - 1])

    if idx == len(frame_timestamps) - 1:
        end_t = all_events[-1, 0]
    else:
        end_t = float(frame_timestamps[idx + 1])

    mask = (all_events[:, 0] > start_t) & (all_events[:, 0] < end_t)
    return all_events[mask].copy()


def scan_scene_pairs_and_events(root_dir, num_bins):
    """
    扫描数据集目录，返回所有样本信息。

    数据结构：
        root_dir/
            i_0/
                low/
                    xxx.png
                    xxx.npz (逐帧事件)
                    ...
                    lowlight_event.npz (整序列事件，可能存在)
                normal/
                    xxx.png
                    ...

    Returns:
        List of dicts: {
            'scene_name': str,
            'low_path': str,
            'normal_path': str,
            'event_npz_path': str,
            'is_split_event': bool,
            'frame_timestamps': list (仅整序列 npz 模式),
            'frame_idx': int,
        }
    """
    samples = []
    if not os.path.isdir(root_dir):
        return samples

    for scene_name in sorted(os.listdir(root_dir)):
        scene_dir = os.path.join(root_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        low_dir = os.path.join(scene_dir, 'low')
        normal_dir = os.path.join(scene_dir, 'normal')
        if not (os.path.isdir(low_dir) and os.path.isdir(normal_dir)):
            continue

        # 获取 png 列表（排除 npz）
        low_png_list = sorted([f for f in os.listdir(low_dir) if f.endswith('.png')])
        normal_png_list = sorted([f for f in os.listdir(normal_dir) if f.endswith('.png')])

        if not low_png_list or len(low_png_list) != len(normal_png_list):
            print(f"[WARN] Scene {scene_name}: {len(low_png_list)} low vs "
                  f"{len(normal_png_list)} normal — skipped")
            continue

        # 帧时间戳列表
        frame_timestamps = [int(f.replace('.png', '')) for f in low_png_list]

        # 识别事件文件
        low_npz_all = [f for f in os.listdir(low_dir) if f.endswith('.npz')]
        frame_npz_map = {}
        for npz_f in low_npz_all:
            base = npz_f.replace('.npz', '')
            png_candidate = base + '.png'
            if png_candidate in low_png_list:
                frame_npz_map[base] = os.path.join(low_dir, npz_f)

        seq_npz_path = os.path.join(low_dir, 'lowlight_event.npz')
        use_seq_npz = not frame_npz_map and os.path.exists(seq_npz_path)

        for i, (low_fname, normal_fname) in enumerate(zip(low_png_list, normal_png_list)):
            low_base = low_fname.replace('.png', '')

            if frame_npz_map and low_base in frame_npz_map:
                ev_path = frame_npz_map[low_base]
                is_split = True
            elif use_seq_npz:
                ev_path = seq_npz_path
                is_split = False
            else:
                print(f"[WARN] No event file for {scene_name}/{low_fname}, skipping")
                continue

            samples.append({
                'scene_name': scene_name,
                'low_path': os.path.join(low_dir, low_fname),
                'normal_path': os.path.join(normal_dir, normal_fname),
                'event_npz_path': ev_path,
                'is_split_event': is_split,
                'frame_timestamps': frame_timestamps,
                'frame_idx': i,
            })

    return samples


def setup_logger(output_dir, phase='test'):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    log_file = os.path.join(output_dir, f'{phase}_{timestamp}.log')

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger('OSEDiffEventTest')
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
        help='Path to SDE dataset root (test/ subfolder with i_n/ scenes)')
    parser.add_argument('--output_dir', '-o', type=str, default='output_lle_event/test',
        help='Directory to save output images')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None)
    parser.add_argument('--osediff_path', type=str, required=True,
        help='Path to trained OSEDiff-LLE-Event checkpoint (.pkl)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'],
        default='nofix', help='Color correction method')
    parser.add_argument('--fixed_prompt', type=str,
        default='high quality, bright, detailed photo')
    parser.add_argument('--device', type=str, default='cuda')

    # Precision
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")

    # Merge LoRA
    parser.add_argument("--merge_and_unload_lora", default=False,
        help="Merge LoRA weights before inference")

    # Tiling
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    # Timestep
    parser.add_argument("--inference_timestep", type=int, default=999,
        help="Timestep for single-step denoising during inference.")

    # Event / voxel
    parser.add_argument("--voxel_grid_channel", type=int, default=5,
        help="Number of channels (time bins) in voxel grid")
    parser.add_argument("--vae_in_channels", type=int, default=8,
        help="VAE encoder input channels: 3 RGB + voxel_grid_channel")

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Scan test samples
    test_samples = scan_scene_pairs_and_events(args.input_image, args.voxel_grid_channel)
    print(f"[INFO] Found {len(test_samples)} test samples in {args.input_image}")

    if len(test_samples) == 0:
        print("[ERROR] No valid samples found. Check dataset structure.")
        sys.exit(1)

    # Initialize the model
    print(f"[INFO] Loading OSEDiff-Event model from: {args.osediff_path}")
    model = OSEDiff_test(args)

    # Make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.output_dir, phase='test_metrics')
    logger.info("===== Configuration =====")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("==========================\n")

    # Initialize metrics
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)

    # Accumulators
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

    # Group by scene for sequential naming
    scene_to_samples = {}
    for sample in test_samples:
        scene_to_samples.setdefault(sample['scene_name'], []).append(sample)

    for scene_name in sorted(scene_to_samples.keys()):
        samples_in_scene = scene_to_samples[scene_name]
        scene_output_dir = os.path.join(args.output_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        for k, sample in enumerate(samples_in_scene, start=1):
            print(f"[{total_count+1}/{len(test_samples)}] Processing: {scene_name}/{k}")

            # 1. Load low-light image
            low_pil = Image.open(sample['low_path']).convert('RGB')

            # 2. Build input tensor (RGB + voxel concat)
            input_tensor = build_input_tensor(
                low_pil,
                event_npz_path=sample['event_npz_path'],
                num_bins=args.voxel_grid_channel,
                is_split_event=sample['is_split_event'],
                frame_timestamps=sample['frame_timestamps'] if not sample['is_split_event'] else None,
                frame_idx=sample['frame_idx'] if not sample['is_split_event'] else None,
            )  # [3+num_bins, H, W]
            lq = input_tensor.unsqueeze(0)  # [1, 3+num_bins, H, W]

            # 3. Load GT for metrics
            gt_pil = Image.open(sample['normal_path']).convert('RGB')

            # 4. Inference
            start_time = time.time()
            with torch.no_grad():
                output_tensor = model(lq, prompt=prompt)
            inference_time = time.time() - start_time

            # 5. Convert output to PIL
            output_pil = transforms.ToPILImage()(
                output_tensor[0].cpu().float().clamp(-1, 1) * 0.5 + 0.5
            )

            # 6. Color fix (optional)
            low_processed = preprocess_image(low_pil, args)
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=low_processed)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=low_processed)

            # 7. Save images
            prefix = f"{k:04d}"
            lowlight_path = os.path.join(scene_output_dir, f"{prefix}_lowlight.png")
            result_path = os.path.join(scene_output_dir, f"{prefix}_result.png")
            gt_path = os.path.join(scene_output_dir, f"{prefix}_gt.png")

            low_processed.save(lowlight_path)
            output_pil.save(result_path)
            gt_pil.save(gt_path)

            # 8. Compute metrics
            if gt_pil.size != output_pil.size:
                gt_resized = gt_pil.resize(output_pil.size, Image.LANCZOS)
            else:
                gt_resized = gt_pil

            sr_img = cv2.imread(result_path, cv2.IMREAD_COLOR)
            gt_img_cv = np.array(gt_resized)[:, :, ::-1]  # RGB -> BGR

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
