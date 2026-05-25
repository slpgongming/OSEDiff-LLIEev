import argparse
import logging
import os
import time

import cv2
import numpy as np
import pyiqa
import torch
from basicsr.utils import img2tensor
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.sde_event_dataset import SDEEventDataset
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from osediff_hdrev import OSEDiffHDRevTest


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("OSEDiffHDRevTest")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(os.path.join(output_dir, "test_metrics.log"), mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def tensor_to_pil(tensor):
    """Convert one image tensor in [-1, 1] to a PIL RGB image."""
    tensor = tensor.detach().cpu().float().clamp(-1, 1) * 0.5 + 0.5
    return transforms.ToPILImage()(tensor)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_lle_hdrev_stage3")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--osediff_path", type=str, required=True)
    parser.add_argument("--hdrev_pretrained_root", type=str, default="checkpoints/HDRev")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--event_norm", action="store_true", default=True)
    parser.add_argument("--no_event_norm", dest="event_norm", action="store_false")
    parser.add_argument("--fixed_prompt", type=str, default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--inference_timestep", type=int, default=999)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="nofix")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # SDEEventDataset expects train_data_root/test_data_root names; for inference
    # both point to the user-provided test root.
    args.train_data_root = args.input_image
    args.test_data_root = args.input_image
    dataset = SDEEventDataset(split="test", args=args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = OSEDiffHDRevTest(args)
    logger = setup_logger(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device)
    ssim_metric = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device)

    scene_psnr, scene_ssim, scene_counts = {}, {}, {}
    overall_psnr, overall_ssim, total_count = 0.0, 0.0, 0

    logger.info("===== Stage 3 HDRev inference config =====")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key}: {value}")

    for batch in dataloader:
        scene = batch["scene"][0]
        sample_index = int(batch["sample_index"][0])
        scene_output_dir = os.path.join(args.output_dir, scene)
        os.makedirs(scene_output_dir, exist_ok=True)

        start = time.time()
        with torch.no_grad():
            output_tensor = model(
                batch["hdrev_pixel_values"],
                batch["event_voxel"],
                prompt=args.fixed_prompt,
            )
        elapsed = time.time() - start

        low_pil = tensor_to_pil(batch["conditioning_pixel_values"][0])
        gt_pil = tensor_to_pil(batch["output_pixel_values"][0])
        output_pil = tensor_to_pil(output_tensor[0])

        if args.align_method == "adain":
            output_pil = adain_color_fix(target=output_pil, source=low_pil)
        elif args.align_method == "wavelet":
            output_pil = wavelet_color_fix(target=output_pil, source=low_pil)

        prefix = f"{sample_index + 1:04d}"
        low_path = os.path.join(scene_output_dir, f"{prefix}_lowlight.png")
        result_path = os.path.join(scene_output_dir, f"{prefix}_result.png")
        gt_path = os.path.join(scene_output_dir, f"{prefix}_gt.png")
        low_pil.save(low_path)
        output_pil.save(result_path)
        gt_pil.save(gt_path)

        # Metrics are computed from saved/result PIL images to include optional
        # color correction in the reported numbers.
        sr_img = cv2.imread(result_path, cv2.IMREAD_COLOR)
        gt_img_cv = np.array(gt_pil)[:, :, ::-1]
        sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0
        gt_tensor = img2tensor(gt_img_cv, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            psnr_val = psnr_metric(sr_tensor, gt_tensor).item()
            ssim_val = ssim_metric(sr_tensor, gt_tensor).item()

        scene_psnr[scene] = scene_psnr.get(scene, 0.0) + psnr_val
        scene_ssim[scene] = scene_ssim.get(scene, 0.0) + ssim_val
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        overall_psnr += psnr_val
        overall_ssim += ssim_val
        total_count += 1

        logger.info(
            "%s/%s PSNR=%.4f SSIM=%.4f time=%.2fs saved=%s",
            scene,
            prefix,
            psnr_val,
            ssim_val,
            elapsed,
            result_path,
        )

    logger.info("===== Per-scene average =====")
    for scene in sorted(scene_counts):
        count = scene_counts[scene]
        logger.info(
            "%s PSNR=%.4f SSIM=%.4f n=%d",
            scene,
            scene_psnr[scene] / count,
            scene_ssim[scene] / count,
            count,
        )

    if total_count > 0:
        logger.info(
            "===== Overall PSNR=%.4f SSIM=%.4f n=%d =====",
            overall_psnr / total_count,
            overall_ssim / total_count,
            total_count,
        )


if __name__ == "__main__":
    main(parse_args())
