import argparse
import logging
import os
import time

import cv2
import numpy as np
import pyiqa
import torch
from basicsr.utils import img2tensor
from torchvision import transforms

from dataloaders.sde_event_dataset import SDEEventDataset
from models.event_latent_projector import EventPyramidLatentProjector
from models.hdrev_encoder import HDRevEncoder
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from osediff_hdrev import initialize_frozen_vae, tensor_stats


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("HDRevPyramidDirectDecode")
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


class HDRevPyramidDirectDecode(torch.nn.Module):
    """
    D-ablation inference model.

    This path deliberately bypasses OSEDiff UNet and DDPMScheduler:
      low RGB [0,1] + event voxel -> HDRevEncoder -> pyramid projector
      -> scaled clean latent [B,4,H/8,W/8] -> frozen SD VAE decoder -> image.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = (
            torch.float16 if args.mixed_precision == "fp16" and self.device.type == "cuda" else torch.float32
        )

        # Do not load original HDRev weights here. This ablation must evaluate
        # exactly the encoder/projector weights stored in the provided checkpoint.
        self.hdrev_encoder = HDRevEncoder(
            num_bins=args.num_bins,
            pretrained_root="",
            load_pretrained=False,
        )
        self.event_latent_projector = EventPyramidLatentProjector(latent_channels=4)
        self.vae = initialize_frozen_vae(args)

        checkpoint = torch.load(args.encoder_projector_path, map_location="cpu")
        self.load_ckpt(checkpoint, strict=not args.resume_ignore_mismatch)

        self.hdrev_encoder.to(self.device, dtype=self.weight_dtype).eval()
        self.event_latent_projector.to(self.device, dtype=self.weight_dtype).eval()
        self.vae.to(self.device, dtype=self.weight_dtype).eval()
        self.vae.requires_grad_(False)

        self.checkpoint_stage = checkpoint.get("stage", "unknown")
        self.checkpoint_step = checkpoint.get("global_step", None)

    def load_ckpt(self, checkpoint, strict=True):
        """Load HDRevEncoder and pyramid projector weights from a pretrain or Stage 3 checkpoint."""
        if "state_dict_hdrev_encoder" not in checkpoint:
            raise KeyError("Checkpoint is missing state_dict_hdrev_encoder.")
        if "state_dict_event_latent_projector" not in checkpoint:
            raise KeyError("Checkpoint is missing state_dict_event_latent_projector.")

        self.hdrev_encoder.load_state_dict(checkpoint["state_dict_hdrev_encoder"], strict=strict)
        self.event_latent_projector.load_state_dict(
            checkpoint["state_dict_event_latent_projector"],
            strict=strict,
        )

    @torch.no_grad()
    def forward(self, hdrev_rgb, event_voxel):
        hdrev_dtype = next(self.hdrev_encoder.parameters()).dtype
        projector_dtype = next(self.event_latent_projector.parameters()).dtype

        _, hdrev_features = self.hdrev_encoder(
            hdrev_rgb.to(device=self.device, dtype=hdrev_dtype),
            event_voxel.to(device=self.device, dtype=hdrev_dtype),
        )
        # The feature pyramid keeps the same scale convention used in Stage 3:
        # [B,64,H,W], [B,128,H/2,W/2], [B,256,H/4,W/4], [B,512,H/8,W/8].
        hdrev_features = [feature.to(dtype=projector_dtype) for feature in hdrev_features]
        z_pred = self.event_latent_projector(hdrev_features)

        vae_dtype = next(self.vae.parameters()).dtype
        output = self.vae.decode(
            z_pred.to(device=self.device, dtype=vae_dtype) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)
        return output, z_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_lle_hdrev_direct_decode")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--encoder_projector_path",
        type=str,
        required=True,
        help=(
            "Checkpoint containing state_dict_hdrev_encoder and "
            "state_dict_event_latent_projector. Clean-latent pretrain checkpoints "
            "and Stage 3 checkpoints are both supported."
        ),
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--event_norm", action="store_true", default=True)
    parser.add_argument("--no_event_norm", dest="event_norm", action="store_false")
    parser.add_argument("--fixed_prompt", type=str, default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="nofix")
    parser.add_argument("--resume_ignore_mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Reuse the Stage 3 test dataset logic so low/normal index pairing and
    # event voxelization are exactly the same as OSEDiff-HDRev inference.
    args.train_data_root = args.input_image
    args.test_data_root = args.input_image
    dataset = SDEEventDataset(split="test", args=args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = HDRevPyramidDirectDecode(args)
    logger = setup_logger(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device)
    ssim_metric = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr").to(device)

    scene_psnr, scene_ssim, scene_counts = {}, {}, {}
    scene_clip, scene_latent_std = {}, {}
    overall_psnr, overall_ssim, total_count = 0.0, 0.0, 0

    logger.info("===== HDRev pyramid direct-decode D-ablation config =====")
    logger.info("  checkpoint_stage: %s", model.checkpoint_stage)
    logger.info("  checkpoint_step: %s", model.checkpoint_step)
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key}: {value}")

    for batch in dataloader:
        scene = batch["scene"][0]
        sample_index = int(batch["sample_index"][0])
        scene_output_dir = os.path.join(args.output_dir, scene)
        os.makedirs(scene_output_dir, exist_ok=True)

        start = time.time()
        with torch.no_grad():
            output_tensor, z_pred = model(batch["hdrev_pixel_values"], batch["event_voxel"])
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
        result_path = os.path.join(scene_output_dir, f"{prefix}_direct_decode.png")
        gt_path = os.path.join(scene_output_dir, f"{prefix}_gt.png")
        low_pil.save(low_path)
        output_pil.save(result_path)
        gt_pil.save(gt_path)

        sr_img = cv2.imread(result_path, cv2.IMREAD_COLOR)
        gt_img_cv = np.array(gt_pil)[:, :, ::-1]
        sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0
        gt_tensor = img2tensor(gt_img_cv, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            psnr_val = psnr_metric(sr_tensor, gt_tensor).item()
            ssim_val = ssim_metric(sr_tensor, gt_tensor).item()

        clip_fraction = (output_tensor.detach().abs() > 0.999).float().mean().item()
        z_stats = tensor_stats(z_pred)

        scene_psnr[scene] = scene_psnr.get(scene, 0.0) + psnr_val
        scene_ssim[scene] = scene_ssim.get(scene, 0.0) + ssim_val
        scene_clip[scene] = scene_clip.get(scene, 0.0) + clip_fraction
        scene_latent_std[scene] = scene_latent_std.get(scene, 0.0) + z_stats["std"]
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        overall_psnr += psnr_val
        overall_ssim += ssim_val
        total_count += 1

        logger.info(
            "%s/%s PSNR=%.4f SSIM=%.4f z(mean/std)=%.4f/%.4f clip=%.6f time=%.2fs saved=%s",
            scene,
            prefix,
            psnr_val,
            ssim_val,
            z_stats["mean"],
            z_stats["std"],
            clip_fraction,
            elapsed,
            result_path,
        )

    logger.info("===== Per-scene average =====")
    for scene in sorted(scene_counts):
        count = scene_counts[scene]
        logger.info(
            "%s PSNR=%.4f SSIM=%.4f clip=%.6f z_std=%.4f n=%d",
            scene,
            scene_psnr[scene] / count,
            scene_ssim[scene] / count,
            scene_clip[scene] / count,
            scene_latent_std[scene] / count,
            count,
        )

    if total_count > 0:
        logger.info(
            "===== DirectDecode Overall PSNR=%.4f SSIM=%.4f n=%d =====",
            overall_psnr / total_count,
            overall_ssim / total_count,
            total_count,
        )


if __name__ == "__main__":
    main(parse_args())
