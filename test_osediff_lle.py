import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, required=True,
        help='Path to input low-light image or directory of images')
    parser.add_argument('--output_dir', '-o', type=str, default='output_lle/test',
        help='Directory to save the output images')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None,
        help='Path to pretrained Stable Diffusion 2.1 model')
    parser.add_argument('--osediff_path', type=str, required=True,
        help='Path to trained OSEDiff-LLE checkpoint (.pkl)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'],
        default='adain', help='Color correction method')
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

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Initialize the model
    print(f"[INFO] Loading OSEDiff-LLE model from: {args.osediff_path}")
    model = OSEDiff_test(args)

    # Get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(os.path.join(args.input_image, "*.png")))
        if len(image_names) == 0:
            image_names = sorted(glob.glob(os.path.join(args.input_image, "*.jpg")))
        if len(image_names) == 0:
            image_names = sorted(glob.glob(os.path.join(args.input_image, "*.PNG")))
    else:
        image_names = [args.input_image]

    print(f"[INFO] Found {len(image_names)} image(s) to process")

    # Make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Get prompt
    prompt = get_validation_prompt(args)
    print(f"[INFO] Using prompt: {prompt}")

    for image_path in image_names:
        bname = os.path.basename(image_path)
        print(f"Processing: {bname}")

        # Load and preprocess
        input_image = Image.open(image_path).convert('RGB')
        original_size = input_image.size  # (width, height)

        # Preprocess: ensure divisible by 8
        input_image_processed = preprocess_image(input_image, args)
        crop_w, crop_h = input_image_processed.size

        # Convert to tensor and normalize to [-1, 1]
        lq = tensor_transforms(input_image_processed).unsqueeze(0) * 2 - 1

        # Inference
        with torch.no_grad():
            output_tensor = model(lq, prompt=prompt)

        # Convert back to PIL
        output_pil = transforms.ToPILImage()(
            output_tensor[0].cpu().float().clamp(-1, 1) * 0.5 + 0.5
        )

        # Color fix (optional)
        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image_processed)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image_processed)

        # Save output
        output_path = os.path.join(args.output_dir, bname)
        output_pil.save(output_path)
        print(f"  Saved: {output_path}")

    print(f"\n[Done] Processed {len(image_names)} image(s). Output: {os.path.abspath(args.output_dir)}")
