import os
import gc
import lpips
import clip
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from osediff import OSEDiff_reg, OSEDiff_gen
from dataloaders.lle_dataset import LowLightDataset

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")


def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")


def parse_str_list(arg):
    return arg.split(',')


def parse_train_timestep_range(arg):
    """
    Parse the --train_timestep_range argument into a list of (low, high, weight) tuples.

    Supported formats:
      - Single value:          "100"       → [(0, 100, 1.0)]
      - Simple range:          "0,200"     → [(0, 200, 1.0)]
      - Weighted ranges:        "0,0.5;100,0.3;200,0.2"
                                 → [(0, 100, 0.5), (100, 200, 0.2), (200, 200, 0.3)]
      - Weighted ranges (short): "0,0.5;200,0.5"  → [(0, 200, 0.5), (200, 200, 0.5)]
      - Mixed:                  "0,999"     → [(0, 999, 1.0)]  (full range)

    Returns a list of (low, high, weight) tuples where low/high are ints and weight is float.
    """
    arg = arg.strip()
    if ';' in arg:
        parts = arg.split(';')
        ranges = []
        total_weight = 0.0
        for p in parts:
            p = p.strip()
            if ',' not in p:
                raise argparse.ArgumentTypeError(
                    f"Weighted range '{p}' must contain a comma (e.g. '0,0.5')."
                )
            val_str, weight_str = p.rsplit(',', 1)
            val = int(val_str.strip())
            weight = float(weight_str.strip())
            if weight < 0:
                raise argparse.ArgumentTypeError(f"Weight must be non-negative, got {weight}.")
            low = 0 if val == 0 else val
            high = val
            ranges.append((low, high, weight))
            total_weight += weight
        if total_weight <= 0:
            raise argparse.ArgumentTypeError("Total weight must be positive.")
        return ranges
    else:
        vals = [v.strip() for v in arg.split(',')]
        if len(vals) == 1:
            high = int(vals[0])
            return [(0, high, 1.0)]
        elif len(vals) == 2:
            low = int(vals[0])
            high = int(vals[1])
            return [(low, high, 1.0)]
        else:
            raise argparse.ArgumentTypeError(
                f"Range format expects at most 2 values (low,high), got {len(vals)}: '{arg}'."
            )


def parse_args(input_args=None):
    """
    Parses command-line arguments for low-light enhancement training.
    Based on train_osediff.py with modifications for LLE task.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # --- Paths ---
    parser.add_argument("--train_data_root", type=str, required=True,
        help="Root directory of the SDE dataset (contains i_n/ subfolder)")
    parser.add_argument("--low_light_subdir", type=str, default="i_n/low",
        help="Subdirectory name for low-light images, relative to train_data_root")
    parser.add_argument("--normal_subdir", type=str, default="i_n/normal",
        help="Subdirectory name for normal-light images, relative to train_data_root")

    # --- Training details ---
    parser.add_argument("--output_dir", default='experience/osediff_lle')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=512,
        help="Crop size for training images (HxW = image_size x image_size)")
    parser.add_argument("--train_batch_size", type=int, default=4,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help='Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help='Supported: "tensorboard", "wandb", "comet_ml", "all"')
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--tracker_project_name", type=str, default="train_osediff_lle")

    # --- Model ---
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)

    # --- Loss weights ---
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)

    # --- Text prompts (fixed, no RAM model) ---
    parser.add_argument("--fixed_prompt", default="high quality, bright, detailed photo", type=str,
        help="Fixed positive prompt used for all training samples")
    parser.add_argument("--neg_prompt", default=(
        "underexposed, dark, low-light, noisy, blurry, "
        "painting, oil painting, illustration, drawing, art, sketch, cartoon, "
        "CG Style, 3D render, worst quality, low quality, frames, watermark, signature, "
        "jpeg artifacts, deformed, lowres, over-smooth"
    ), type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)

    # --- LoRA ---
    parser.add_argument("--lora_rank", default=4, type=int)

    # --- Timestep control ---
    parser.add_argument("--train_timestep_range", default="0,200", type=parse_train_timestep_range,
        help="Timestep range(s) for training sampling. Three modes are supported:\n"
             "  1. Single value:   '100'          → sample uniformly from [0, 100]\n"
             "  2. Simple range:   '0,200'         → sample uniformly from [0, 200]\n"
             "  3. Weighted ranges: '0,0.5;100,0.3;200,0.2'\n"
             "                       → first pick a bucket by weight, then sample uniformly inside it.\n"
             "  Note: 999 is the scheduler's max timestep (alpha→0 regime). Using smaller values\n"
             "         avoids the numerically unstable region near alpha=0.\n"
             "Examples:\n"
             "  --train_timestep_range '100'           (fixed range [0,100])\n"
             "  --train_timestep_range '0,200'         (uniform range [0,200])\n"
             "  --train_timestep_range '0,0.5;200,0.5' (50%% weight for [0,200], 50%% for t=200)")

    # --- Resume training ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
        help="Path to a checkpoint to resume training from. "
             "If 'latest', will use the last checkpoint in output_dir/checkpoints/")
    parser.add_argument("--resume_ignore_mismatch", action="store_true",
        help="Ignore mismatched parameters when resuming (useful when changing batch size etc.)")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_") and f.endswith(".pkl")]
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.replace("model_", "").replace(".pkl", "")))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # --- File logging (always write to output_dir/train.log) ---
    log_file = os.path.join(args.output_dir, "train.log")
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler (root logger + tqdm output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # --- Determine checkpoint path ---
    resume_checkpoint_path = None
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint == "latest":
            resume_checkpoint_path = find_latest_checkpoint(
                os.path.join(args.output_dir, "checkpoints")
            )
            if resume_checkpoint_path is None:
                if accelerator.is_main_process:
                    logging.warning("[WARNING] No checkpoint found. Starting from scratch.")
        else:
            resume_checkpoint_path = args.resume_from_checkpoint

    # --- Initialize models ---
    model_gen = OSEDiff_gen(args)
    model_gen.set_train()

    model_reg = OSEDiff_reg(args=args, accelerator=accelerator)
    model_reg.set_train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # Set LoRA adapters
    model_gen.vae.set_adapter(['default_encoder'])
    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
            model_reg.unet_fix.enable_xformers_memory_efficient_attention()
            model_reg.unet_update.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        model_reg.unet_fix.enable_gradient_checkpointing()
        model_reg.unet_update.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer for generator ---
    layers_to_opt = []
    for n, _p in model_gen.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    for n, _p in model_gen.vae.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    # --- Optimizer for regression model ---
    layers_to_opt_reg = []
    for n, _p in model_reg.unet_update.named_parameters():
        if "lora" in n:
            layers_to_opt_reg.append(_p)
    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler_reg = get_scheduler(args.lr_scheduler, optimizer=optimizer_reg,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    # --- Data loaders ---
    # NOTE: LowLightDataset uses train_data_root + low/normal subdirs directly.
    # No text file list or degradation pipeline needed.
    dataset_train = LowLightDataset(split="train", args=args)
    dataset_val = LowLightDataset(split="test", args=args)

    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prepare with Accelerator ---
    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
    )
    net_lpips = accelerator.prepare(net_lpips)

    # Weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # --- Trackers ---
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
        logging.info("=" * 60)
        logging.info("Training config:")
        for k, v in sorted(vars(args).items()):
            logging.info(f"  {k}: {v}")
        logging.info(f"train_timestep_range: {args.train_timestep_range}")
        logging.info("=" * 60)

    # --- Load checkpoint if resuming ---
    start_epoch = 0
    if resume_checkpoint_path is not None:
        if accelerator.is_main_process:
            logging.info(f"[INFO] Loading checkpoint from: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")

        # Load model_gen weights
        accelerator.unwrap_model(model_gen).load_ckpt(checkpoint)

        # Load model_reg weights
        # Get reg state dict from additional_info if available
        reg_checkpoint = checkpoint.copy()
        if "additional_info" in checkpoint and "reg_checkpoint" in checkpoint["additional_info"]:
            reg_checkpoint.update(checkpoint["additional_info"]["reg_checkpoint"])
        accelerator.unwrap_model(model_reg).load_ckpt(reg_checkpoint)

        # Load optimizer and scheduler states
        if "optimizer_state_dict" in checkpoint and not args.resume_ignore_mismatch:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    logging.info("[INFO] Optimizer state loaded.")
                except Exception as e:
                    logging.warning(f"[WARNING] Failed to load optimizer state: {e}")

        if "lr_scheduler_state_dict" in checkpoint and not args.resume_ignore_mismatch:
                try:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                    logging.info("[INFO] LR Scheduler state loaded.")
                except Exception as e:
                    logging.warning(f"[WARNING] Failed to load scheduler state: {e}")

        if "lr_scheduler_reg_state_dict" in checkpoint and not args.resume_ignore_mismatch:
                try:
                    lr_scheduler_reg.load_state_dict(checkpoint["lr_scheduler_reg_state_dict"])
                    logging.info("[INFO] Reg LR Scheduler state loaded.")
                except Exception as e:
                    logging.warning(f"[WARNING] Failed to load reg scheduler state: {e}")

        if "optimizer_reg_state_dict" in checkpoint and not args.resume_ignore_mismatch:
                try:
                    optimizer_reg.load_state_dict(checkpoint["optimizer_reg_state_dict"])
                    logging.info("[INFO] Reg Optimizer state loaded.")
                except Exception as e:
                    logging.warning(f"[WARNING] Failed to load reg optimizer state: {e}")

        # Get starting step and epoch
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)

        if accelerator.is_main_process:
            logging.info(f"[INFO] Resuming from epoch {start_epoch}, step {global_step}")

    # Initialize step tracking for non-resume case
    if 'global_step' not in dir():
        global_step = 0
    if 'start_epoch' not in dir():
        start_epoch = 0

    # Initialize progress bar AFTER checkpoint loading to show correct step
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process)

    # --- Training loop ---
    epoch_loss_sum = 0.0
    epoch_loss_count = 0
    for epoch in range(start_epoch, args.num_training_epochs):
        logging.info(f"--- Epoch {epoch} started ---")
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen, model_reg]
            with accelerator.accumulate(*m_acc):

                x_src = batch["conditioning_pixel_values"]   # low-light input
                x_tgt = batch["output_pixel_values"]           # normal-light GT

                # Fixed prompt for all samples (no RAM model needed)
                batch["prompt"] = [args.fixed_prompt] * x_src.shape[0]

                # --- Generator forward pass ---
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = model_gen(
                    x_src, batch=batch, args=args
                )

                # --- Reconstruction losses ---
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                # --- KL / VSD loss ---
                if torch.cuda.device_count() > 1:
                    loss_kl = model_reg.module.distribution_matching_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        neg_prompt_embeds=neg_prompt_embeds,
                        args=args
                    ) * args.lambda_vsd
                    loss_d = model_reg.module.diff_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        args=args
                    ) * args.lambda_vsd_lora
                else:
                    loss_kl = model_reg.distribution_matching_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        neg_prompt_embeds=neg_prompt_embeds,
                        args=args
                    ) * args.lambda_vsd
                    loss_d = model_reg.diff_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        args=args
                    ) * args.lambda_vsd_lora

                loss = loss + loss_kl

                # --- Generator backward ---
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # --- Regression model backward ---
                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            # --- Logging & checkpointing ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_d": loss_d.detach().item(),
                        "loss_kl": loss_kl.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        # Save reg state dict first
                        reg_checkpoint = {
                            "state_dict_reg": {
                                k: v for k, v in accelerator.unwrap_model(model_reg).unet_update.state_dict().items() if "lora" in k
                            }
                        }
                        accelerator.unwrap_model(model_gen).save_model(
                            outf,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            lr_scheduler_reg=lr_scheduler_reg,
                            optimizer_reg=optimizer_reg,
                            global_step=global_step,
                            epoch=epoch,
                            additional_info={"reg_checkpoint": reg_checkpoint}
                        )
                        logging.info(f"[Checkpoint] Step={global_step} | loss={loss.item():.6f} | Saved: {outf}")

                    accelerator.log(logs, step=global_step)

                    # Write step-level loss to log file every 10 steps
                    if global_step % 10 == 0:
                        logging.info(
                            f"step={global_step} | loss={loss.item():.6f} | "
                            f"loss_l2={loss_l2.item():.6f} | loss_lpips={loss_lpips.item():.6f} | "
                            f"loss_kl={loss_kl.item():.6f} | loss_d={loss_d.item():.6f} | "
                            f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                        )
                    epoch_loss_sum += loss.item()
                    epoch_loss_count += 1

            # Safety: prevent infinite loop in debug
            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

        avg_epoch_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0
        logging.info(f"--- Epoch {epoch} finished, steps so far: {global_step}, avg_loss={avg_epoch_loss:.6f} ---")

    # --- Final save ---
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoints", f"model_final.pkl")
        # Save reg state dict first
        reg_checkpoint = {
            "state_dict_reg": {
                k: v for k, v in accelerator.unwrap_model(model_reg).unet_update.state_dict().items() if "lora" in k
            }
        }
        accelerator.unwrap_model(model_gen).save_model(
            final_path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_reg=lr_scheduler_reg,
            optimizer_reg=optimizer_reg,
            global_step=global_step,
            epoch=epoch,
            additional_info={"reg_checkpoint": reg_checkpoint}
        )
        logging.info(f"[Final] Training complete. Model saved: {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
