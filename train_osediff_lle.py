import os
import gc
import lpips
import clip
import argparse
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

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


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

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process)

    global_step = 0

    # --- Training loop ---
    for epoch in range(0, args.num_training_epochs):
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
                        accelerator.unwrap_model(model_gen).save_model(outf)
                        print(f"\n[Saved] {outf}")

                    accelerator.log(logs, step=global_step)

            # Safety: prevent infinite loop in debug
            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # --- Final save ---
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoints", f"model_final.pkl")
        accelerator.unwrap_model(model_gen).save_model(final_path)
        print(f"\n[Final model saved] {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
