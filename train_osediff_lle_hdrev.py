import argparse
import logging
import os
from pathlib import Path

# Some managed/HPC environments ship an incomplete CA certificate chain.
# Set these at process startup, before Accelerate can initialize wandb, so all
# torchrun ranks inherit the same SSL behavior. WANDB_INSECURE is kept for
# compatibility with older W&B versions; WANDB_INSECURE_DISABLE_SSL matches the
# current W&B Settings field `insecure_disable_ssl`.
os.environ["WANDB_INSECURE"] = "true"
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"

import diffusers
import lpips
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm

from dataloaders.sde_event_dataset import SDEEventDataset
from osediff import OSEDiff_reg
from osediff_hdrev import OSEDiffHDRevGen, tensor_stats


def tensor_is_finite(tensor):
    """Return 1.0 when all tensor values are finite, else 0.0 for tracker logging."""
    if tensor is None:
        return 1.0
    return float(torch.isfinite(tensor.detach()).all().item())


def parse_train_timestep_range(arg):
    """Parse fixed/range/weighted timestep options used by the OSEDiff one-step scheduler."""
    arg = arg.strip()
    if ";" in arg:
        ranges = []
        total_weight = 0.0
        for part in arg.split(";"):
            val_str, weight_str = part.rsplit(",", 1)
            val = int(val_str.strip())
            weight = float(weight_str.strip())
            low = 0 if val == 0 else val
            ranges.append((low, val, weight))
            total_weight += weight
        if total_weight <= 0:
            raise argparse.ArgumentTypeError("Total timestep weight must be positive.")
        return ranges

    vals = [v.strip() for v in arg.split(",")]
    if len(vals) == 1:
        val = int(vals[0])
        return [(val, val, 1.0)]
    if len(vals) == 2:
        return [(int(vals[0]), int(vals[1]), 1.0)]
    raise argparse.ArgumentTypeError(f"Invalid timestep range: {arg}")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--train_data_root", type=str, required=True)
    parser.add_argument("--test_data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experience/osediff_lle_hdrev_stage3")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--hdrev_pretrained_root",
        type=str,
        default="checkpoints/HDRev",
        help=(
            "Original HDRev checkpoint root. Used only when "
            "--encoder_projector_pretrain_path is not provided."
        ),
    )
    parser.add_argument("--hdrev_encoder_i_path", type=str, default=None)
    parser.add_argument("--hdrev_encoder_e_path", type=str, default=None)
    parser.add_argument("--hdrev_merge_path", type=str, default=None)
    parser.add_argument("--hdrev_decoder_path", type=str, default=None)
    parser.add_argument(
        "--encoder_projector_pretrain_path",
        type=str,
        default=None,
        help=(
            "Optional clean-latent pretrain checkpoint used to initialize "
            "HDRevEncoder and EventPyramidLatentProjector before Stage 3 training."
        ),
    )

    # Data
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--event_norm", action="store_true", default=True)
    parser.add_argument("--no_event_norm", dest="event_norm", action="store_false")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # Optimization
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--set_grads_to_none", action="store_true")

    # Losses
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=2.0)
    parser.add_argument("--lambda_vsd", type=float, default=1.0)
    parser.add_argument("--lambda_vsd_lora", type=float, default=1.0)
    parser.add_argument("--lambda_latent_align", type=float, default=0.05)
    parser.add_argument("--cfg_vsd", type=float, default=7.5)
    parser.add_argument("--train_timestep_range", default="999", type=parse_train_timestep_range)

    # Prompts / LoRA
    parser.add_argument("--fixed_prompt", type=str, default="high quality, bright, detailed photo")
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default=(
            "underexposed, dark, low-light, noisy, blurry, painting, oil painting, "
            "illustration, drawing, art, sketch, cartoon, CG Style, 3D render, "
            "worst quality, low quality, frames, watermark, signature, jpeg artifacts, "
            "deformed, lowres, over-smooth"
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    # Logging / resume
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, default="train_osediff_lle_hdrev")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name. Defaults to tracker_project_name when report_to uses wandb.",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Optional W&B entity/team.")
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags, e.g. stage3,hdrev,pretrain-init.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode. Use offline when the server has no internet during training.",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_ignore_mismatch", action="store_true")

    return parser.parse_args(input_args)


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_") and f.endswith(".pkl")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.replace("model_", "").replace(".pkl", "")))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def trainable_parameters(model_gen):
    """Collect exactly the Stage 3 trainable tensors: UNet LoRA/conv_in, HDRev, and projector."""
    params = []
    for name, param in model_gen.unet.named_parameters():
        if "lora" in name:
            params.append(param)
    params += list(model_gen.unet.conv_in.parameters())
    params += list(model_gen.hdrev_encoder.parameters())
    params += list(model_gen.event_latent_projector.parameters())
    return params


def main(args):
    if args.test_data_root is None:
        args.test_data_root = args.train_data_root

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

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    log_file = os.path.join(args.output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a", encoding="utf-8"), logging.StreamHandler()],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    model_gen = OSEDiffHDRevGen(args)
    model_gen.set_train()
    if args.encoder_projector_pretrain_path and args.resume_from_checkpoint:
        logging.warning(
            "[PretrainInit] Ignoring encoder_projector_pretrain_path because "
            "resume_from_checkpoint restores the full Stage 3 training state."
        )
    elif args.encoder_projector_pretrain_path:
        # Clean-latent pretraining teaches HDRevEncoder + pyramid projector to
        # produce scaled SD-VAE clean latents. Stage 3 then starts from those
        # weights while UNet LoRA/VSD components remain freshly initialized.
        pretrain_checkpoint = torch.load(args.encoder_projector_pretrain_path, map_location="cpu")
        model_gen.load_encoder_projector_pretrain(
            pretrain_checkpoint,
            strict=not args.resume_ignore_mismatch,
        )
        logging.info("[PretrainInit] Loaded HDRev/projector weights from %s", args.encoder_projector_pretrain_path)

    model_reg = OSEDiff_reg(args=args, accelerator=accelerator)
    model_reg.set_train()
    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)

    model_gen.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available.")
        model_gen.unet.enable_xformers_memory_efficient_attention()
        model_reg.unet_fix.enable_xformers_memory_efficient_attention()
        model_reg.unet_update.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        model_reg.unet_fix.enable_gradient_checkpointing()
        model_reg.unet_update.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    layers_to_opt = trainable_parameters(model_gen)
    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    layers_to_opt_reg = [p for n, p in model_reg.unet_update.named_parameters() if "lora" in n]
    optimizer_reg = torch.optim.AdamW(
        layers_to_opt_reg,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler_reg = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_reg,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    dataset_train = SDEEventDataset(split="train", args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
    )
    net_lpips = accelerator.prepare(net_lpips)

    if accelerator.is_main_process:
        tracker_config = {k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in vars(args).items()}
        init_kwargs = {}
        report_targets = [target.strip() for target in args.report_to.split(",")]
        uses_wandb = args.report_to == "all" or "wandb" in report_targets
        if uses_wandb:
            import wandb

            # Only the main process initializes the W&B run through Accelerate.
            # These kwargs keep run naming/grouping explicit for multi-GPU
            # torchrun jobs while leaving tensorboard behavior unchanged.
            wandb_kwargs = {}
            if args.wandb_run_name:
                wandb_kwargs["name"] = args.wandb_run_name
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if args.wandb_tags:
                wandb_kwargs["tags"] = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
            if args.wandb_mode:
                wandb_kwargs["mode"] = args.wandb_mode
            # Explicitly pass the SSL bypass through wandb.init as well as the
            # environment. This covers W&B versions where WANDB_INSECURE is
            # ignored but Settings(insecure_disable_ssl=True) is honored.
            try:
                wandb_kwargs["settings"] = wandb.Settings(insecure_disable_ssl=True)
            except TypeError:
                logging.warning(
                    "This wandb version does not accept Settings(insecure_disable_ssl=True); "
                    "falling back to WANDB_INSECURE/WANDB_INSECURE_DISABLE_SSL environment variables."
                )
            init_kwargs["wandb"] = wandb_kwargs

        tracker_project_name = args.wandb_project if uses_wandb and args.wandb_project else args.tracker_project_name
        accelerator.init_trackers(tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)
        logging.info("===== Stage 3 HDRev training config =====")
        for key, value in sorted(vars(args).items()):
            logging.info(f"  {key}: {value}")

    resume_checkpoint_path = None
    if args.resume_from_checkpoint:
        resume_checkpoint_path = (
            find_latest_checkpoint(os.path.join(args.output_dir, "checkpoints"))
            if args.resume_from_checkpoint == "latest"
            else args.resume_from_checkpoint
        )

    start_epoch = 0
    global_step = 0
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
        accelerator.unwrap_model(model_gen).load_ckpt(checkpoint, strict=not args.resume_ignore_mismatch)
        reg_checkpoint = checkpoint.get("additional_info", {}).get("reg_checkpoint", {})
        if reg_checkpoint:
            accelerator.unwrap_model(model_reg).load_ckpt(reg_checkpoint)
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        if accelerator.is_main_process:
            logging.info(f"[Resume] {resume_checkpoint_path} at step={global_step}, epoch={start_epoch}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(start_epoch, args.num_training_epochs):
        for batch in dl_train:
            with accelerator.accumulate(model_gen, model_reg):
                bsz = batch["conditioning_pixel_values"].shape[0]
                batch["prompt"] = [args.fixed_prompt] * bsz
                batch["neg_prompt"] = [args.neg_prompt] * bsz

                x_tgt = batch["output_pixel_values"]
                need_target_latent = args.lambda_latent_align > 0
                (
                    x_tgt_pred,
                    latents_pred,
                    prompt_embeds,
                    neg_prompt_embeds,
                    start_latent,
                    target_latent,
                    model_pred,
                    sampled_timesteps,
                ) = model_gen(
                    batch,
                    args=args,
                    return_target_latent=need_target_latent,
                )

                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss_latent = torch.zeros((), device=x_tgt_pred.device)
                if need_target_latent:
                    # This alignment nudges the HDRev-projected start latent toward
                    # the SD VAE manifold without training the frozen VAE itself.
                    loss_latent = (
                        F.mse_loss(start_latent.float(), target_latent.float(), reduction="mean")
                        * args.lambda_latent_align
                    )

                if torch.cuda.device_count() > 1:
                    loss_kl = model_reg.module.distribution_matching_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        neg_prompt_embeds=neg_prompt_embeds,
                        args=args,
                    ) * args.lambda_vsd
                    loss_d = model_reg.module.diff_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        args=args,
                    ) * args.lambda_vsd_lora
                else:
                    loss_kl = model_reg.distribution_matching_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        neg_prompt_embeds=neg_prompt_embeds,
                        args=args,
                    ) * args.lambda_vsd
                    loss_d = model_reg.diff_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        args=args,
                    ) * args.lambda_vsd_lora

                loss = loss_l2 + loss_lpips + loss_latent + loss_kl

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    start_stats = tensor_stats(start_latent)
                    model_pred_stats = tensor_stats(model_pred)
                    denoised_stats = tensor_stats(latents_pred)
                    target_stats = tensor_stats(target_latent) if target_latent is not None else None
                    output_stats = tensor_stats(x_tgt_pred)
                    event_sparsity = (batch["event_voxel"] != 0).float().mean().item()
                    finite_flags = {
                        "start_latent_finite": tensor_is_finite(start_latent),
                        "model_pred_finite": tensor_is_finite(model_pred),
                        "x_denoised_finite": tensor_is_finite(latents_pred),
                        "output_finite": tensor_is_finite(x_tgt_pred),
                    }

                    logs = {
                        "loss": loss.detach().item(),
                        "loss_d": loss_d.detach().item(),
                        "loss_kl": loss_kl.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_latent": loss_latent.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "start_latent_mean": start_stats["mean"],
                        "start_latent_std": start_stats["std"],
                        "model_pred_mean": model_pred_stats["mean"],
                        "model_pred_std": model_pred_stats["std"],
                        "x_denoised_mean": denoised_stats["mean"],
                        "x_denoised_std": denoised_stats["std"],
                        "event_sparsity": event_sparsity,
                        "output_min": output_stats["min"],
                        "output_max": output_stats["max"],
                        **finite_flags,
                    }
                    if target_stats is not None:
                        logs["target_latent_mean"] = target_stats["mean"]
                        logs["target_latent_std"] = target_stats["std"]
                        logs["target_latent_min"] = target_stats["min"]
                        logs["target_latent_max"] = target_stats["max"]

                    progress_bar.set_postfix(loss=logs["loss"], loss_latent=logs["loss_latent"])
                    accelerator.log(logs, step=global_step)

                    if min(finite_flags.values()) < 1.0:
                        # Keep training behavior unchanged for now, but emit enough
                        # context to identify whether NaN appears before or after
                        # scheduler.step().
                        logging.warning(
                            "[NonFinite] step=%d timestep=%s loss=%.6f "
                            "start_finite=%.0f model_pred_finite=%.0f x_denoised_finite=%.0f output_finite=%.0f "
                            "start(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f "
                            "model_pred(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f "
                            "denoised(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f "
                            "target(mean/std/min/max)=%s event_sparsity=%.6f",
                            global_step,
                            sampled_timesteps.detach().cpu().tolist(),
                            logs["loss"],
                            finite_flags["start_latent_finite"],
                            finite_flags["model_pred_finite"],
                            finite_flags["x_denoised_finite"],
                            finite_flags["output_finite"],
                            start_stats["mean"],
                            start_stats["std"],
                            start_stats["min"],
                            start_stats["max"],
                            model_pred_stats["mean"],
                            model_pred_stats["std"],
                            model_pred_stats["min"],
                            model_pred_stats["max"],
                            denoised_stats["mean"],
                            denoised_stats["std"],
                            denoised_stats["min"],
                            denoised_stats["max"],
                            (
                                f"{target_stats['mean']:.4f}/{target_stats['std']:.4f}/"
                                f"{target_stats['min']:.4f}/{target_stats['max']:.4f}"
                                if target_stats is not None
                                else "None"
                            ),
                            event_sparsity,
                        )

                    if global_step % 10 == 0:
                        logging.info(
                            "step=%d loss=%.6f l2=%.6f lpips=%.6f kl=%.6f latent=%.6f "
                            "start(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f "
                            "model_pred(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f "
                            "denoised(mean/std)=%.4f/%.4f event_sparsity=%.6f "
                            "finite(start/pred/den/out)=%.0f/%.0f/%.0f/%.0f output(min/max)=%.4f/%.4f",
                            global_step,
                            logs["loss"],
                            logs["loss_l2"],
                            logs["loss_lpips"],
                            logs["loss_kl"],
                            logs["loss_latent"],
                            start_stats["mean"],
                            start_stats["std"],
                            start_stats["min"],
                            start_stats["max"],
                            model_pred_stats["mean"],
                            model_pred_stats["std"],
                            model_pred_stats["min"],
                            model_pred_stats["max"],
                            denoised_stats["mean"],
                            denoised_stats["std"],
                            event_sparsity,
                            finite_flags["start_latent_finite"],
                            finite_flags["model_pred_finite"],
                            finite_flags["x_denoised_finite"],
                            finite_flags["output_finite"],
                            output_stats["min"],
                            output_stats["max"],
                        )

                    if global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        reg_checkpoint = {
                            "state_dict_reg": {
                                k: v
                                for k, v in accelerator.unwrap_model(model_reg).unet_update.state_dict().items()
                                if "lora" in k
                            }
                        }
                        accelerator.unwrap_model(model_gen).save_model(
                            outf,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            optimizer_reg=optimizer_reg,
                            lr_scheduler_reg=lr_scheduler_reg,
                            global_step=global_step,
                            epoch=epoch,
                            additional_info={"reg_checkpoint": reg_checkpoint},
                        )
                        logging.info(f"[Checkpoint] step={global_step} saved={outf}")

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoints", "model_final.pkl")
        reg_checkpoint = {
            "state_dict_reg": {
                k: v
                for k, v in accelerator.unwrap_model(model_reg).unet_update.state_dict().items()
                if "lora" in k
            }
        }
        accelerator.unwrap_model(model_gen).save_model(
            final_path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_reg=optimizer_reg,
            lr_scheduler_reg=lr_scheduler_reg,
            global_step=global_step,
            epoch=epoch,
            additional_info={"reg_checkpoint": reg_checkpoint},
        )
        logging.info(f"[Final] saved={final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
