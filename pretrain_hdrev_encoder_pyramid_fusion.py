import argparse
import logging
import os
from pathlib import Path

import diffusers
import lpips
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from dataloaders.sde_event_dataset import SDEEventDataset
from models.event_latent_projector import EventPyramidLatentProjector
from models.hdrev_encoder import HDRevEncoder
from osediff_hdrev import initialize_frozen_vae, tensor_stats


def tensor_is_finite(tensor):
    """Return 1.0 when a tensor has no NaN/Inf values, for compact logging."""
    if tensor is None:
        return 1.0
    return float(torch.isfinite(tensor.detach()).all().item())


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest numbered clean-latent pretrain checkpoint."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_") and f.endswith(".pkl")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.replace("model_", "").replace(".pkl", "")))
    return os.path.join(checkpoint_dir, checkpoints[-1])


class HDRevCleanLatentPretrain(torch.nn.Module):
    """
    Pretrain HDRevEncoder + projector to predict a clean SD-VAE latent.

    Input:
      hdrev_pixel_values: low-light RGB in [0, 1], shape [B, 3, H, W].
      event_voxel: voxelized event tensor, shape [B, num_bins, H, W].

    Output:
      z_pred: scaled clean latent, shape [B, 4, H/8, W/8].
      image_pred: frozen VAE decode of z_pred, shape [B, 3, H, W] in [-1, 1].
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use the same HDRev initialization path as Stage 3 OSEDiff-HDRev.
        # These weights provide the RGB/event feature extractor before this
        # clean-latent pretrain adapts it to the SDE low-light enhancement task.
        self.hdrev_encoder = HDRevEncoder(
            num_bins=args.num_bins,
            pretrained_root=args.hdrev_pretrained_root,
            pretrained_i_path=getattr(args, "hdrev_encoder_i_path", None),
            pretrained_e_path=getattr(args, "hdrev_encoder_e_path", None),
            pretrained_merge_path=getattr(args, "hdrev_merge_path", None),
            pretrained_d_path=getattr(args, "hdrev_decoder_path", None),
            load_pretrained=True,
        )
        self.event_latent_projector = EventPyramidLatentProjector(latent_channels=4)

        # The VAE is frozen, but decoder gradients still flow back to z_pred.
        # This teaches HDRev/projector to emit a latent that the fixed SD VAE
        # decoder can actually map to the normal-light image.
        self.vae = initialize_frozen_vae(args)
        self.vae.requires_grad_(False)
        self.vae.eval()

    def set_train(self):
        """Train only HDRevEncoder and EventPyramidLatentProjector."""
        self.hdrev_encoder.train()
        self.event_latent_projector.train()
        self.vae.eval()
        self.hdrev_encoder.requires_grad_(True)
        # This clean-latent pretrain uses HDRev only as an RGB/event feature
        # extractor. HDRev's own image decoder is not called when
        # return_img=False, so keeping it trainable would create unused DDP
        # parameters with no gradient contribution to image/latent losses.
        self.hdrev_encoder.netDecoder.eval()
        self.hdrev_encoder.netDecoder.requires_grad_(False)
        self.event_latent_projector.requires_grad_(True)
        self.vae.requires_grad_(False)

    def encode_start_latent(self, hdrev_rgb, event_voxel):
        """
        Convert low-light RGB + event voxel into a scaled clean SD latent.

        The projector output uses the same scaling convention as OSEDiff:
        z_scaled = VAE latent * vae.config.scaling_factor.
        """
        hdrev_dtype = next(self.hdrev_encoder.parameters()).dtype
        projector_dtype = next(self.event_latent_projector.parameters()).dtype
        _, hdrev_features = self.hdrev_encoder(
            hdrev_rgb.to(device=self.device, dtype=hdrev_dtype),
            event_voxel.to(device=self.device, dtype=hdrev_dtype),
        )
        hdrev_features = [feat.to(dtype=projector_dtype) for feat in hdrev_features]
        return self.event_latent_projector(hdrev_features)

    @torch.no_grad()
    def encode_target_latent(self, image_minus1_1):
        """
        Encode GT normal-light image into a deterministic scaled VAE latent.

        mode() is used instead of sample() so the target latent is stable across
        epochs and the projector is not asked to fit encoder sampling noise.
        """
        vae_dtype = next(self.vae.parameters()).dtype
        posterior = self.vae.encode(image_minus1_1.to(device=self.device, dtype=vae_dtype)).latent_dist
        return posterior.mode() * self.vae.config.scaling_factor

    def decode_latent(self, z_scaled):
        """
        Decode a scaled latent with the frozen SD VAE decoder.

        Do not wrap this in no_grad: although VAE weights are frozen, the image
        loss must backpropagate through the decoder into z_scaled.
        """
        vae_dtype = next(self.vae.parameters()).dtype
        z = z_scaled.to(device=self.device, dtype=vae_dtype) / self.vae.config.scaling_factor
        return self.vae.decode(z).sample.clamp(-1, 1)

    def forward(self, batch):
        z_pred = self.encode_start_latent(batch["hdrev_pixel_values"], batch["event_voxel"])
        image_pred = self.decode_latent(z_pred)
        z_target = self.encode_target_latent(batch["output_pixel_values"])
        return image_pred, z_pred, z_target

    def load_ckpt(self, checkpoint, strict=True):
        """Load clean-latent pretrain weights for resume or later reuse."""
        self.hdrev_encoder.load_state_dict(checkpoint["state_dict_hdrev_encoder"], strict=strict)
        self.event_latent_projector.load_state_dict(
            checkpoint["state_dict_event_latent_projector"],
            strict=strict,
        )

    def save_model(
        self,
        outf,
        optimizer=None,
        lr_scheduler=None,
        global_step=0,
        epoch=0,
        additional_info=None,
    ):
        """Save only trainable pretrain modules plus optimizer state."""
        sd = {
            "stage": "osediff_lle_hdrev_clean_latent_pretrain",
            "state_dict_hdrev_encoder": self.hdrev_encoder.state_dict(),
            "state_dict_event_latent_projector": self.event_latent_projector.state_dict(),
            "stage_config": {
                "num_bins": self.args.num_bins,
                "event_norm": bool(getattr(self.args, "event_norm", True)),
                "image_size": self.args.image_size,
                "hdrev_pretrained_root": self.args.hdrev_pretrained_root,
                "lambda_l1": self.args.lambda_l1,
                "lambda_l2": self.args.lambda_l2,
                "lambda_lpips": self.args.lambda_lpips,
                "lambda_latent_align": self.args.lambda_latent_align,
            },
            "global_step": global_step,
            "epoch": epoch,
        }
        if optimizer is not None:
            sd["optimizer_state_dict"] = optimizer.state_dict()
        if lr_scheduler is not None:
            sd["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        if additional_info is not None:
            sd["additional_info"] = additional_info
        torch.save(sd, outf)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    # Paths are intentionally aligned with train_osediff_lle_hdrev.py.
    parser.add_argument("--train_data_root", type=str, required=True)
    parser.add_argument("--test_data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experience/osediff_lle_hdrev_clean_latent")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--hdrev_pretrained_root", type=str, default="checkpoints/HDRev")
    parser.add_argument("--hdrev_encoder_i_path", type=str, default=None)
    parser.add_argument("--hdrev_encoder_e_path", type=str, default=None)
    parser.add_argument("--hdrev_merge_path", type=str, default=None)
    parser.add_argument("--hdrev_decoder_path", type=str, default=None)

    # Dataset options are the same as the Stage 3 HDRev dataset path.
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--event_norm", action="store_true", default=True)
    parser.add_argument("--no_event_norm", dest="event_norm", action="store_false")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # Optimization.
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
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

    # Loss weights. Defaults emphasize reconstruction and latent compatibility,
    # because this stage is not yet trying to optimize the diffusion prior.
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=0.5)
    parser.add_argument("--lambda_latent_align", type=float, default=0.5)

    # SDEEventDataset returns prompt fields, so keep these args for compatibility
    # even though this clean-latent pretrain does not use text conditioning.
    parser.add_argument("--fixed_prompt", type=str, default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", type=str, default="underexposed, dark, low-light, noisy")

    # Logging / resume.
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="train_osediff_lle_hdrev_clean_latent")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_ignore_mismatch", action="store_true")

    return parser.parse_args(input_args)


def trainable_parameters(model):
    """Collect exactly the modules trained by clean-latent pretraining."""
    # Filter by requires_grad so HDRev's unused image decoder stays out of the
    # optimizer while the active RGB/event encoders, merge block, and latent
    # projector remain trainable.
    return [
        param
        for param in list(model.hdrev_encoder.parameters()) + list(model.event_latent_projector.parameters())
        if param.requires_grad
    ]


def main(args):
    if args.test_data_root is None:
        args.test_data_root = args.train_data_root

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = HDRevCleanLatentPretrain(args)
    model.set_train()
    layers_to_opt = trainable_parameters(model)

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

    dataset_train = SDEEventDataset(split="train", args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # LPIPS is only used as a perceptual image-space loss. It receives [-1, 1]
    # tensors and stays frozen throughout pretraining.
    net_lpips = lpips.LPIPS(net="vgg")
    net_lpips.requires_grad_(False)

    model, optimizer, dl_train, lr_scheduler, net_lpips = accelerator.prepare(
        model,
        optimizer,
        dl_train,
        lr_scheduler,
        net_lpips,
    )

    if accelerator.is_main_process:
        tracker_config = {k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in vars(args).items()}
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
        logging.info("===== Clean latent pretrain config =====")
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
        accelerator.unwrap_model(model).load_ckpt(checkpoint, strict=not args.resume_ignore_mismatch)
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        if "optimizer_state_dict" in checkpoint and not args.resume_ignore_mismatch:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in checkpoint and not args.resume_ignore_mismatch:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if accelerator.is_main_process:
            logging.info(f"[Resume] {resume_checkpoint_path} at step={global_step}, epoch={start_epoch}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="CleanLatentSteps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(start_epoch, args.num_training_epochs):
        for batch in dl_train:
            with accelerator.accumulate(model):
                target_image = batch["output_pixel_values"]
                image_pred, z_pred, z_target = model(batch)

                # Image-space losses teach the frozen VAE decoder output to match
                # normal-light GT. Latent loss keeps z_pred close to the stable
                # VAE-encoded GT manifold so later one-step diffusion training has
                # a well-behaved initialization.
                loss_l1 = F.l1_loss(image_pred.float(), target_image.float(), reduction="mean") * args.lambda_l1
                loss_l2 = F.mse_loss(image_pred.float(), target_image.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(image_pred.float(), target_image.float()).mean() * args.lambda_lpips
                loss_latent = (
                    F.mse_loss(z_pred.float(), z_target.float(), reduction="mean") * args.lambda_latent_align
                )
                loss = loss_l1 + loss_l2 + loss_lpips + loss_latent

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    z_pred_stats = tensor_stats(z_pred)
                    z_target_stats = tensor_stats(z_target)
                    output_stats = tensor_stats(image_pred)
                    event_sparsity = (batch["event_voxel"] != 0).float().mean().item()
                    output_clip_fraction = (image_pred.detach().abs() > 0.999).float().mean().item()
                    finite_flags = {
                        "z_pred_finite": tensor_is_finite(z_pred),
                        "z_target_finite": tensor_is_finite(z_target),
                        "output_finite": tensor_is_finite(image_pred),
                    }

                    logs = {
                        "loss": loss.detach().item(),
                        "loss_l1": loss_l1.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_latent": loss_latent.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "z_pred_mean": z_pred_stats["mean"],
                        "z_pred_std": z_pred_stats["std"],
                        "z_target_mean": z_target_stats["mean"],
                        "z_target_std": z_target_stats["std"],
                        "output_min": output_stats["min"],
                        "output_max": output_stats["max"],
                        "output_clip_fraction": output_clip_fraction,
                        "event_sparsity": event_sparsity,
                        **finite_flags,
                    }
                    progress_bar.set_postfix(loss=logs["loss"], latent=logs["loss_latent"])
                    accelerator.log(logs, step=global_step)

                    if global_step % 10 == 0:
                        logging.info(
                            "step=%d loss=%.6f l1=%.6f l2=%.6f lpips=%.6f latent=%.6f "
                            "z_pred(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f "
                            "z_target(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f "
                            "output(min/max/clip)=%.4f/%.4f/%.6f event_sparsity=%.6f "
                            "finite(z_pred/z_target/out)=%.0f/%.0f/%.0f",
                            global_step,
                            logs["loss"],
                            logs["loss_l1"],
                            logs["loss_l2"],
                            logs["loss_lpips"],
                            logs["loss_latent"],
                            z_pred_stats["mean"],
                            z_pred_stats["std"],
                            z_pred_stats["min"],
                            z_pred_stats["max"],
                            z_target_stats["mean"],
                            z_target_stats["std"],
                            z_target_stats["min"],
                            z_target_stats["max"],
                            output_stats["min"],
                            output_stats["max"],
                            output_clip_fraction,
                            event_sparsity,
                            finite_flags["z_pred_finite"],
                            finite_flags["z_target_finite"],
                            finite_flags["output_finite"],
                        )

                    if min(finite_flags.values()) < 1.0:
                        logging.warning("[NonFinite] step=%d loss=%.6f", global_step, logs["loss"])

                    if global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model).save_model(
                            outf,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            global_step=global_step,
                            epoch=epoch,
                        )
                        logging.info(f"[Checkpoint] step={global_step} saved={outf}")

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoints", "model_final.pkl")
        accelerator.unwrap_model(model).save_model(
            final_path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            global_step=global_step,
            epoch=epoch,
        )
        logging.info(f"[Final] saved={final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
