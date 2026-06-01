import argparse
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataloaders.sde_indoor_eglol_dataset import SDEIndoorEGLolDataset
from models.event_guided_osediff import EventGuidedOSEDiff


def parse_args(input_args=None, stage_default="stage1"):
    parser = argparse.ArgumentParser("Event-guided OSEDiff Stage 1/2 training")
    parser.add_argument("--stage", type=str, default=stage_default, choices=["stage1", "stage2"])
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--test_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrained_osediff_lora_path", type=str, default="checkpoints/osediff_lle_5-14-1935/checkpoints/model_94001.pkl")
    parser.add_argument("--pretrained_lle_encoder_path", type=str, default="checkpoints/HDRev")
    parser.add_argument("--stage1_checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_checkpoint_path", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")

    parser.add_argument("--train_timestep", type=str, default="500")
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--event_norm", type=str, default="none", choices=["none", "zscore"])
    parser.add_argument("--split_event", action="store_true")

    parser.add_argument("--fixed_prompt", type=str, default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", type=str, default="")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_lle_encoder", type=float, default=None)
    parser.add_argument("--lr_pyramid_fusion", type=float, default=None)
    parser.add_argument("--lr_control_adapter", type=float, default=None)
    parser.add_argument("--lr_vae_lora", type=float, default=None)
    parser.add_argument("--lr_unet_lora", type=float, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=0.0)

    parser.add_argument("--control_alpha_init", type=float, default=0.0)
    parser.add_argument("--control_alpha_final", type=float, default=1.0)
    parser.add_argument("--control_alpha_warmup_steps", type=int, default=5000)
    parser.add_argument("--use_learnable_control_alpha", action="store_true")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="event_guided_osediff")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_log_image_interval", type=int, default=5000)
    parser.add_argument("--num_wandb_visual_samples", type=int, default=4)
    parser.add_argument("--wandb_visual_timestep", type=int, default=500)

    args = parser.parse_args(input_args)
    for name in ["lr_lle_encoder", "lr_pyramid_fusion", "lr_control_adapter"]:
        if getattr(args, name) is None:
            setattr(args, name, args.lr)
    if args.lr_vae_lora is None:
        args.lr_vae_lora = args.lr * 0.2
    if args.lr_unet_lora is None:
        args.lr_unet_lora = args.lr * 0.2
    return args


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_wandb_ssl_bypass(wandb_module=None):
    # Some Linux servers use an internal CA bundle that W&B cannot verify.
    # Set both environment switches before wandb.init so the HTTP client skips
    # TLS certificate verification instead of failing with "unknown authority".
    os.environ.setdefault("WANDB_INSECURE", "true")
    os.environ.setdefault("WANDB_INSECURE_DISABLE_SSL", "true")

    if wandb_module is None:
        return None

    try:
        # Newer wandb versions also accept an explicit Settings flag. The
        # environment variables above remain the fallback for older releases.
        return wandb_module.Settings(insecure_disable_ssl=True)
    except Exception:
        return None


def build_loader(args, rank: int, world_size: int):
    dataset = SDEIndoorEGLolDataset(
        root=args.train_root,
        patch_size=args.patch_size,
        num_bins=args.num_bins,
        is_train=True,
        full_resolution=False,
        event_norm=args.event_norm,
        split_event=args.split_event,
        max_samples=args.max_train_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No SDE indoor samples found under: {args.train_root}")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader, sampler


def maybe_build_lpips(args, device):
    if args.lambda_lpips <= 0:
        return None
    import lpips
    net_lpips = lpips.LPIPS(net="vgg").to(device)
    net_lpips.requires_grad_(False)
    net_lpips.eval()
    return net_lpips


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def find_resume_path(args):
    if args.resume_checkpoint_path:
        return args.resume_checkpoint_path
    if args.auto_resume:
        candidate = os.path.join(args.output_dir, "latest_train_state.pkl")
        if os.path.exists(candidate):
            return candidate
    return None


def save_checkpoint(model, optimizer, scaler, args, global_step: int, epoch: int, rank: int, suffix: str = "latest"):
    if not is_main_process(rank):
        return
    os.makedirs(args.output_dir, exist_ok=True)
    raw_model = unwrap_model(model)
    payload = raw_model.checkpoint_dict(args=args, global_step=global_step, epoch=epoch)
    model_path = os.path.join(args.output_dir, f"{suffix}_model.pkl")
    train_path = os.path.join(args.output_dir, f"{suffix}_train_state.pkl")
    torch.save(payload, model_path)
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["scaler_state_dict"] = scaler.state_dict() if scaler is not None else None
    torch.save(payload, train_path)
    print(f"[Checkpoint] Saved model={model_path} train_state={train_path}")


def resume_if_needed(model, optimizer, scaler, args, device):
    resume_path = find_resume_path(args)
    if not resume_path:
        return 0, 0
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_event_guided_checkpoint(resume_path, load_lora=True)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    print(f"[Resume] Loaded {resume_path} at step={checkpoint.get('global_step', 0)} epoch={checkpoint.get('epoch', 0)}")
    return int(checkpoint.get("global_step", 0)), int(checkpoint.get("epoch", 0))


def image_tensor_to_numpy(tensor):
    tensor = tensor.detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


def event_tensor_to_numpy(event_tensor):
    event = event_tensor.detach().float().cpu()
    pos = torch.clamp(event, min=0).sum(dim=0)
    neg = torch.clamp(-event, min=0).sum(dim=0)
    if pos.max() > 0:
        pos = pos / pos.max()
    if neg.max() > 0:
        neg = neg / neg.max()
    img = torch.zeros((3, event.shape[1], event.shape[2]))
    img[0] = pos
    img[2] = neg
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def log_wandb_images(wandb, batch, outputs, global_step, max_samples):
    count = min(max_samples, outputs["output"].shape[0])
    images = []
    for idx in range(count):
        low = image_tensor_to_numpy(batch["conditioning_pixel_values"][idx])
        gt = image_tensor_to_numpy(batch["output_pixel_values"][idx])
        pred = image_tensor_to_numpy(outputs["output"][idx])
        event = event_tensor_to_numpy(batch["event_values"][idx])
        compare = np.concatenate([low, gt, pred, event], axis=1)
        images.append(wandb.Image(compare, caption="low | gt | pred | event"))
    wandb.log({"train_vis/compare": images}, step=global_step)


def train_main(args):
    rank, local_rank, world_size, device = setup_distributed()
    set_random_seed(args.seed + rank)

    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[Config] stage={args.stage} world_size={world_size} device={device}")

    loader, sampler = build_loader(args, rank, world_size)
    model = EventGuidedOSEDiff(args, stage=args.stage, device=device)
    if args.stage == "stage2" and args.stage1_checkpoint_path:
        model.load_event_guided_checkpoint(args.stage1_checkpoint_path, load_lora=True)
        model.configure_trainable("stage2")

    param_groups = model.build_optimizer_param_groups(args, stage=args.stage)
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    use_amp = args.mixed_precision in ["fp16", "bf16"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16" and device.type == "cuda")
    net_lpips = maybe_build_lpips(args, device)

    global_step, start_epoch = resume_if_needed(model, optimizer, scaler, args, device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)

    wandb = None
    if args.use_wandb and is_main_process(rank):
        configure_wandb_ssl_bypass()
        import wandb as wandb_module
        wandb = wandb_module
        wandb_settings = configure_wandb_ssl_bypass(wandb)
        wandb_init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name or args.stage,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
            "config": vars(args),
        }
        if wandb_settings is not None:
            wandb_init_kwargs["settings"] = wandb_settings
        wandb.init(**wandb_init_kwargs)

    if args.dry_run:
        batch = next(iter(loader))
        with torch.no_grad():
            outputs = unwrap_model(model)(
                batch,
                train_timestep=args.train_timestep,
                global_step=max(global_step, 1),
                fixed_prompt=args.fixed_prompt,
                neg_prompt=args.neg_prompt,
            )
        if is_main_process(rank):
            print(
                f"[DryRun] output={tuple(outputs['output'].shape)} "
                f"alpha={outputs['control_alpha'].item():.6f} "
                f"residual_norm={outputs['control_residual_norm'].item():.6f} "
                f"event_nonzero={batch['event_values'].count_nonzero().item()}"
            )
        return

    model.train()
    for epoch in range(start_epoch, args.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16):
                outputs = model(
                    batch,
                    train_timestep=args.train_timestep,
                    global_step=max(global_step, 1),
                    fixed_prompt=args.fixed_prompt,
                    neg_prompt=args.neg_prompt,
                )
                target = batch["output_pixel_values"].to(device)
                loss_l1 = F.l1_loss(outputs["output"].float(), target.float()) * args.lambda_l1
                if net_lpips is not None:
                    loss_lpips = net_lpips(outputs["output"].float(), target.float()).mean() * args.lambda_lpips
                else:
                    loss_lpips = torch.tensor(0.0, device=device)
                loss = loss_l1 + loss_lpips

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if is_main_process(rank) and global_step % 10 == 0:
                print(
                    f"step={global_step} loss={loss.item():.6f} "
                    f"l1={loss_l1.item():.6f} lpips={loss_lpips.item():.6f} "
                    f"alpha={outputs['control_alpha'].item():.4f} "
                    f"res_norm={outputs['control_residual_norm'].item():.4f}"
                )
                if wandb is not None:
                    wandb.log({
                        "loss/total": loss.item(),
                        "loss/l1": loss_l1.item(),
                        "loss/lpips": loss_lpips.item(),
                        "control/alpha": outputs["control_alpha"].item(),
                        "control/residual_norm": outputs["control_residual_norm"].item(),
                    }, step=global_step)

            if wandb is not None and args.wandb_log_image_interval > 0 and global_step % args.wandb_log_image_interval == 0:
                log_wandb_images(wandb, batch, outputs, global_step, args.num_wandb_visual_samples)

            if global_step % args.checkpointing_steps == 0:
                save_checkpoint(model, optimizer, scaler, args, global_step, epoch, rank)
                if world_size > 1:
                    dist.barrier()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    save_checkpoint(model, optimizer, scaler, args, global_step, epoch, rank, suffix="latest")
    if wandb is not None:
        wandb.finish()
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train_main(parse_args(stage_default="stage1"))
