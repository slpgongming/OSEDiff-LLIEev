"""Stage5：在 Stage4 refinement decoder 基础上的受限全量微调。

冻结 Stable Diffusion UNet 主干、原始 VAE decoder 主体、文本编码器和
RGB/Event 辅助编码器；联合训练 UNet/VAE LoRA、EventEncoder、Event-LoRA
调制分支以及 refinement decoder 的 feature-fusion blocks。
"""

import argparse
import logging
import math
import os
import time

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
import pyiqa
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from models.event_film_lora import parse_event_lora_targets
from osediff_event_film import parse_unet_lora_parts
from osediff_event_film import OSEDiffEventFiLMGen, resolve_decoder_refinement_args
from train_idea3_decoder_refinement import (
    build_scheduler,
    event_edge_loss,
    find_latest_checkpoint,
    frozen_parameter_checksum,
    load_checkpoint_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--train_data_root", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--event_lora_checkpoint", required=True,
                        help="Stage4 refinement decoder checkpoint.")
    parser.add_argument("--event_lora_targets", type=lambda value: parse_event_lora_targets(value, allow_auto=True), default="auto")
    parser.add_argument("--unet_lora_active_parts", type=parse_unet_lora_parts, default=None)
    parser.add_argument("--unet_lora_trainable_parts", type=parse_unet_lora_parts, default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--rgb_event_pretrained_root", default=None)
    parser.add_argument("--rgb_event_encoder_i_path", default=None)
    parser.add_argument("--rgb_event_encoder_e_path", default=None)
    parser.add_argument("--rgb_event_fusion_path", default=None)
    parser.add_argument(
        "--rgb_event_strict_load", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--event_lora_variant", default="residual_cross_attn",
        choices=(
            "film_affine", "film", "cbam_lowrank_residual", "cbam_output_gate",
            "residual_cross_attn", "residual_cross_attn_bax",
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--event_channels", type=int, default=5)
    parser.add_argument("--event_bins", type=int, default=5)
    parser.add_argument("--event_base_channels", type=int, default=32)
    parser.add_argument(
        "--clora_event_active_bins", type=int, default=5,
        help="cLoRA 输入使用的前置有效 event bin 数；1 将全部事件聚集到 bin 0，通道数仍为 5。",
    )
    parser.add_argument(
        "--rgb_event_active_bins", type=int, default=5,
        help="RGB-event encoder 输入使用的前置有效 event bin 数；1 将全部事件聚集到 bin 0，通道数仍为 5。",
    )
    parser.add_argument("--mod_channels", type=int, default=64)
    parser.add_argument("--modulation_scale", type=float, default=0.1)
    parser.add_argument("--mod_num_heads", type=int, default=4)
    parser.add_argument("--mod_window_size", type=int, default=8)
    parser.add_argument("--event_lora_cbam_reduction", type=int, default=8)
    parser.add_argument("--event_lora_cbam_spatial_kernel", type=int, default=7)
    parser.add_argument("--event_lora_output_gate_channels", default="spatial")

    parser.add_argument(
        "--decoder_fusion_enabled", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--decoder_fusion_weight", type=float, default=1.0)
    parser.add_argument("--decoder_fusion_init", default="near_zero", choices=("near_zero",))
    parser.add_argument(
        "--decoder_refinement_variant",
        default="auto",
        choices=("auto", "legacy_fused", "cgdrs"),
        help="Refinement path; auto inherits the Stage4 checkpoint configuration.",
    )
    parser.add_argument("--cgdrs_event_levels", default="1,2,3")
    parser.add_argument("--cgdrs_gate_bias", type=float, default=-2.0)
    parser.add_argument(
        "--cgdrs_activity_mode", choices=("abs_mean", "abs_sum"), default="abs_mean"
    )
    parser.add_argument("--refinement_fusion_mode", default="both",
                        choices=("both", "rgb", "event"))
    parser.add_argument("--bundle_rgb_event_encoder", action="store_true")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fixed_prompt", default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", default="")
    parser.add_argument("--generator_timestep", type=int, default=999)
    parser.add_argument("--train_batch_size", type=int, default=6)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # Stage5 同时更新多个分支，默认学习率低于 Stage4；可由 shell 覆盖。
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--mixed_precision", default="fp16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--report_to", default=None)
    parser.add_argument("--tracker_project_name", default="OSEDiff_clora_stage5")
    parser.add_argument("--wandb_run_name", default="stage5_full_finetune")

    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=0.1)
    parser.add_argument("--lambda_edge", type=float, default=0.1)
    parser.add_argument("--lambda_vsd", type=float, default=0.0)
    parser.add_argument("--lambda_vsd_lora", type=float, default=0.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_mod", type=float, default=0.0)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--resume_ignore_mismatch", action="store_true")
    parser.add_argument(
        "--find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DDP 中允许部分解冻参数在当前 forward 不产生梯度。",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.decoder_refinement_enabled = True
    args.event_ablation_mode = "correct"
    args.return_raw_event_voxel = args.lambda_edge > 0
    return args


def validate_paths(args):
    if not args.decoder_fusion_enabled:
        raise ValueError("Stage5 requires --decoder-fusion-enabled.")
    for label, path in {
        "stage1_checkpoint": args.stage1_checkpoint,
        "event_lora_checkpoint": args.event_lora_checkpoint,
    }.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} does not exist: {path}")
    for label, path in {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "train_data_root": args.train_data_root,
    }.items():
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{label} does not exist: {path}")
    if not any((
        args.rgb_event_pretrained_root,
        args.rgb_event_encoder_i_path,
        args.rgb_event_encoder_e_path,
        args.rgb_event_fusion_path,
    )):
        raise ValueError("RGB-event pretrained weights are required for Stage5.")
    if args.rgb_event_pretrained_root and not os.path.isdir(args.rgb_event_pretrained_root):
        raise NotADirectoryError(
            f"rgb_event_pretrained_root does not exist: {args.rgb_event_pretrained_root}"
        )
    forbidden = {
        "lambda_vsd": args.lambda_vsd,
        "lambda_vsd_lora": args.lambda_vsd_lora,
        "lambda_smooth": args.lambda_smooth,
        "lambda_mod": args.lambda_mod,
    }
    nonzero = {key: value for key, value in forbidden.items() if value != 0}
    if nonzero:
        raise ValueError(f"Stage5 currently supports only L2/LPIPS/edge losses: {nonzero}")


def save_stage5_checkpoint(accelerator, model, optimizer, lr_scheduler, args,
                           global_step, epoch, step_in_epoch, final=False):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = "model_final.pkl" if final else f"model_{global_step}.pkl"
    output_path = os.path.join(checkpoint_dir, filename)
    accelerator.unwrap_model(model).save_model(
        output_path,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=accelerator.scaler,
        global_step=global_step,
        epoch=epoch,
        stage="stage5_full_finetune",
        timestep_config={"generator_timestep": args.generator_timestep},
        additional_info={
            "step_in_epoch": step_in_epoch,
            "event_normalization": "dual_view_clora_plus_nonzero_global_mean_std_v1",
            "event_active_bins": {
                "clora": args.clora_event_active_bins,
                "rgb_event_encoder": args.rgb_event_active_bins,
                "total_channels": args.event_bins,
            },
            "stage5_trainable_scope": (
                "unet_lora,event_encoder,vae_encoder_lora,decoder_feature_fusion"
            ),
        },
    )
    logging.info("Saved Stage5 checkpoint: %s", output_path)


def main(args):
    validate_paths(args)
    resolve_decoder_refinement_args(args, args.event_lora_checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # Stage5 联合解冻了多个分支；某些 LoRA/调制参数可能不在每次 forward
        # 的实际路径中，必须让 DDP 正确处理 unused parameters。
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters,
            )
        ],
    )
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(args.seed)

    resume_path = args.resume_from_checkpoint
    if resume_path == "latest":
        resume_path = find_latest_checkpoint(os.path.join(args.output_dir, "checkpoints"))
        if resume_path is None:
            raise FileNotFoundError("No numeric checkpoint found for --resume_from_checkpoint latest.")
    resume_checkpoint = load_checkpoint_file(resume_path) if resume_path else None

    # 初始化顺序：Stable Diffusion -> Stage1 -> Stage4 -> 可选 Stage5 resume。
    model = OSEDiffEventFiLMGen(args)
    model.load_stage1_full(args.stage1_checkpoint)
    model.load_event_checkpoint(load_checkpoint_file(args.event_lora_checkpoint))
    model.decoder_fusion_enabled = args.decoder_fusion_enabled
    model.decoder_fusion_weight = args.decoder_fusion_weight
    if resume_checkpoint is not None:
        model.load_event_checkpoint(
            resume_checkpoint,
            ignore_config_mismatch=args.resume_ignore_mismatch,
        )

    trainable_parameters = model.set_train_stage5()
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("Stage5 produced no trainable parameters.")
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    logging.info("Event-LoRA variant: %s", model.event_lora_config.variant)
    logging.info("Stage5 trainable parameter count: %d / %d", trainable_count, total_count)
    for name in trainable_names:
        logging.info("[Stage5 Trainable] %s", name)
    if args.dry_run:
        logging.info("Stage5 dry run complete; no training step was executed.")
        return

    dataset = SDEEventFilmDataset(split="train", args=args, root_dir=args.train_data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = build_scheduler(optimizer, args.lr_warmup_steps, args.max_train_steps)
    if resume_checkpoint is not None:
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in resume_checkpoint:
            lr_scheduler.load_state_dict(resume_checkpoint["lr_scheduler_state_dict"])

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    lpips_metric = pyiqa.create_metric("lpips").to(accelerator.device).eval()
    lpips_metric.requires_grad_(False)
    if accelerator.is_main_process and args.report_to:
        tracker_config = vars(args).copy()
        init_kwargs = {}
        if "wandb" in args.report_to or args.report_to == "all":
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs=init_kwargs,
        )

    start_epoch = int(resume_checkpoint.get("epoch", 0)) if resume_checkpoint else 0
    global_step = int(resume_checkpoint.get("global_step", 0)) if resume_checkpoint else 0
    start_step_in_epoch = int(
        resume_checkpoint.get("additional_info", {}).get("step_in_epoch", 0)
    ) if resume_checkpoint else 0
    if resume_checkpoint and accelerator.scaler is not None:
        scaler_state = resume_checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            accelerator.scaler.load_state_dict(scaler_state)

    initial_frozen_checksum = frozen_parameter_checksum(accelerator.unwrap_model(model))
    progress = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="stage5 full finetune",
    )
    started_at = time.time()
    for epoch in range(start_epoch, args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            if epoch == start_epoch and step < start_step_in_epoch:
                continue
            with accelerator.accumulate(model):
                rgb = batch["conditioning_pixel_values"]
                target = batch["output_pixel_values"]
                events = batch["events"]
                events_refinement = batch["events_refinement"]
                timestep = torch.tensor(
                    [args.generator_timestep], device=rgb.device, dtype=torch.long
                )
                prediction, _, _, _ = model(
                    rgb,
                    events,
                    timestep,
                    batch=batch,
                    events_refinement=events_refinement,
                    decoder_fusion_enabled=args.decoder_fusion_enabled,
                    refinement_fusion_mode=args.refinement_fusion_mode,
                )
                loss_l2 = F.mse_loss(prediction.float(), target.float())
                loss_lpips = lpips_metric(
                    prediction.float().clamp(-1, 1) * 0.5 + 0.5,
                    target.float().clamp(-1, 1) * 0.5 + 0.5,
                ).mean()
                edge_events = batch.get("event_voxel_raw", events_refinement)
                loss_edge = event_edge_loss(
                    prediction.float(), target.float(), edge_events.float()
                )
                loss = (
                    args.lambda_l2 * loss_l2
                    + args.lambda_lpips * loss_lpips
                    + args.lambda_edge * loss_edge
                )
                accelerator.backward(loss)
                if (
                    accelerator.sync_gradients
                    and global_step == 0
                    and accelerator.is_main_process
                ):
                    # 只在首个优化 step 报告一次，便于判断哪些解冻参数没有参与图。
                    unwrapped = accelerator.unwrap_model(model)
                    unused_names = [
                        name
                        for name, parameter in unwrapped.named_parameters()
                        if parameter.requires_grad and parameter.grad is None
                    ]
                    if unused_names:
                        logging.warning(
                            "Stage5 parameters without gradients on the first step "
                            "(%d): %s",
                            len(unused_names),
                            unused_names[:40],
                        )
                step_gradient_norm = 0.0
                if accelerator.sync_gradients:
                    clipped_norm = accelerator.clip_grad_norm_(
                        trainable_parameters, args.max_grad_norm
                    )
                    step_gradient_norm = float(clipped_norm.detach())
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if global_step % args.log_steps == 0:
                    logs = {
                        "loss/total": float(loss.detach()),
                        "loss/l2": float(loss_l2.detach()),
                        "loss/lpips": float(loss_lpips.detach()),
                        "loss/edge": float(loss_edge.detach()),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/gradient_norm": step_gradient_norm,
                        "time/elapsed_seconds": time.time() - started_at,
                    }
                    logs.update(accelerator.unwrap_model(model).debug_stats())
                    accelerator.log(logs, step=global_step)
                    progress.set_postfix(loss=f"{logs['loss/total']:.4f}")
                if global_step % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_stage5_checkpoint(
                            accelerator, model, optimizer, lr_scheduler, args,
                            global_step, epoch, step + 1,
                        )
                if global_step >= args.max_train_steps:
                    break
        start_step_in_epoch = 0
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    final_frozen_checksum = frozen_parameter_checksum(accelerator.unwrap_model(model))
    if not math.isclose(initial_frozen_checksum, final_frozen_checksum, rel_tol=0, abs_tol=0):
        raise RuntimeError(
            "Frozen parameter checksum changed during Stage5: "
            f"{initial_frozen_checksum} -> {final_frozen_checksum}"
        )
    if accelerator.is_main_process:
        save_stage5_checkpoint(
            accelerator, model, optimizer, lr_scheduler, args,
            global_step, epoch, 0, final=True,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
