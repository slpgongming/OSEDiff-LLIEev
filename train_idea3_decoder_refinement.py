"""idea3 Phase 1B：只训练 VAE decoder feature-fusion blocks。"""

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
from osediff_event_film import OSEDiffEventFiLMGen, parse_unet_lora_parts


def event_edge_loss(prediction, target, events):
    """在事件活跃区域约束图像梯度，沿用当前 Stage 2/3 的监督语义。"""
    def gradients(image):
        dx = image[..., :, 1:] - image[..., :, :-1]
        dy = image[..., 1:, :] - image[..., :-1, :]
        return dx, dy

    weight = events.abs().sum(dim=1, keepdim=True)
    weight = weight / weight.flatten(1).amax(dim=1).reshape(-1, 1, 1, 1).clamp_min(1e-6)
    pred_dx, pred_dy = gradients(prediction)
    gt_dx, gt_dy = gradients(target)
    return (
        (weight[..., :, 1:] * (pred_dx - gt_dx).abs()).mean()
        + (weight[..., 1:, :] * (pred_dy - gt_dy).abs()).mean()
    )


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("model_") and filename.endswith(".pkl"):
            stem = filename[len("model_"):-len(".pkl")]
            if stem.isdigit():
                candidates.append((int(stem), os.path.join(checkpoint_dir, filename)))
    return max(candidates, default=(None, None))[1]


def load_checkpoint_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--train_data_root", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--event_lora_checkpoint", required=True)
    parser.add_argument("--event_lora_targets", type=lambda value: parse_event_lora_targets(value, allow_auto=True), default="auto")
    parser.add_argument("--unet_lora_active_parts", type=parse_unet_lora_parts,
                        default=("down", "mid", "up"))
    parser.add_argument("--unet_lora_trainable_parts", type=parse_unet_lora_parts, default=None,
                        help="Stage4 freezes UNet; only none is accepted.")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--rgb_event_pretrained_root", default=None)
    parser.add_argument("--rgb_event_encoder_i_path", default=None)
    parser.add_argument("--rgb_event_encoder_e_path", default=None)
    parser.add_argument("--rgb_event_fusion_path", default=None)
    parser.add_argument(
        "--rgb_event_strict_load",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--event_lora_variant",
        default="cbam_lowrank_residual",
        choices=(
            "film_affine",
            "film",
            "cbam_lowrank_residual",
            "cbam_output_gate",
            "residual_cross_attn",
            "residual_cross_attn_bax",
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
    parser.add_argument("--mod_channels", type=int, default=32)
    parser.add_argument("--modulation_scale", type=float, default=0.1)
    parser.add_argument("--mod_num_heads", type=int, default=4)
    parser.add_argument("--mod_window_size", type=int, default=8)
    parser.add_argument("--event_lora_cbam_reduction", type=int, default=8)
    parser.add_argument("--event_lora_cbam_spatial_kernel", type=int, default=7)
    parser.add_argument("--event_lora_output_gate_channels", default="spatial")

    parser.add_argument(
        "--decoder_fusion_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--decoder_fusion_weight", type=float, default=1.0)
    parser.add_argument("--decoder_fusion_init", default="near_zero", choices=("near_zero",))
    parser.add_argument(
        "--decoder_refinement_variant",
        default="legacy_fused",
        choices=("legacy_fused", "cgdrs"),
        help="Decoder refinement implementation; legacy_fused preserves the previous path.",
    )
    parser.add_argument(
        "--cgdrs_event_levels",
        default="1,2,3",
        help="CGDRS encoder pyramid levels that may receive event residuals (0-based).",
    )
    parser.add_argument("--cgdrs_gate_bias", type=float, default=-2.0)
    parser.add_argument(
        "--cgdrs_activity_mode",
        choices=("abs_mean", "abs_sum"),
        default="abs_mean",
    )
    parser.add_argument(
        "--refinement_fusion_mode", default="both", choices=("both", "rgb", "event")
    )
    parser.add_argument("--bundle_rgb_event_encoder", action="store_true")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fixed_prompt", default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", default="")
    parser.add_argument("--generator_timestep", type=int, default=999)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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
    parser.add_argument("--tracker_project_name", default="idea3-decoder-refinement")
    parser.add_argument(
        "--run_name",
        "--wandb_run_name",
        dest="run_name",
        type=str,
        default=None,
        help="Tracker run name (legacy --wandb_run_name alias is retained).",
    )

    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_edge", type=float, default=0.1)
    parser.add_argument("--lambda_vsd", type=float, default=0.0)
    parser.add_argument("--lambda_vsd_lora", type=float, default=0.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_mod", type=float, default=0.0)

    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--resume_ignore_mismatch", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.decoder_refinement_enabled = False
    args.event_ablation_mode = "correct"
    args.return_raw_event_voxel = args.lambda_edge > 0
    return args


def validate_paths(args):
    if not args.decoder_fusion_enabled:
        raise ValueError("Phase 1B training requires --decoder-fusion-enabled.")
    required_files = {
        "stage1_checkpoint": args.stage1_checkpoint,
        "event_lora_checkpoint": args.event_lora_checkpoint,
    }
    required_dirs = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "train_data_root": args.train_data_root,
    }
    for label, path in required_files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} does not exist: {path}")
    for label, path in required_dirs.items():
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{label} does not exist: {path}")
    if not any((
        args.rgb_event_pretrained_root,
        args.rgb_event_encoder_i_path,
        args.rgb_event_encoder_e_path,
        args.rgb_event_fusion_path,
    )):
        raise ValueError("RGB-event pretrained weights are required for Phase 1B.")
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
        raise ValueError(f"Phase 1B requires frozen cLoRA/UNet regularizers: {nonzero}")


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        span = max(total_steps - warmup_steps, 1)
        return remaining / span
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def frozen_parameter_checksum(model):
    """轻量 checksum，用于启动/结束时发现冻结参数被意外更新。"""
    checksum = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            checksum += parameter.detach().double().sum().cpu()
    return float(checksum)


def save_checkpoint(
    accelerator,
    model,
    optimizer,
    lr_scheduler,
    args,
    global_step,
    epoch,
    step_in_epoch,
    final=False,
):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = "model_final.pkl" if final else f"model_{global_step}.pkl"
    output_path = os.path.join(checkpoint_dir, filename)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_model(
        output_path,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=accelerator.scaler,
        global_step=global_step,
        epoch=epoch,
        stage="idea3_phase1b",
        timestep_config={"generator_timestep": args.generator_timestep},
        additional_info={
            "step_in_epoch": step_in_epoch,
            "event_normalization": "dual_view_clora_plus_nonzero_global_mean_std_v1",
            "event_active_bins": {
                "clora": args.clora_event_active_bins,
                "rgb_event_encoder": args.rgb_event_active_bins,
                "total_channels": args.event_bins,
            },
        },
    )
    logging.info("Saved checkpoint: %s", output_path)


def main(args):
    if getattr(args, "unet_lora_trainable_parts", None) not in (None, ()):
        raise ValueError("Stage4 freezes UNet; ordinary UNet LoRA cannot be trainable.")
    validate_paths(args)
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # CGDRS 默认屏蔽 level 0 event 分支；DDP 需允许该分支参数本轮无梯度。
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ],
    )
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(args.seed)

    resume_path = args.resume_from_checkpoint
    if resume_path == "latest":
        resume_path = find_latest_checkpoint(
            os.path.join(args.output_dir, "checkpoints")
        )
        if resume_path is None:
            raise FileNotFoundError("No numeric checkpoint found for --resume_from_checkpoint latest.")
    resume_checkpoint = load_checkpoint_file(resume_path) if resume_path else None

    # 初始化顺序：base -> Stage1 -> cLoRA checkpoint -> RGB-event -> feature decoder -> resume。
    model = OSEDiffEventFiLMGen(args)
    model.load_stage1_full(args.stage1_checkpoint)
    model.load_event_checkpoint(load_checkpoint_file(args.event_lora_checkpoint))
    args.decoder_refinement_enabled = True
    model.decoder_refinement_enabled = True
    model.decoder_fusion_enabled = args.decoder_fusion_enabled
    model.decoder_fusion_weight = args.decoder_fusion_weight
    model.install_decoder_refinement()
    if resume_checkpoint is not None:
        model.load_event_checkpoint(
            resume_checkpoint,
            ignore_config_mismatch=args.resume_ignore_mismatch,
        )

    model.activate_lora_adapters(args.unet_lora_active_parts)

    trainable_parameters = model.set_train_decoder_refinement()
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or not all(
        "vae.decoder.feature_fusion_blocks." in name for name in trainable_names
    ):
        raise RuntimeError(f"Invalid Phase 1B trainable whitelist: {trainable_names}")
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    logging.info("Event-LoRA variant: %s", model.event_lora_config.variant)
    logging.info("Trainable parameter count: %d / %d", trainable_count, total_count)
    for name in trainable_names:
        logging.info("[Trainable] %s", name)

    if args.dry_run:
        logging.info("Dry run complete; no training step was executed.")
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
    lr_scheduler = build_scheduler(
        optimizer, args.lr_warmup_steps, args.max_train_steps
    )
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
            init_kwargs["wandb"] = {}
            if args.run_name:
                init_kwargs["wandb"]["name"] = args.run_name
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

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
        desc="idea3 refinement",
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
                        save_checkpoint(
                            accelerator,
                            model,
                            optimizer,
                            lr_scheduler,
                            args,
                            global_step,
                            epoch,
                            step + 1,
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
            "Frozen parameter checksum changed during Phase 1B: "
            f"{initial_frozen_checksum} -> {final_frozen_checksum}"
        )
    if accelerator.is_main_process:
        save_checkpoint(
            accelerator,
            model,
            optimizer,
            lr_scheduler,
            args,
            global_step,
            epoch,
            0,
            final=True,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
