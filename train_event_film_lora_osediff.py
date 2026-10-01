import argparse
import logging
import os
import time
from pathlib import Path

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

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from models.event_lora_utils import collect_event_lora_regularization
from osediff_event_film import (
    OSEDiffEventFiLMGen,
    OSEDiffEventFiLMReg,
    parse_unet_lora_parts,
    validate_unet_lora_training_parts,
)


def setup_wandb_ssl_bypass():
    # 远程 Linux 服务器可能有自签证书问题；必须在 tracker 初始化前设置。
    os.environ.setdefault("WANDB_INSECURE", "true")
    os.environ.setdefault("WANDB_INSECURE_DISABLE_SSL", "true")


def parse_timestep_spec(value):
    value = str(value).strip()
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 1:
        t = int(parts[0])
        return t, t
    if len(parts) == 2:
        low, high = int(parts[0]), int(parts[1])
        if low > high:
            raise argparse.ArgumentTypeError(f"Invalid timestep range: {value}")
        return low, high
    raise argparse.ArgumentTypeError("Timestep must be 'n' or 'a,b'.")


def sample_shared_timestep(spec, device):
    low, high = spec
    if low == high:
        return torch.tensor([low], device=device).long()
    return torch.randint(low, high + 1, (1,), device=device).long()


def same_timestep_spec(a, b):
    return tuple(a) == tuple(b)


def format_duration(seconds):
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("model_") and f.endswith(".pkl")
    ]
    if not candidates:
        return None

    def step_id(path):
        name = os.path.basename(path).replace("model_", "").replace(".pkl", "")
        if name == "final":
            return -1
        return int(name)

    candidates.sort(key=step_id)
    return candidates[-1]


def event_edge_loss(pred, gt, events):
    # 在 event 活跃区域约束图像梯度，鼓励事件边缘附近的增强结果贴近 GT 结构。
    pred_gray = pred.mean(dim=1, keepdim=True)
    gt_gray = gt.mean(dim=1, keepdim=True)
    mask = events.abs().sum(dim=1, keepdim=True)
    mask = mask / mask.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)

    pred_dx = pred_gray[:, :, :, 1:] - pred_gray[:, :, :, :-1]
    gt_dx = gt_gray[:, :, :, 1:] - gt_gray[:, :, :, :-1]
    pred_dy = pred_gray[:, :, 1:, :] - pred_gray[:, :, :-1, :]
    gt_dy = gt_gray[:, :, 1:, :] - gt_gray[:, :, :-1, :]
    loss_x = (mask[:, :, :, 1:] * (pred_dx - gt_dx).abs()).mean()
    loss_y = (mask[:, :, 1:, :] * (pred_dy - gt_dy).abs()).mean()
    return loss_x + loss_y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[2, 3], default=2)
    parser.add_argument(
        "--unet_lora_active_parts", type=parse_unet_lora_parts, default=(),
        help="训练前向中激活的普通 UNet LoRA 分组：none 或 down,mid,up 的逗号分隔子集；默认 none。",
    )
    parser.add_argument(
        "--unet_lora_trainable_parts", type=parse_unet_lora_parts, default=(),
        help="训练时解冻的普通 UNet LoRA 分组，须为激活分组的子集；默认 none。Event-LoRA 不受此参数控制。",
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experience/event_film_lora_osediff")
    parser.add_argument("--stage1_checkpoint", type=str,
        default="checkpoints/osediff_ConcatEv_5-23-2350/checkpoints/model_93501.pkl")
    parser.add_argument("--init_from_event_checkpoint", type=str, default=None,
        help="Initialize model weights from a Stage2/Stage3 Event-FiLM checkpoint without resuming step/optimizer state.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
        help="'latest' or a checkpoint path for breakpoint resume.")
    parser.add_argument("--resume_ignore_mismatch", action="store_true")

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--event_bins", type=int, default=5)
    parser.add_argument("--event_channels", type=int, default=5)
    parser.add_argument(
        "--clora_event_active_bins", type=int, default=5,
        help="cLoRA 输入使用的前置有效 event bin 数；1 将全部事件聚集到 bin 0，通道数仍为 5。",
    )
    parser.add_argument(
        "--rgb_event_active_bins", type=int, default=5,
        help="RGB-event encoder 输入使用的前置有效 event bin 数；仅 Stage4/5 refinement 使用。",
    )
    parser.add_argument("--event_base_channels", type=int, default=32)
    parser.add_argument(
        "--mod_channels",
        type=int,
        default=32,
        help="Event/RGB 融合头的中间通道数；越大表达力越强，但 3x3 融合卷积开销近似平方增长。",
    )
    parser.add_argument("--modulation_scale", type=float, default=0.1)
    parser.add_argument(
        "--event_lora_variant",
        type=str,
        default="film_affine",
        choices=[
            "film_affine",
            "film",
            "cbam_lowrank_residual",
            "cbam_output_gate",
            "residual_cross_attn",
            "residual_cross_attn_bax",
        ],
    )
    parser.add_argument(
        "--event_lora_cbam_reduction",
        type=int,
        default=8,
        help="CBAM channel attention 的压缩比例；越大越省参数，但通道关系表达更弱。",
    )
    parser.add_argument(
        "--event_lora_cbam_spatial_kernel",
        type=int,
        default=7,
        choices=[3, 7],
        help="CBAM spatial attention 卷积核；7 的感受野更大，3 更轻量。",
    )
    parser.add_argument(
        "--event_lora_output_gate_channels",
        type=str,
        default="spatial",
        choices=["spatial", "channel_spatial"],
    )
    parser.add_argument("--mod_num_heads", type=int, default=4)
    parser.add_argument("--mod_window_size", type=int, default=8)
    parser.add_argument(
        "--regularizer_lora_mode",
        type=str,
        default="same_as_generator",
        choices=["original", "same_as_generator", "event_lora", "event_film"],
        help="unet_update 使用普通 PEFT LoRA，或使用与 Generator 同结构但权重独立的 Event-LoRA。",
    )

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--learning_rate_event", type=float, default=1e-4)
    parser.add_argument("--learning_rate_lora", type=float, default=1e-5)
    parser.add_argument("--learning_rate_reg", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--set_grads_to_none", action="store_true")

    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_edge", type=float, default=0.05)
    parser.add_argument("--lambda_smooth", type=float, default=0.001)
    parser.add_argument("--lambda_mod", type=float, default=0.001)
    parser.add_argument("--lambda_vsd", type=float, default=0.0)
    parser.add_argument("--lambda_vsd_lora", type=float, default=0.0)
    parser.add_argument("--cfg_vsd", type=float, default=7.5)

    parser.add_argument("--generator_train_timestep", type=parse_timestep_spec, default="999")
    parser.add_argument("--regularizer_train_timestep", type=parse_timestep_spec, default="999")
    parser.add_argument("--fixed_prompt", type=str, default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", type=str, default=(
        "underexposed, dark, low-light, noisy, blurry, painting, oil painting, illustration, "
        "drawing, art, sketch, cartoon, CG Style, 3D render, worst quality, low quality, "
        "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
    ))
    parser.add_argument("--lora_rank", type=int, default=4)

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, default="event_film_lora_osediff")
    parser.add_argument("--wandb_run_name", type=str, default=None,
        help="Optional W&B run name. If omitted, W&B will auto-generate a run name.")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    return parser.parse_args()


def main(args):
    active_parts, trainable_parts = validate_unet_lora_training_parts(
        args.unet_lora_active_parts, args.unet_lora_trainable_parts
    )
    args.unet_lora_active_parts = active_parts
    args.unet_lora_trainable_parts = trainable_parts
    if not same_timestep_spec(args.generator_train_timestep, args.regularizer_train_timestep):
        raise ValueError(
            "Generator 和 fine-tuned regularizer 的 timestep 参数必须一致，"
            f"got {args.generator_train_timestep} vs {args.regularizer_train_timestep}"
        )

    if "wandb" in args.report_to or args.report_to == "all":
        setup_wandb_ssl_bypass()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # 每个 Event-LoRA 层都保存多个 event 尺度对应的 1x1 projection，但一次
    # forward 只会选择与当前 token 分辨率最接近的一支。其余 projection 在该轮
    # 不参与计算，因此 DDP 必须启用 unused-parameter 检测；否则第一步 backward
    # 后仍有 bucket 未完成归约，第二步 forward 会报
    # "Expected to have finished reduction in the prior iteration"。
    #
    # 旧版 smooth/mod 的调制增量已 detach；mod 中事件头 L2 仍是独立参数梯度路径。
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    log_file = os.path.join(args.output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
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

    resume_path = None
    if args.resume_from_checkpoint is not None:
        resume_path = (
            find_latest_checkpoint(os.path.join(args.output_dir, "checkpoints"))
            if args.resume_from_checkpoint == "latest"
            else args.resume_from_checkpoint
        )
        if resume_path is None:
            logging.warning("No resume checkpoint found; starting from Stage1 checkpoint.")

    model_gen = OSEDiffEventFiLMGen(args)
    model_reg = OSEDiffEventFiLMReg(args=args, accelerator=accelerator)
    model_reg.set_shared_event_encoder(
        model_gen.event_encoder, model_gen.event_lora_config
    )

    resume_checkpoint = None
    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        logging.info(f"[Resume] Loading checkpoint: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        model_gen.load_event_checkpoint(resume_checkpoint)
        reg_state = resume_checkpoint.get("additional_info", {}).get("reg_checkpoint", {})
        if reg_state:
            model_reg.load_ckpt(reg_state, strict_config=True)
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        global_step = int(resume_checkpoint.get("global_step", 0))
    elif args.init_from_event_checkpoint is not None:
        logging.info(f"[Init] Loading Event-FiLM checkpoint without optimizer/step resume: {args.init_from_event_checkpoint}")
        init_checkpoint = torch.load(args.init_from_event_checkpoint, map_location="cpu")
        model_gen.load_event_checkpoint(init_checkpoint)
        reg_state = init_checkpoint.get("additional_info", {}).get("reg_checkpoint", {})
        if reg_state:
            model_reg.load_ckpt(reg_state, strict_config=False)
    else:
        logging.info(f"[Stage1] Loading full Stage1 checkpoint: {args.stage1_checkpoint}")
        stage1_checkpoint = model_gen.load_stage1_full(args.stage1_checkpoint)
        reg_state = stage1_checkpoint.get("additional_info", {}).get("reg_checkpoint", {})
        if reg_state:
            model_reg.load_ckpt(reg_state, strict_config=False)

    model_gen.set_train_stage(args.stage)
    model_reg.set_train()

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

    event_params = model_gen.get_event_params()
    lora_params = model_gen.get_lora_params()
    param_groups = []
    if event_params:
        param_groups.append({"params": event_params, "lr": args.learning_rate_event})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.learning_rate_lora})
    if not param_groups:
        raise RuntimeError("No trainable generator parameters found.")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    reg_params = [p for p in model_reg.parameters() if p.requires_grad]
    optimizer_reg = torch.optim.AdamW(
        reg_params,
        lr=args.learning_rate_reg,
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
    lr_scheduler_reg = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_reg,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    dataset_train = SDEEventFilmDataset(split="train", args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )
    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)

    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg, net_lpips = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg, net_lpips
    )

    if resume_checkpoint is not None and not args.resume_ignore_mismatch:
        for key, obj, label in [
            ("optimizer_state_dict", optimizer, "optimizer"),
            ("lr_scheduler_state_dict", lr_scheduler, "lr_scheduler"),
            ("optimizer_reg_state_dict", optimizer_reg, "optimizer_reg"),
            ("lr_scheduler_reg_state_dict", lr_scheduler_reg, "lr_scheduler_reg"),
        ]:
            if key in resume_checkpoint:
                try:
                    obj.load_state_dict(resume_checkpoint[key])
                    logging.info(f"[Resume] Loaded {label} state.")
                except Exception as exc:
                    logging.warning(f"[Resume] Failed to load {label}: {exc}")

    if accelerator.is_main_process:
        tracker_config = {k: str(v) if isinstance(v, tuple) else v for k, v in vars(args).items()}
        init_kwargs = {}
        if "wandb" in args.report_to or args.report_to == "all":
            try:
                import wandb
                init_kwargs["wandb"] = {"settings": wandb.Settings(insecure_disable_ssl=True)}
                if args.wandb_run_name:
                    init_kwargs["wandb"]["name"] = args.wandb_run_name
            except Exception:
                init_kwargs = {}
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)
        logging.info("=" * 60)
        for k, v in sorted(vars(args).items()):
            logging.info(f"{k}: {v}")
        logging.info("=" * 60)

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        desc=f"Stage{args.stage}",
        disable=not accelerator.is_local_main_process,
    )
    total_start = time.time()
    last_update_time = time.time()
    epoch_loss_sum, epoch_loss_count = 0.0, 0

    for epoch in range(start_epoch, args.num_training_epochs):
        logging.info(f"--- Epoch {epoch} started ---")
        for _, batch in enumerate(dl_train):
            with accelerator.accumulate(model_gen, model_reg):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                events = batch["events"]
                bsz = x_src.shape[0]
                batch["prompt"] = [args.fixed_prompt] * bsz
                batch["neg_prompt"] = [args.neg_prompt] * bsz
                shared_timestep = sample_shared_timestep(args.generator_train_timestep, x_src.device)

                x_pred, latents_pred, prompt_embeds, neg_prompt_embeds = model_gen(
                    x_src, events, shared_timestep, batch=batch
                )
                loss_l2 = F.mse_loss(x_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss_edge = event_edge_loss(x_pred.float(), x_tgt.float(), events.float()) * args.lambda_edge
                smooth_reg, mod_reg = collect_event_lora_regularization(
                    accelerator.unwrap_model(model_gen).unet,
                    device=x_src.device,
                )
                loss_smooth = smooth_reg * args.lambda_smooth
                loss_mod = mod_reg * args.lambda_mod
                loss = loss_l2 + loss_lpips + loss_edge + loss_smooth + loss_mod

                reg_runner = model_reg.module if hasattr(model_reg, "module") else model_reg
                if args.lambda_vsd > 0:
                    loss_kl = reg_runner.distribution_matching_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        neg_prompt_embeds=neg_prompt_embeds,
                        timesteps=shared_timestep,
                        cfg_vsd=args.cfg_vsd,
                        events=events,
                    ) * args.lambda_vsd
                    loss = loss + loss_kl
                else:
                    loss_kl = torch.zeros((), device=x_src.device)

                if args.lambda_vsd_lora > 0:
                    loss_d = reg_runner.diff_loss(
                        latents=latents_pred,
                        prompt_embeds=prompt_embeds,
                        timesteps=shared_timestep,
                        events=events,
                    ) * args.lambda_vsd_lora
                else:
                    loss_d = torch.zeros((), device=x_src.device)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if args.lambda_vsd_lora > 0:
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_([p for p in reg_runner.parameters() if p.requires_grad], args.max_grad_norm)
                    optimizer_reg.step()
                    lr_scheduler_reg.step()
                    optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                now = time.time()
                s_per_step = now - last_update_time
                last_update_time = now
                global_step += 1
                progress_bar.update(1)
                eta_seconds = s_per_step * max(args.max_train_steps - global_step, 0)
                elapsed = now - total_start

                if accelerator.is_main_process:
                    debug_stats = accelerator.unwrap_model(model_gen).debug_stats()
                    logs = {
                        "time/s_per_step": s_per_step,
                        "time/eta_seconds": eta_seconds,
                        "loss/total": loss.detach().item(),
                        "loss/l2": loss_l2.detach().item(),
                        "loss/lpips": loss_lpips.detach().item(),
                        "loss/edge": loss_edge.detach().item(),
                        "loss/smooth": loss_smooth.detach().item(),
                        "loss/mod": loss_mod.detach().item(),
                        "loss/vsd": loss_kl.detach().item(),
                        "loss/regularizer_diff": loss_d.detach().item(),
                    }
                    logs.update({f"event_lora/{k}": v for k, v in debug_stats.items()})
                    logs["event_lora/variant_id"] = {
                        "film_affine": 0,
                        "film": 0,
                        "cbam_lowrank_residual": 1,
                        "cbam_output_gate": 2,
                        "residual_cross_attn": 3,
                        "residual_cross_attn_bax": 4,
                    }[args.event_lora_variant]
                    progress_bar.set_postfix({
                        "loss": f"{logs['loss/total']:.4f}",
                        "eta": format_duration(eta_seconds),
                    })
                    accelerator.log(logs, step=global_step)
                    if global_step % 10 == 0:
                        logging.info(
                            f"step={global_step} | loss={logs['loss/total']:.6f} | "
                            f"elapsed={format_duration(elapsed)} | eta={format_duration(eta_seconds)}"
                        )
                    if global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(
                            outf,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            optimizer_reg=optimizer_reg,
                            lr_scheduler_reg=lr_scheduler_reg,
                            global_step=global_step,
                            epoch=epoch,
                            stage=args.stage,
                            timestep_config={
                                "generator_train_timestep": args.generator_train_timestep,
                                "regularizer_train_timestep": args.regularizer_train_timestep,
                            },
                            additional_info={
                                "reg_checkpoint": accelerator.unwrap_model(model_reg).checkpoint_state(),
                                "event_active_bins": {
                                    "clora": args.clora_event_active_bins,
                                    "rgb_event_encoder": args.rgb_event_active_bins,
                                    "total_channels": args.event_bins,
                                },
                            },
                        )
                        logging.info(f"[Checkpoint] Saved: {outf}")
                    epoch_loss_sum += logs["loss/total"]
                    epoch_loss_count += 1

            if global_step >= args.max_train_steps:
                break
        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        logging.info(f"--- Epoch {epoch} finished | global_step={global_step} | avg_loss={avg_loss:.6f} ---")
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoints", "model_final.pkl")
        accelerator.unwrap_model(model_gen).save_model(
            final_path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            optimizer_reg=optimizer_reg,
            lr_scheduler_reg=lr_scheduler_reg,
            global_step=global_step,
            epoch=epoch,
            stage=args.stage,
            timestep_config={
                "generator_train_timestep": args.generator_train_timestep,
                "regularizer_train_timestep": args.regularizer_train_timestep,
            },
            additional_info={
                "reg_checkpoint": accelerator.unwrap_model(model_reg).checkpoint_state(),
                "event_active_bins": {
                    "clora": args.clora_event_active_bins,
                    "rgb_event_encoder": args.rgb_event_active_bins,
                    "total_channels": args.event_bins,
                },
            },
        )
        logging.info(f"[Final] Saved: {final_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
