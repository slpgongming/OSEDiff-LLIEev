"""双卡 24GB 起步：每卡 batch=1，累积4，只训练 mask/alpha，无 VSD 模型。"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from models.rgb_event_mask_lora import MaskFusionConfig, load_mask_state
from osediff_mask_fusion import OSEDiffMaskFusion, file_sha256, load_checkpoint


class EpochSeededDataset(Dataset):
    """按 epoch/index 固定共享几何增强，resume 跳过已消费样本时不改变后续 crop。"""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        py_state, np_state = random.getstate(), np.random.get_state()
        seed = (self.seed + self.epoch * len(self) + index) % (2 ** 32)
        try:
            random.seed(seed)
            np.random.seed(seed)
            # 数据集增强仅使用 Python/NumPy；不改 CUDA RNG，VAE 采样由 Accelerate 保存。
            return self.dataset[index]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


class EpochShuffleSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self), generator=generator).tolist())

    def set_epoch(self, epoch):
        self.epoch = epoch


def prepare_epoch_loader(accelerator, loader, dataset, sampler, epoch, next_batch):
    """skip_first_batches 会新建 DataLoaderShard；必须给新对象也设置 epoch。"""
    dataset.epoch = epoch
    sampler.set_epoch(epoch)
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)
    active = accelerator.skip_first_batches(loader, next_batch) if next_batch else loader
    if hasattr(active, "set_epoch"):
        active.set_epoch(epoch)
    return active


def event_edge_loss(pred, gt, events):
    """沿用 idea4 的 event 活跃区 GT 梯度监督；监督目标来自 GT 而非事件噪声。"""
    pred, gt = pred.mean(1, keepdim=True), gt.mean(1, keepdim=True)
    activity = events.abs().sum(1, keepdim=True)
    activity = activity / activity.flatten(1).amax(1).view(-1, 1, 1, 1).clamp_min(1e-6)
    dx = (pred[..., 1:] - pred[..., :-1]) - (gt[..., 1:] - gt[..., :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (gt[..., 1:, :] - gt[..., :-1, :])
    return (activity[..., 1:] * dx.abs()).mean() + (activity[..., 1:, :] * dy.abs()).mean()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--train_data_root", required=True)
    parser.add_argument("--rgb_checkpoint", help="6 号 checkpoint；resume 时从 idea6 权重内读取")
    parser.add_argument("--event_checkpoint", help="90 号 checkpoint；不会覆盖 6 号 RGB 权重")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_from_checkpoint", help="checkpoint-STEP 目录或 latest")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--learning_rate_mask", type=float, default=1e-4)
    parser.add_argument("--learning_rate_strength", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_edge", type=float, default=0.05)
    parser.add_argument("--mask_channels", type=int, default=16)
    parser.add_argument("--max_strength", type=float, default=2.0)
    parser.add_argument("--legacy_event_lora_alpha", type=float, default=8.0)
    parser.add_argument("--generator_timestep", type=int, default=500)
    parser.add_argument("--fixed_prompt", default="high quality, bright, detailed photo")
    parser.add_argument("--mixed_precision", choices=("fp16", "bf16", "no"), default="fp16")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="使用 idea6 显式闭包重算 UNet/VAE；不调用 idea4 的有状态 checkpoint")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--report_to", choices=("tensorboard", "none"), default="tensorboard")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    if not args.resume_from_checkpoint and not (args.rgb_checkpoint and args.event_checkpoint):
        parser.error("fresh training requires --rgb_checkpoint and --event_checkpoint")
    for key in ("train_batch_size", "gradient_accumulation_steps", "max_train_steps", "checkpointing_steps", "logging_steps"):
        if getattr(args, key) < 1:
            parser.error(f"--{key} must be positive")
    if args.dataloader_num_workers < 0:
        parser.error("dataloader_num_workers must be nonnegative")
    for key in ("learning_rate_mask", "learning_rate_strength", "max_grad_norm"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key} must be finite and positive")
    if any(not np.isfinite(x) or x < 0 for x in (args.lambda_l2, args.lambda_lpips, args.lambda_edge)):
        parser.error("loss weights must be finite and nonnegative")
    if args.lambda_l2 + args.lambda_lpips + args.lambda_edge == 0:
        parser.error("at least one image loss must be positive")
    # 首次实验固定 256/5 bins，杜绝 CLI 漂移造成不公平比较。
    args.image_size = 256
    args.event_bins = args.event_channels = args.clora_event_active_bins = args.rgb_event_active_bins = 5
    args.event_ablation_mode = "correct"
    return args


def resolve_resume(args):
    if not args.resume_from_checkpoint:
        return None
    if args.resume_from_checkpoint == "latest":
        candidates = [p for p in Path(args.output_dir).glob("checkpoint-*")
                      if p.name.split("-")[-1].isdigit() and (p / "progress.json").is_file()]
        if not candidates:
            raise FileNotFoundError("No complete checkpoint found for latest")
        return max(candidates, key=lambda p: int(p.name.split("-")[-1]))
    path = Path(args.resume_from_checkpoint)
    if not (path / "progress.json").is_file():
        raise FileNotFoundError(f"Incomplete/missing resume directory: {path}")
    return path


def register_checkpoint_hooks(accelerator):
    """只保存源 adapter/mask；另存各 rank RNG，兼容旧 Accelerate + 新 torch.load 默认值。"""
    def save_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(models[0]).export_checkpoint(), Path(output_dir) / "fusion.pt")
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
        if accelerator.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(accelerator.device)
        torch.save(rng, Path(output_dir) / f"idea6_rng_{accelerator.process_index}.pt")
        weights.clear()  # 避免 Accelerate 另存整套冻结 SD 权重。

    def load_hook(models, input_dir):
        saved = load_checkpoint(Path(input_dir) / "fusion.pt")
        load_mask_state(accelerator.unwrap_model(models[0]).unet, saved["mask_state_dict"])
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def restore_rank_rng(accelerator, directory):
    # 在 accelerator.load_state 之后显式恢复，不能依赖旧版本吞掉异常的 RNG 加载。
    state = load_checkpoint(Path(directory) / f"idea6_rng_{accelerator.process_index}.pt")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if accelerator.device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], accelerator.device)


def main(args):
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import ProjectConfiguration, set_seed
    from dataloaders.sde_event_film_dataset import SDEEventFilmDataset

    for path in (args.pretrained_model_name_or_path, args.train_data_root):
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    resume = resolve_resume(args)
    if not resume and (Path(args.output_dir) / "run.json").exists():
        raise FileExistsError("Existing run.json: use resume or a new output_dir")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision, rng_types=[],
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir,
                                             logging_dir=str(Path(args.output_dir) / "logs")),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("Full SD training requires CUDA; CPU tensor tests are under tests/")
    set_seed(args.seed, device_specific=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[args.mixed_precision]
    if resume:
        progress = json.loads((resume / "progress.json").read_text(encoding="utf-8"))
        # 数据/优化语义不可在恢复时悄悄改变；max steps、日志和 checkpoint 间隔可调整。
        critical = ("train_data_root", "train_batch_size", "gradient_accumulation_steps", "mixed_precision",
                    "seed", "lambda_l2", "lambda_lpips", "lambda_edge", "learning_rate_mask",
                    "learning_rate_strength", "max_grad_norm", "pretrained_model_name_or_path")
        for key in critical:
            if progress["args"][key] != getattr(args, key):
                raise ValueError(f"Resume config changed: {key}")
        if progress["world_size"] != accelerator.num_processes:
            raise ValueError("Resume requires the same world size for RNG/data alignment")
        model = OSEDiffMaskFusion.from_checkpoint(args.pretrained_model_name_or_path,
                 load_checkpoint(resume / "fusion.pt"), device=accelerator.device, frozen_dtype=dtype,
                 gradient_checkpointing=args.gradient_checkpointing)
    else:
        config = MaskFusionConfig(args.mask_channels, args.max_strength, args.legacy_event_lora_alpha)
        model = OSEDiffMaskFusion(args.pretrained_model_name_or_path,
                    load_checkpoint(args.rgb_checkpoint), load_checkpoint(args.event_checkpoint), config,
                    fixed_prompt=args.fixed_prompt, timestep=args.generator_timestep,
                    device=accelerator.device, frozen_dtype=dtype,
                    gradient_checkpointing=args.gradient_checkpointing)
        model.provenance = {name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
                            for name, path in (("rgb6", args.rgb_checkpoint), ("event90", args.event_checkpoint))}
        progress = {"global_step": 0, "epoch": 0, "next_batch": 0}
    # 数据集 prompt 只用于记录；实际生成器使用 checkpoint 内固定 prompt/timestep。
    args.fixed_prompt, args.generator_timestep = model.fixed_prompt, model.timestep
    args.mask_channels = model.fusion_config.mask_channels
    args.max_strength = model.fusion_config.max_strength
    args.legacy_event_lora_alpha = model.fusion_config.legacy_event_lora_alpha
    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()
    trainable_count = model.assert_trainable_scope()
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith("raw_strength")],
         "lr": args.learning_rate_mask},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.endswith("raw_strength")],
         "lr": args.learning_rate_strength},
    ]
    # 首次实验常数 LR、无 weight decay，减少 gate/alpha 缩放耦合的额外偏置。
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    dataset = EpochSeededDataset(SDEEventFilmDataset("train", args), args.seed)
    sampler = EpochShuffleSampler(dataset, args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, sampler=sampler,
              num_workers=args.dataloader_num_workers, pin_memory=True,
              generator=torch.Generator().manual_seed(args.seed))
    # 只有使用 LPIPS 时才创建 VGG；不存在 regularizer、第二个优化器或第二套 UNet。
    perceptual = None
    if args.lambda_lpips > 0:
        import lpips
        perceptual = lpips.LPIPS(net="vgg").eval().requires_grad_(False).to(accelerator.device)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if resume and progress["dataset_size"] != len(dataset):
        raise ValueError("Dataset size changed since checkpoint")

    register_checkpoint_hooks(accelerator)
    if resume:
        accelerator.load_state(str(resume))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (out / "run.json").write_text(json.dumps({"args": vars(args), "trainable_parameters": trainable_count,
                    "world_size": accelerator.num_processes, "provenance": accelerator.unwrap_model(model).provenance},
                    ensure_ascii=False, indent=2), encoding="utf-8")
    accelerator.init_trackers("idea6_mask_fusion", config=vars(args))
    accelerator.print(f"idea6 trainable={trainable_count:,}; effective batch="
                      f"{args.train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    global_step, epoch, next_batch = progress["global_step"], progress["epoch"], progress["next_batch"]

    def save_state(current_epoch, batch_index):
        directory = out / f"checkpoint-{global_step}"
        accelerator.save_state(str(directory))  # 所有 rank 保存各自 RNG；仅主进程写模型/进度。
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            payload = dict(global_step=global_step, epoch=current_epoch, next_batch=batch_index,
                           world_size=accelerator.num_processes, dataset_size=len(dataset), args=vars(args))
            # progress.json 最后写入，latest 不会选到未完成的 checkpoint。
            temp = directory / "progress.json.tmp"
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(directory / "progress.json")
        accelerator.wait_for_everyone()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    started, last_saved = time.perf_counter(), -1
    if resume:
        # tracker/文件初始化可能使用随机数；在进入训练前最后恢复采样 RNG。
        restore_rank_rng(accelerator, resume)
    while global_step < args.max_train_steps:
        active_loader = prepare_epoch_loader(accelerator, loader, dataset, sampler, epoch, next_batch)
        offset = next_batch
        for local_index, batch in enumerate(active_loader):
            batch_index = offset + local_index
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    pred = model(batch["conditioning_pixel_values"], batch["events"],
                                 collect_stats=(global_step == 0 or (global_step + 1) % args.logging_steps == 0))
                    target = batch["output_pixel_values"].float()
                    loss_l2 = F.mse_loss(pred.float(), target) * args.lambda_l2
                    loss_lpips = (perceptual(pred.float(), target).mean() * args.lambda_lpips
                                  if perceptual is not None else pred.new_zeros(()))
                    loss_edge = (event_edge_loss(pred.float(), target, batch["events"].float()) * args.lambda_edge
                                 if args.lambda_edge > 0 else pred.new_zeros(()))
                    loss = loss_l2 + loss_lpips + loss_edge
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {global_step}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            next_batch = batch_index + 1
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                global_step += 1
                if global_step == 1 or global_step % args.logging_steps == 0:
                    values = torch.stack((loss.detach(), loss_l2.detach(), loss_lpips.detach(), loss_edge.detach())).float()
                    values = accelerator.reduce(values, reduction="mean").cpu().tolist()
                    stats = dict(zip(("loss", "loss_l2", "loss_lpips", "loss_edge"), values))
                    # 各卡峰值单独记录；DDP 显存不能相加视为单卡容量。
                    memory = torch.tensor([torch.cuda.max_memory_allocated(accelerator.device),
                                           torch.cuda.max_memory_reserved(accelerator.device)], device=accelerator.device) / 2**30
                    gathered = accelerator.gather(memory).cpu().reshape(-1, 2).tolist()
                    for rank, (allocated, reserved) in enumerate(gathered):
                        stats[f"gpu{rank}/peak_allocated_gib"] = allocated
                        stats[f"gpu{rank}/peak_reserved_gib"] = reserved
                    stats.update(accelerator.unwrap_model(model).debug_stats())
                    accelerator.log(stats, step=global_step)
                    if accelerator.is_main_process:
                        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"step": global_step, "elapsed_s": time.perf_counter()-started, **stats}) + "\n")
                    accelerator.print(f"step={global_step} loss={values[0]:.6f} memory_GiB={gathered}")
                if global_step % args.checkpointing_steps == 0:
                    save_state(epoch, next_batch)
                    last_saved = global_step
                if global_step >= args.max_train_steps:
                    break
        if next_batch >= len(loader):
            epoch += 1
            next_batch = 0
    if last_saved != global_step:
        save_state(epoch, next_batch)
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
