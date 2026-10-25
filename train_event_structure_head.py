"""idea8 事件结构/置信度两阶段训练；不加载或训练 OSEDiff 生成器。"""

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataloaders.sde_event_structure_dataset import SDEEventStructureDataset, StructureEventConfig
from models.event_structure_head import (EventStructurePredictor, StructureHeadConfig,
                                        load_structure_checkpoint, structure_checkpoint)
from models.image_structure import StructureOperatorConfig, image_structure


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--scene_split", required=True, help="JSON: train/calibration/validation 三组互斥场景")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--phase", choices=("structure", "confidence"), default="structure")
    p.add_argument("--resume_from_checkpoint")
    p.add_argument("--init_structure_checkpoint")
    p.add_argument("--structure_mode", choices=("idea4_compatible", "fixed_window"),
                   help="新训练默认 idea4_compatible；续训/置信度训练默认继承 checkpoint。")
    p.add_argument("--structure_time_unit", choices=("ms", "us", "ns"))
    p.add_argument("--structure_window_before", type=int)
    p.add_argument("--structure_window_after", type=int)
    p.add_argument("--structure_reference", choices=("filename", "manifest"))
    p.add_argument("--structure_timestamp_offset", type=int)
    p.add_argument("--structure_manifest")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4, help="每个进程/GPU 的 batch size；全局 batch 为此值乘 world_size。")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--confidence_tau", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=50,
                   help="每多少个训练/验证 batch 输出进度（默认 50）；仅 rank 0 输出，首/末 batch 也输出。")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.log_interval < 1:
        p.error("log_interval must be a positive integer.")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or any(
        not math.isfinite(value) or value <= 0 for value in (args.confidence_tau, args.learning_rate)
    ):
        p.error("epochs/batch_size/learning_rate/confidence_tau must be positive.")
    if args.resume_from_checkpoint and args.init_structure_checkpoint:
        p.error("Use resume or weights-only initialization, not both.")
    if args.image_size != 256:
        p.error("Initial idea8 preprocessing/checkpoints use image_size=256.")
    return args


class StepProgress:
    """主进程的阶段进度；只读取本 rank 的指标，不在不等长验证循环里增加通信。"""

    def __init__(self, args, rank, epoch, stage, steps):
        self.enabled = rank == 0
        self.interval = args.log_interval
        self.steps = steps
        self.prefix = f"[Progress] phase={args.phase} stage={stage} epoch={epoch + 1}/{args.epochs}"
        self.started = time.perf_counter()
        if self.enabled:
            # 在请求首批数据前输出，让耗时的数据读取/worker 启动也有明确阶段提示。
            print(f"{self.prefix} step=0/{steps} status=starting", flush=True)

    def update(self, step, loss, loss_sum, samples):
        if not self.enabled or (step != 1 and step % self.interval and step != self.steps):
            return
        # 仅打印时读回 GPU 标量；不改变 RNG、训练图或 epoch 末的全局指标汇总。
        current = loss.detach().item()
        mean = (loss_sum / samples).item()
        elapsed = time.perf_counter() - self.started
        seconds_per_step = elapsed / step
        eta = seconds_per_step * (self.steps - step)
        print(f"{self.prefix} step={step}/{self.steps} ({100 * step / self.steps:.1f}%) "
              f"loss_rank0={current:.6f} avg_loss_rank0={mean:.6f} "
              f"elapsed={timedelta(seconds=int(elapsed))} eta={timedelta(seconds=int(eta))} "
              f"sec_per_step={seconds_per_step:.3f}", flush=True)


def read_scene_split(path):
    with open(path, encoding="utf-8") as handle:
        split = json.load(handle)
    seen = set()
    for key in ("train", "calibration", "validation"):
        scenes = split.get(key)
        if not isinstance(scenes, list) or not scenes or len(set(scenes)) != len(scenes):
            raise ValueError(f"{key} must contain unique scene names.")
        if seen.intersection(scenes):
            raise ValueError("Scene splits overlap; adjacent-frame leakage is forbidden.")
        seen.update(scenes)
    return split


def collate_structure(samples):
    # 训练只保留所需张量；源时间戳与几何 metadata 不由默认 collate 混排。
    return {key: torch.stack([s[key] for s in samples])
            for key in ("events_structure", "output_pixel_values")}


def phase_loss(model, batch, phase, operator, tau, device):
    events = batch["events_structure"].to(device)
    gt = batch["output_pixel_values"].to(device).float() * 0.5 + 0.5
    target = image_structure(gt, operator)
    edge, confidence = model(events)
    error = (edge - target).abs()
    if phase == "structure":
        # 结构监督不乘预测置信度，防止通过 C→0 逃避训练。
        loss = F.smooth_l1_loss(edge, target, beta=0.1)
    else:
        expected_confidence = torch.exp(-error.detach() / tau)
        loss = F.mse_loss(confidence, expected_confidence)
    confidence_target = torch.exp(-error.detach() / tau)
    return loss, error.detach().mean(), (confidence.detach() - confidence_target).abs().mean()


def setup_distributed(args):
    """torchrun 一进程一卡；CPU/Gloo 路径用于真实多进程回归。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid distributed rank/world size.")
    device = torch.device(args.device)
    if device.type == "cuda":
        index = local_rank if world_size > 1 else (device.index or 0)
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError("Not enough visible CUDA devices for this process; check CUDA_VISIBLE_DEVICES.")
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    elif device.type != "cpu":
        raise ValueError("Structure training supports cuda or cpu devices.")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                init_method="env://", timeout=timedelta(minutes=5))
    return rank, world_size, device


def capture_rng(device):
    # 每个 rank 分别保存随机增强、采样和当前 GPU 的随机状态。
    return dict(random_state=random.getstate(), numpy_state=np.random.get_state(),
                torch_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(checkpoint, rank, world_size, device):
    if checkpoint.get("world_size", 1) != world_size:
        raise ValueError("Resume world_size mismatch; use init_structure_checkpoint for weights-only initialization.")
    states = checkpoint.get("rng_states")
    if states is not None and len(states) != world_size:
        raise ValueError("Checkpoint lacks RNG state for every rank.")
    state = states[rank] if states is not None else checkpoint
    random.setstate(state["random_state"])
    np.random.set_state(state["numpy_state"])
    torch.set_rng_state(state["torch_state"])
    if device.type == "cuda":
        if state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"], device)
        elif states is None and checkpoint.get("cuda_state") is not None:
            # 兼容此前单进程 checkpoint 的 get_rng_state_all 格式。
            torch.cuda.set_rng_state(checkpoint["cuda_state"][device.index], device)


def main(args):
    rank, world_size, device = setup_distributed(args)
    try:
        train_main(args, rank, world_size, device)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def train_main(args, rank, world_size, device):
    distributed = world_size > 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split = read_scene_split(args.scene_split)
    for scene in {s for scenes in split.values() for s in scenes}:
        if not (Path(args.data_root) / scene).is_dir():
            raise FileNotFoundError(f"Scene not found: {scene}")
    source = args.resume_from_checkpoint or args.init_structure_checkpoint
    checkpoint = None
    if source:
        model, checkpoint, operator = load_structure_checkpoint(source, require_calibrated=False)
        event_config = StructureEventConfig.from_dict(checkpoint["event_config"])
        if checkpoint["scene_split"] != split:
            raise ValueError("Changing scene splits across training phases/resume is forbidden.")
        for name in ("mode", "time_unit", "window_before", "window_after", "reference", "timestamp_offset"):
            requested = getattr(args, "structure_" + name)
            if requested is not None and requested != getattr(event_config, name):
                raise ValueError(f"Checkpoint event {name} mismatch.")
        if args.resume_from_checkpoint and checkpoint["phase"] != args.phase:
            raise ValueError("Resume must keep phase; use init_structure_checkpoint for calibration.")
    else:
        if args.phase == "confidence":
            raise ValueError("Confidence phase requires --init_structure_checkpoint.")
        event_config = StructureEventConfig(args.structure_time_unit, args.structure_window_before,
                                            args.structure_window_after, reference=args.structure_reference or "filename",
                                            timestamp_offset=args.structure_timestamp_offset or 0,
                                            mode=args.structure_mode or "idea4_compatible")
        model = EventStructurePredictor(StructureHeadConfig(base_channels=args.base_channels))
        operator = StructureOperatorConfig()
    model.to(device).set_training_phase(args.phase)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    start_epoch = 0
    resume_config = dict(batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
                         device_type=device.type)
    if args.resume_from_checkpoint:
        if checkpoint.get("world_size", 1) != world_size:
            raise ValueError("Resume world_size mismatch; use init_structure_checkpoint for weights-only initialization.")
        if checkpoint.get("training_config", resume_config) != resume_config:
            raise ValueError("Resume must keep per-rank batch_size, num_workers, seed and device type.")
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if checkpoint["confidence_tau"] != args.confidence_tau:
            raise ValueError("Resume must keep confidence_tau.")
    # phase 在包装前确定，只有当前阶段的可训练参数参与梯度同步。
    training_model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                            broadcast_buffers=False) if distributed else model
    if distributed:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
    train_scenes = split["train" if args.phase == "structure" else "calibration"]
    train = SDEEventStructureDataset("train" if args.phase == "structure" else "test",
                                     args, event_config, args.data_root, train_scenes)
    valid = SDEEventStructureDataset("test", args, event_config, args.data_root, split["validation"])
    # 训练 sampler 补齐到 world_size 的倍数，确保每个 rank 步数与 batch 大小一致。
    # 非整除时最多重复 world_size-1 个样本；统计按实际处理次数汇总。
    sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, seed=args.seed,
                                 shuffle=True, drop_last=False) if distributed else None
    # NCCL 初始化后不能再用 Linux 默认 fork 创建数据 worker；spawn 避免继承通信状态。
    worker_options = {"multiprocessing_context": "spawn"} if distributed and args.num_workers > 0 else {}
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                        num_workers=args.num_workers, collate_fn=collate_structure, pin_memory=device.type == "cuda",
                        **worker_options)
    # 验证不补齐、不重复；某些 rank 可以没有样本，使用未包装模型避免不等长 forward 的通信死锁。
    validation_loader = DataLoader(valid, batch_size=args.batch_size, sampler=range(rank, len(valid), world_size),
                                   num_workers=args.num_workers, collate_fn=collate_structure,
                                   pin_memory=device.type == "cuda", **worker_options)
    if args.resume_from_checkpoint:
        restore_rng(checkpoint, rank, world_size, device)
    output = Path(args.output_dir)
    # 同时保存实际解析后的预处理配置，尤其是从 checkpoint 继承的 mode/窗口。
    run_config = dict(vars(args), structure_event_config=asdict(event_config), world_size=world_size,
                      global_batch_size=args.batch_size * world_size, device_resolved=str(device),
                      training_samples_per_epoch=len(sampler) * world_size if sampler is not None else len(train),
                      validation_samples=len(valid))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Train] phase={args.phase} world_size={world_size} per_rank_batch={args.batch_size} "
              f"global_batch={args.batch_size * world_size} log_interval={args.log_interval}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        epoch_label = f"phase={args.phase} epoch={epoch + 1}/{args.epochs}"
        if sampler is not None:
            sampler.set_epoch(epoch)
        training_model.train()
        total = torch.zeros(2, dtype=torch.float64, device=device)
        progress = StepProgress(args, rank, epoch, "train", len(loader))
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = phase_loss(training_model, batch, args.phase, operator, args.confidence_tau, device)
            finite = torch.isfinite(loss).to(dtype=torch.int32)
            if distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite structure training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                          error_if_nonfinite=True)
            optimizer.step()
            count = batch["events_structure"].shape[0]
            total[0] += loss.detach().double() * count
            total[1] += count
            progress.update(step, loss, total[0], total[1])
        model.eval()
        values = torch.zeros(4, dtype=torch.float64, device=device)
        progress = StepProgress(args, rank, epoch, "validation", len(validation_loader))
        with torch.no_grad():
            for step, batch in enumerate(validation_loader, 1):
                loss, edge_error, calibration_error = phase_loss(
                    model, batch, args.phase, operator, args.confidence_tau, device)
                count = batch["events_structure"].shape[0]
                values[:3] += torch.stack([loss, edge_error, calibration_error]).double() * count
                values[3] += count
                progress.update(step, loss, values[0], values[3])
        if rank == 0:
            print(f"[Epoch] {epoch_label} status=aggregating_metrics_and_rng", flush=True)
        if distributed:
            dist.all_reduce(total)
            dist.all_reduce(values)
        if int(values[3].item()) != len(valid):
            raise RuntimeError("Distributed validation must count each sample exactly once.")
        values[:3] /= values[3].clone()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite structure validation metrics; checkpoint not saved.")
        row = dict(epoch=epoch, train_loss=float(total[0]/total[1]), validation_loss=float(values[0]),
                   edge_mae=float(values[1]), confidence_mae=float(values[2]),
                   train_samples=int(total[1].item()), validation_samples=int(values[3].item()))
        rng_state = capture_rng(device)
        rng_states = [None] * world_size
        if distributed:
            dist.all_gather_object(rng_states, rng_state)
        else:
            rng_states[0] = rng_state
        if rank == 0:
            print(json.dumps(row), flush=True)
            save_started = time.perf_counter()
            print(f"[Checkpoint] {epoch_label} status=saving path={output / 'checkpoint_last.pt'}", flush=True)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            saved = structure_checkpoint(
                model, operator, event_config, phase=args.phase, epoch=epoch, optimizer=optimizer.state_dict(),
                confidence_trained=args.phase == "confidence", confidence_tau=args.confidence_tau,
                scene_split=split, image_size=args.image_size, metrics=row,
                world_size=world_size, training_config=resume_config, rng_states=rng_states,
                **rng_states[0],
            )
            # 只由主进程原子保存；保存的是原模型，无 module. 前缀，推理无需 DDP。
            temporary = output / "checkpoint.tmp"
            torch.save(saved, temporary)
            os.replace(temporary, output / "checkpoint_last.pt")
            print(f"[Checkpoint] {epoch_label} status=saved elapsed={time.perf_counter() - save_started:.2f}s",
                  flush=True)
        if distributed:
            if rank == 0:
                print(f"[Epoch] {epoch_label} status=waiting_for_ranks", flush=True)
            dist.barrier()
        if rank == 0:
            print(f"[Epoch] {epoch_label} status=complete elapsed={time.perf_counter() - epoch_started:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main(parse_args())
