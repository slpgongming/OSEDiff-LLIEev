"""idea8 事件结构/置信度两阶段训练；不加载或训练 OSEDiff 生成器。"""

import argparse
from dataclasses import asdict, dataclass, replace
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
from models.structure_supervision import (StructureSupervisionConfig, StructureValidationAccumulator,
                                          structure_supervision_loss)


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
    p.add_argument("--head_variant", choices=("half", "full_detail"),
                   help="新训练默认 half；H2/H3 用 full_detail；加载 checkpoint 时继承且禁止跨结构加载。")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4, help="每个进程/GPU 的 batch size；全局 batch 为此值乘 world_size。")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--confidence_tau", type=float, default=0.1)
    p.add_argument("--structure_loss", choices=("smooth_l1", "region_balanced"),
                   help="H0 用 smooth_l1（新训练默认）；H1 用 region_balanced；续训默认继承权重元信息。")
    p.add_argument("--region_thresholds", type=float, nargs=2, metavar=("WEAK", "STRONG"),
                   help="固定 GT 结构分区阈值，初值 0.05/0.2；正式对照前按训练集统计确认。")
    p.add_argument("--region_weights", type=float, nargs=3, metavar=("FLAT", "WEAK", "STRONG"),
                   help="H1 的区域权重，默认 1/1/1，三个区域均须保留正权重。")
    p.add_argument("--best_metric", choices=("auto", "balanced_mae", "validation_loss"),
                   help="auto：结构阶段按区域等权 MAE 选 best；置信度阶段按验证 MSE；续训默认继承。")
    p.add_argument("--validation_border", type=int, help="结构验证去除的边界宽度，默认 4；不改变训练标签。")
    p.add_argument("--edge_tolerance", type=int, help="边缘匹配允许的像素位移，默认 2。")
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
    try:
        StructureSupervisionConfig(mode=args.structure_loss or "smooth_l1",
                                   thresholds=args.region_thresholds or (0.05, 0.2),
                                   region_weights=args.region_weights or (1., 1., 1.))
        SelectionConfig(metric="validation_loss" if args.phase == "confidence" else "balanced_mae",
                        border=args.validation_border if args.validation_border is not None else 4,
                        tolerance=args.edge_tolerance if args.edge_tolerance is not None else 2)
    except ValueError as exc:
        p.error(str(exc))
    return args


@dataclass(frozen=True)
class SelectionConfig:
    version: int = 1
    metric: str = "balanced_mae"
    border: int = 4
    tolerance: int = 2

    def __post_init__(self):
        if self.version != 1 or self.metric not in {"balanced_mae", "validation_loss"}:
            raise ValueError("Invalid best-checkpoint selection configuration.")
        if not isinstance(self.border, int) or not 0 <= self.border < 128:
            raise ValueError("validation_border must be in [0,128).")
        if not isinstance(self.tolerance, int) or not 0 <= self.tolerance < 256:
            raise ValueError("edge_tolerance must be in [0,256).")


def resolve_training_configs(args, checkpoint):
    """显式参数覆盖仅权重初始化；完整续训及置信度阶段不能静默改变结构监督定义。"""
    checkpoint = checkpoint or {}
    original = StructureSupervisionConfig(**checkpoint.get("supervision_config", {}))
    supervision = replace(original,
                          mode=args.structure_loss or original.mode,
                          thresholds=tuple(args.region_thresholds) if args.region_thresholds is not None else original.thresholds,
                          region_weights=tuple(args.region_weights) if args.region_weights is not None else original.region_weights)
    if checkpoint and (args.resume_from_checkpoint or args.phase == "confidence") and supervision != original:
        raise ValueError("Resume/calibration must keep structure supervision; use structure weights-only initialization to change it.")
    phase_metric = "validation_loss" if args.phase == "confidence" else "balanced_mae"
    inherited = SelectionConfig(**checkpoint.get("selection_config", {}))
    metric = inherited.metric if checkpoint.get("phase") == args.phase and checkpoint.get("selection_config") else phase_metric
    if args.best_metric is not None:
        metric = phase_metric if args.best_metric == "auto" else args.best_metric
    selection = replace(inherited, metric=metric,
                        border=args.validation_border if args.validation_border is not None else inherited.border,
                        tolerance=args.edge_tolerance if args.edge_tolerance is not None else inherited.tolerance)
    if args.phase == "confidence" and selection.metric != "validation_loss":
        raise ValueError("Confidence best selection must use validation_loss; the structure branch is frozen.")
    if args.resume_from_checkpoint and checkpoint.get("selection_config") and selection != inherited:
        raise ValueError("Resume must keep best metric, validation border and edge tolerance.")
    return supervision, selection


def atomic_checkpoint_save(payload, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resume_selection_state(checkpoint, resume_path, output, selection):
    """保留历史 best；迁移续训目录时从原 best 恢复，不把 last 冒充 best。"""
    start = checkpoint["epoch"] + 1 if resume_path else 0
    state = dict(checkpoint.get("selection_state", {})) if resume_path else {}
    if not state:
        # 旧 checkpoint 没有历史 best 权重，只从本次实际验证开始记录新的选择窗口。
        return dict(metric=selection.metric, best_score=None, best_epoch=None, history_start_epoch=start), None
    if state.get("metric") != selection.metric:
        raise ValueError("Checkpoint selection history does not match the requested metric.")
    if state.get("best_epoch") is None:
        return state, None
    if not math.isfinite(state["best_score"]) or not 0 <= state["best_epoch"] <= checkpoint["epoch"]:
        raise ValueError("Invalid best-checkpoint history.")
    if state["best_epoch"] == checkpoint["epoch"]:
        best = checkpoint
    else:
        best = None
        for path in (Path(resume_path).parent / "checkpoint_best.pt", output / "checkpoint_best.pt"):
            if path.is_file():
                candidate = torch.load(path, map_location="cpu", weights_only=False)
                if (candidate.get("epoch") == state["best_epoch"] and candidate.get("phase") == checkpoint["phase"]
                        and candidate.get("selection_config") == asdict(selection)
                        and candidate.get("supervision_config") == checkpoint.get("supervision_config")
                        and candidate.get("scene_split") == checkpoint["scene_split"]
                        and candidate.get("metrics", {}).get(selection.metric) == state["best_score"]):
                    best = candidate
                    break
        if best is None:
            raise ValueError("Historical checkpoint_best.pt is missing or incompatible; preserve it alongside checkpoint_last.pt when resuming.")
    return state, best


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
    result = {key: torch.stack([s[key] for s in samples])
              for key in ("events_structure", "output_pixel_values")}
    result["scene_names"] = [s["scene_name"] for s in samples]
    return result


def phase_loss(model, batch, phase, operator, tau, device, supervision=None, return_details=False):
    events = batch["events_structure"].to(device)
    gt = batch["output_pixel_values"].to(device).float() * 0.5 + 0.5
    target = image_structure(gt, operator)
    edge, confidence = model(events)
    error = (edge - target).abs()
    if phase == "structure":
        # 结构监督不乘预测置信度，防止通过 C→0 逃避训练。
        loss = structure_supervision_loss(edge, target, supervision)
    else:
        expected_confidence = torch.exp(-error.detach() / tau)
        loss = F.mse_loss(confidence, expected_confidence)
    confidence_target = torch.exp(-error.detach() / tau)
    metrics = (loss, error.detach().mean(), (confidence.detach() - confidence_target).abs().mean())
    return (*metrics, dict(prediction=edge.detach(), target=target.detach())) if return_details else metrics


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
        if args.head_variant is not None and args.head_variant != model.config.variant:
            raise ValueError("Checkpoint head variant mismatch; start a new architecture experiment.")
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
        model = EventStructurePredictor(StructureHeadConfig(base_channels=args.base_channels,
                                                           variant=args.head_variant or "half"))
        operator = StructureOperatorConfig()
    supervision, selection = resolve_training_configs(args, checkpoint)
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
    selection_state, historical_best = resume_selection_state(checkpoint or {}, args.resume_from_checkpoint, output, selection)
    global_step = checkpoint.get("global_step", start_epoch * len(loader)) if args.resume_from_checkpoint else 0
    # 同时保存实际解析后的预处理配置，尤其是从 checkpoint 继承的 mode/窗口。
    run_config = dict(vars(args), structure_event_config=asdict(event_config), world_size=world_size,
                      global_batch_size=args.batch_size * world_size, device_resolved=str(device),
                      training_samples_per_epoch=len(sampler) * world_size if sampler is not None else len(train),
                      validation_samples=len(valid), supervision_config=asdict(supervision),
                      selection_config=asdict(selection), selection_history_start_epoch=selection_state["history_start_epoch"],
                      optimizer_learning_rates=[group["lr"] for group in optimizer.param_groups])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if historical_best is not None:
            atomic_checkpoint_save(historical_best, output / "checkpoint_best.pt")
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
            loss, _, _ = phase_loss(training_model, batch, args.phase, operator, args.confidence_tau, device,
                                    supervision=supervision)
            finite = torch.isfinite(loss).to(dtype=torch.int32)
            if distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite structure training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                          error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            count = batch["events_structure"].shape[0]
            total[0] += loss.detach().double() * count
            total[1] += count
            progress.update(step, loss, total[0], total[1])
        model.eval()
        values = torch.zeros(4, dtype=torch.float64, device=device)
        structure_metrics = StructureValidationAccumulator(len(operator.sigmas), supervision, split["validation"], device,
                                                           border=selection.border, tolerance=selection.tolerance)
        progress = StepProgress(args, rank, epoch, "validation", len(validation_loader))
        with torch.no_grad():
            for step, batch in enumerate(validation_loader, 1):
                loss, edge_error, calibration_error, details = phase_loss(
                    model, batch, args.phase, operator, args.confidence_tau, device,
                    supervision=supervision, return_details=True)
                count = batch["events_structure"].shape[0]
                values[:3] += torch.stack([loss, edge_error, calibration_error]).double() * count
                values[3] += count
                structure_metrics.update(details["prediction"], details["target"], batch["scene_names"])
                progress.update(step, loss, values[0], values[3])
        if rank == 0:
            print(f"[Epoch] {epoch_label} status=aggregating_metrics_and_rng", flush=True)
        if distributed:
            dist.all_reduce(total)
            dist.all_reduce(values)
        structure_metrics.synchronize()
        if int(values[3].item()) != len(valid):
            raise RuntimeError("Distributed validation must count each sample exactly once.")
        values[:3] /= values[3].clone()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite structure validation metrics; checkpoint not saved.")
        row = dict(epoch=epoch, train_loss=float(total[0]/total[1]), validation_loss=float(values[0]),
                   edge_mae=float(values[1]), confidence_mae=float(values[2]),
                   train_samples=int(total[1].item()), validation_samples=int(values[3].item()), global_step=global_step)
        diagnostic = structure_metrics.compute()
        if diagnostic["overall"]["samples"] != len(valid) or any(
                value is not None and not math.isfinite(value)
                for key, value in diagnostic["overall"].items() if key not in {"metric_counts"}):
            raise FloatingPointError("Invalid distributed structure diagnostics; checkpoint not saved.")
        row.update({key: value for key, value in diagnostic["overall"].items() if key != "samples"})
        row["by_scene"] = diagnostic["by_scene"]
        score = row[selection.metric]
        is_best = selection_state["best_score"] is None or score < selection_state["best_score"]
        if is_best:
            selection_state.update(best_score=score, best_epoch=epoch)
        row.update(selection_metric=selection.metric, selection_score=score, is_best=is_best,
                   best_score=selection_state["best_score"], best_epoch=selection_state["best_epoch"])
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
                supervision_config=asdict(supervision), selection_config=asdict(selection),
                selection_state=dict(selection_state), global_step=global_step,
                **rng_states[0],
            )
            # 只由主进程原子保存；保存的是原模型，无 module. 前缀，推理无需 DDP。
            atomic_checkpoint_save(saved, output / "checkpoint_last.pt")
            if is_best:
                atomic_checkpoint_save(saved, output / "checkpoint_best.pt")
                print(f"[BestCheckpoint] {epoch_label} metric={selection.metric} score={score:.8f} "
                      f"path={output / 'checkpoint_best.pt'}", flush=True)
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
