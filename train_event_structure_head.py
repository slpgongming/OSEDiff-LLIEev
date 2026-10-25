"""idea8 事件结构/置信度两阶段训练；不加载或训练 OSEDiff 生成器。"""

import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    p.add_argument("--structure_time_unit", choices=("ms", "us", "ns"))
    p.add_argument("--structure_window_before", type=int)
    p.add_argument("--structure_window_after", type=int)
    p.add_argument("--structure_reference", choices=("filename", "manifest"))
    p.add_argument("--structure_timestamp_offset", type=int)
    p.add_argument("--structure_manifest")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--confidence_tau", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or any(
        not math.isfinite(value) or value <= 0 for value in (args.confidence_tau, args.learning_rate)
    ):
        p.error("epochs/batch_size/learning_rate/confidence_tau must be positive.")
    if args.resume_from_checkpoint and args.init_structure_checkpoint:
        p.error("Use resume or weights-only initialization, not both.")
    if args.image_size != 256:
        p.error("Initial idea8 preprocessing/checkpoints use image_size=256.")
    return args


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


def main(args):
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
        event_config = StructureEventConfig(**checkpoint["event_config"])
        if checkpoint["scene_split"] != split:
            raise ValueError("Changing scene splits across training phases/resume is forbidden.")
        for name in ("time_unit", "window_before", "window_after", "reference", "timestamp_offset"):
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
                                            timestamp_offset=args.structure_timestamp_offset or 0)
        model = EventStructurePredictor(StructureHeadConfig(base_channels=args.base_channels))
        operator = StructureOperatorConfig()
    model.to(args.device).set_training_phase(args.phase)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    start_epoch = 0
    if args.resume_from_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_state"])
        if checkpoint["confidence_tau"] != args.confidence_tau:
            raise ValueError("Resume must keep confidence_tau.")
    train_scenes = split["train" if args.phase == "structure" else "calibration"]
    train = SDEEventStructureDataset("train" if args.phase == "structure" else "test",
                                     args, event_config, args.data_root, train_scenes)
    valid = SDEEventStructureDataset("test", args, event_config, args.data_root, split["validation"])
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_structure)
    validation_loader = DataLoader(valid, batch_size=args.batch_size, num_workers=args.num_workers,
                                   collate_fn=collate_structure)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = phase_loss(model, batch, args.phase, operator, args.confidence_tau, args.device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite structure training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                          error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * batch["events_structure"].shape[0]
        model.eval()
        values = torch.zeros(3)
        with torch.no_grad():
            for batch in validation_loader:
                loss, edge_error, calibration_error = phase_loss(
                    model, batch, args.phase, operator, args.confidence_tau, args.device)
                values += torch.tensor([float(loss), float(edge_error), float(calibration_error)]) * batch["events_structure"].shape[0]
        values /= len(valid)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite structure validation metrics; checkpoint not saved.")
        row = dict(epoch=epoch, train_loss=total/len(train), validation_loss=float(values[0]),
                   edge_mae=float(values[1]), confidence_mae=float(values[2]))
        print(json.dumps(row), flush=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        saved = structure_checkpoint(
            model, operator, event_config, phase=args.phase, epoch=epoch, optimizer=optimizer.state_dict(),
            confidence_trained=args.phase == "confidence", confidence_tau=args.confidence_tau,
            scene_split=split, image_size=args.image_size, metrics=row,
            random_state=random.getstate(), numpy_state=np.random.get_state(), torch_state=torch.get_rng_state(),
            cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )
        # 原子保存 epoch 边界状态，支持完整恢复 RNG/optimizer；不误称逐 batch 精确恢复。
        temporary = output / "checkpoint.tmp"
        torch.save(saved, temporary)
        os.replace(temporary, output / "checkpoint_last.pt")


if __name__ == "__main__":
    main(parse_args())
