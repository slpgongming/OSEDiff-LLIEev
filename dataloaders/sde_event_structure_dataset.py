"""原事件输入保持不变，额外建立以目标时刻为锚点的正负分离结构视图。"""

import argparse
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset


@dataclass(frozen=True)
class StructureEventConfig:
    time_unit: str
    window_before: int
    window_after: int
    bins: int = 5
    count_scale: float = 3.0
    reference: str = "filename"
    timestamp_offset: int = 0
    version: int = 1

    def __post_init__(self):
        if self.time_unit not in {"ms", "us", "ns"}:
            raise ValueError("Explicit event timestamp unit (ms/us/ns) is required.")
        for value in (self.window_before, self.window_after, self.timestamp_offset, self.bins):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("Event boundaries, offset and bins must be integers in the stated unit.")
        if min(self.window_before, self.window_after) < 0 or self.window_before + self.window_after <= 0:
            raise ValueError("A nonempty fixed time window is required.")
        if self.bins < 1 or not math.isfinite(self.count_scale) or self.count_scale <= 0 or self.version != 1:
            raise ValueError("Invalid structure voxel configuration.")
        if self.reference not in {"filename", "manifest"}:
            raise ValueError("reference must be filename or manifest.")


def structure_voxel(events, t_ref, geometry, config):
    """固定时间 bin 后裁空间，输出 [2*bins,H,W]；前 bins 为正、后 bins 为负。

    必须提供原始 [t,x,y,p]。绝对时间保持 int64，减去窗口起点后才转换为浮点。
    裁剪区域内是否有首尾事件不会改变 bin 的时间边界。
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Structure guidance needs raw [t,x,y,p], not a signed/precomputed voxel.")
    if not torch.isfinite(events).all():
        raise ValueError("Non-finite raw events.")
    if events.is_floating_point() and events.numel():
        if events.dtype != torch.float64 or not torch.equal(events[:, 0], events[:, 0].round()):
            raise ValueError("Raw timestamps must retain integer ticks (int64 or exact float64).")
        if events[:, 0].abs().max() >= 2 ** 53:
            raise ValueError("Float64 timestamps exceed exact integer range; supply int64.")
    start = int(t_ref) - config.window_before
    duration = config.window_before + config.window_after
    ticks = events[:, 0].long() - start
    x = (events[:, 1].double() * geometry["scale"]).floor().long() - geometry["left"]
    y = (events[:, 2].double() * geometry["scale"]).floor().long() - geometry["top"]
    h, w = geometry["crop_h"], geometry["crop_w"]
    valid = (ticks >= 0) & (ticks < duration) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    polarity = events[:, 3]
    if not ((polarity == -1) | (polarity == 0) | (polarity == 1)).all():
        raise ValueError("Raw polarity must be -1/+1 or 0/1.")
    # 时间分箱在相对 int64 ticks 上计算，正负绝不相抵消。
    bins = torch.div(ticks[valid] * config.bins, duration, rounding_mode="floor").long()
    channels = bins + (polarity[valid] <= 0).long() * config.bins
    voxel = torch.zeros(2 * config.bins, h, w, dtype=torch.float32)
    voxel.index_put_((channels.cpu(), y[valid].cpu(), x[valid].cpu()), torch.ones(int(valid.sum())), accumulate=True)
    if geometry["flip"]:
        voxel = voxel.flip(-1)
    size = geometry["image_size"]
    if (h, w) != (size, size):
        voxel = F.interpolate(voxel[None], (size, size), mode="nearest")[0]
    return (voxel.log1p() / config.count_scale).clamp(0, 1)


class SDEEventStructureDataset(SDEEventFilmDataset):
    """共享原数据集的几何参数；新时间表示不改动原 cLoRA/refinement 视图。"""
    def __init__(self, split, args, config, root_dir=None, scenes=None):
        copied = argparse.Namespace(**vars(args))
        copied.return_geometry = True
        super().__init__(split, copied, root_dir)
        if scenes is not None:
            self.samples = [sample for sample in self.samples if sample.scene_name in set(scenes)]
            if not self.samples:
                raise ValueError("No samples in requested scene split.")
        if self.event_ablation_mode not in {"correct", "zero"}:
            raise ValueError("Structure dataset accepts correct/zero; perform mismatch ablations explicitly after loading.")
        self.structure_config = config
        self.references = {}
        if config.reference == "manifest":
            path = getattr(args, "structure_manifest", None)
            if not path:
                raise ValueError("--structure_manifest is required for reference=manifest.")
            with open(path, newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    key = (row["scene"], row["frame_id"])
                    if key in self.references or row["time_unit"] != config.time_unit:
                        raise ValueError("Duplicate manifest frame or timestamp unit mismatch.")
                    self.references[key] = int(row["t_ref"])

    def __getitem__(self, index):
        output = super().__getitem__(index)
        sample = self.samples[index]
        config = self.structure_config
        if config.reference == "filename":
            try:
                t_ref = int(Path(sample.low_path).stem) + config.timestamp_offset
            except ValueError as exc:
                raise ValueError("Non-numeric frame name: use a verified timestamp manifest.") from exc
        else:
            t_ref = self.references[(sample.scene_name, output["frame_id"])] + config.timestamp_offset
        # 完整事件文件优先，避免已有按相邻帧切片提前丢掉结构窗口中的事件。
        path = sample.whole_event_path or sample.split_event_path
        if path is None:
            raise FileNotFoundError(f"No raw event file for {sample.low_path}")
        raw = self._read_npz_event(path)
        structure = structure_voxel(raw, t_ref, output["geometry"], config)
        if self.event_ablation_mode == "zero":
            structure.zero_()
        output["events_structure"] = structure
        output["structure_meta"] = dict(
            **asdict(config), t_ref=t_ref, t_start=t_ref-config.window_before,
            t_end=t_ref+config.window_after, source=str(path),
            # 首/末事件不是传感器记录边界；如无录制元数据，不能声称窗口覆盖已验证。
            coverage="recording_bounds_unverified", geometry=output["geometry"],
        )
        return output
