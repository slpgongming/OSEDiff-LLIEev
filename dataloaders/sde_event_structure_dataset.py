"""原事件输入保持不变，新增 idea4 兼容/固定窗口两种正负分离结构视图。"""

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
    time_unit: str = None
    window_before: int = None
    window_after: int = None
    bins: int = 5
    count_scale: float = 3.0
    reference: str = "filename"
    timestamp_offset: int = 0
    version: int = 2
    mode: str = "idea4_compatible"

    def __post_init__(self):
        if self.mode not in {"idea4_compatible", "fixed_window"}:
            raise ValueError("Unknown structure event mode.")
        for value in (self.timestamp_offset, self.bins):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("Event boundaries, offset and bins must be integers in the stated unit.")
        if self.bins < 1 or not math.isfinite(self.count_scale) or self.count_scale <= 0 or self.version not in {1, 2}:
            raise ValueError("Invalid structure voxel configuration.")
        if self.reference not in {"filename", "manifest"}:
            raise ValueError("reference must be filename or manifest.")
        if self.mode == "idea4_compatible":
            # 兼容模式无固定时间窗口，拒绝遗留参数，避免用户误以为窗口仍生效。
            if self.version != 2 or any(v is not None for v in (self.time_unit, self.window_before, self.window_after)) or self.reference != "filename" or self.timestamp_offset != 0:
                raise ValueError("idea4_compatible does not accept fixed-window time/reference settings.")
        else:
            if self.time_unit not in {"ms", "us", "ns"}:
                raise ValueError("Explicit event timestamp unit (ms/us/ns) is required.")
            if any(not isinstance(v, int) or isinstance(v, bool) for v in (self.window_before, self.window_after)):
                raise ValueError("Fixed window boundaries must be integer ticks.")
            if min(self.window_before, self.window_after) < 0 or self.window_before + self.window_after <= 0:
                raise ValueError("A nonempty fixed time window is required.")

    @classmethod
    def from_dict(cls, values):
        # 旧 v1 checkpoint 缺少 mode，只能解释为原固定窗口，绝不能套用新默认值。
        values = dict(values)
        if "mode" not in values:
            if values.get("version", 1) != 1:
                raise ValueError("Structure event mode missing from versioned checkpoint.")
            values.update(mode="fixed_window", version=1)
        return cls(**values)


def structure_voxel(events, t_ref, geometry, config):
    """输出 [2*bins,H,W]；前 bins 为正、后 bins 为负。

    必须提供原始 [t,x,y,p]。绝对时间保持 int64，减去窗口起点后才转换为浮点。
    兼容模式复用 idea4：空间裁剪后首/末事件决定时间分箱；固定窗口模式边界不变。
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
    if events.shape[0] > 1 and (events[1:, 0] < events[:-1, 0]).any():
        raise ValueError("Raw events must be sorted by timestamp.")
    if config.mode == "idea4_compatible":
        # 使用原实现的缩放/类型转换顺序，保证整数坐标在缩放后与生成器视图一致。
        scaled = SDEEventFilmDataset._scale_event(None, events, geometry["scale"])
        x, y = scaled[:, 1] - geometry["left"], scaled[:, 2] - geometry["top"]
    else:
        x = (events[:, 1].double() * geometry["scale"]).floor().long() - geometry["left"]
        y = (events[:, 2].double() * geometry["scale"]).floor().long() - geometry["top"]
    h, w = geometry["crop_h"], geometry["crop_w"]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    polarity = events[:, 3]
    if not ((polarity == -1) | (polarity == 0) | (polarity == 1)).all():
        raise ValueError("Raw polarity must be -1/+1 or 0/1.")
    if config.mode == "idea4_compatible":
        cropped_ticks = events[valid, 0].long()
        if cropped_ticks.numel():
            relative = cropped_ticks - cropped_ticks[0]
            # 与 idea4 的 float32 比值、末事件 clamp 完全一致；先相减保留绝对时间精度。
            duration = (cropped_ticks[-1] - cropped_ticks[0]).float().clamp_min(1.0)
            bins = (relative.float() / duration * config.bins).long().clamp(0, config.bins - 1)
        else:
            bins = torch.empty(0, dtype=torch.long, device=events.device)
    else:
        start = int(t_ref) - config.window_before
        duration = config.window_before + config.window_after
        ticks = events[:, 0].long() - start
        valid &= (ticks >= 0) & (ticks < duration)
        bins = torch.div(ticks[valid] * config.bins, duration, rounding_mode="floor").long()
    channels = bins + (polarity[valid] <= 0).long() * config.bins
    voxel = torch.zeros(2 * config.bins, h, w, dtype=torch.float32)
    voxel.index_put_((channels.cpu(), y[valid].long().cpu(), x[valid].long().cpu()), torch.ones(int(valid.sum())), accumulate=True)
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
        if config.mode == "idea4_compatible" and getattr(args, "structure_manifest", None):
            raise ValueError("structure_manifest is only used with fixed_window mode.")
        if config.mode == "fixed_window" and config.reference == "manifest":
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
        if config.mode == "idea4_compatible":
            # 同一条父类加载路径：优先分片；无分片时按相邻帧截取完整流。
            path = sample.split_event_path or sample.whole_event_path
            if path is None:
                raise FileNotFoundError(f"No raw event file for {sample.low_path}")
            raw = self._load_event_for_sample(sample)
            t_ref = None
        elif config.reference == "filename":
            try:
                t_ref = int(Path(sample.low_path).stem) + config.timestamp_offset
            except ValueError as exc:
                raise ValueError("Non-numeric frame name: use a verified timestamp manifest.") from exc
        else:
            t_ref = self.references[(sample.scene_name, output["frame_id"])] + config.timestamp_offset
        if config.mode == "fixed_window":
            # 固定窗口保留旧行为：完整流优先，避免预分片提前丢掉所需事件。
            path = sample.whole_event_path or sample.split_event_path
            if path is None:
                raise FileNotFoundError(f"No raw event file for {sample.low_path}")
            raw = self._read_npz_event(path)
        structure = structure_voxel(raw, t_ref, output["geometry"], config)
        if self.event_ablation_mode == "zero":
            structure.zero_()
        output["events_structure"] = structure
        output["structure_meta"] = dict(
            **asdict(config), t_ref=t_ref,
            t_start=t_ref-config.window_before if config.mode == "fixed_window" else None,
            t_end=t_ref+config.window_after if config.mode == "fixed_window" else None,
            source=str(path),
            source_selection=("split" if sample.split_event_path else "neighbor_frames") if config.mode == "idea4_compatible" else "fixed_window",
            # 首/末事件不是传感器记录边界；如无录制元数据，不能声称窗口覆盖已验证。
            coverage="recording_bounds_unverified", geometry=output["geometry"],
        )
        return output
