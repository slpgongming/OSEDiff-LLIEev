import os
import glob
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


@dataclass
class EventFilmSample:
    low_path: str
    normal_path: str
    scene_name: str
    frame_index: int
    split_event_path: str = None
    whole_event_path: str = None


class SDEEventFilmDataset(Dataset):
    """
    SDE 室内低光增强数据集。

    作用：
    - 按 scene/low 与 scene/normal 中排序后的 index 配对 RGB/GT；
    - 参考 references/eglol.py，从 low 下的事件 .npz 读取事件流；
    - 将事件裁剪到和 RGB 相同的空间区域，再体素化为 [C_event,H,W]。

    注意：
    这里默认采用 EGLol 风格的有符号事件体素：polarity=0 视为 -1，
    正负事件累加到同一个时间 bin，因此默认 event_channels=event_bins=5。
    """

    def __init__(self, split="train", args=None, root_dir=None):
        super().__init__()
        self.args = args
        self.split = split
        self.root_dir = root_dir or getattr(args, "train_data_root")
        self.image_size = int(getattr(args, "image_size", 256))
        self.event_bins = int(getattr(args, "event_bins", 5))
        self.event_channels = int(getattr(args, "event_channels", self.event_bins))
        if self.event_channels != self.event_bins:
            raise ValueError(
                "当前实现参考 references/eglol.py，要求 event_channels == event_bins。"
            )
        self.fixed_prompt = getattr(args, "fixed_prompt", "high quality, bright, detailed photo")
        self.neg_prompt = getattr(args, "neg_prompt", "")
        self.event_ablation_mode = getattr(args, "event_ablation_mode", "correct")
        if self.event_ablation_mode not in {"correct", "zero", "shuffled"}:
            raise ValueError(
                f"Unsupported event_ablation_mode={self.event_ablation_mode}."
            )
        self.return_raw_event_voxel = bool(
            getattr(args, "return_raw_event_voxel", False)
        )
        self.samples = self._scan_samples(self.root_dir)
        if not self.samples:
            raise RuntimeError(f"No SDE samples found under: {self.root_dir}")
        print(f"[SDEEventFilmDataset] Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        low_img = Image.open(sample.low_path).convert("RGB")
        normal_img = Image.open(sample.normal_path).convert("RGB")
        width, height = low_img.size

        raw_or_voxel = self._load_event_for_sample(sample)
        if self.event_ablation_mode == "shuffled":
            donor_index = self._shuffled_event_index(index)
            donor = self.samples[donor_index]
            raw_or_voxel = self._load_event_for_sample(donor)
            with Image.open(donor.low_path) as donor_image:
                donor_width, donor_height = donor_image.size
            raw_or_voxel = self._remap_event_extent(
                raw_or_voxel,
                source_width=donor_width,
                source_height=donor_height,
                target_width=width,
                target_height=height,
            )
        low_t = transforms.ToTensor()(low_img)
        normal_t = transforms.ToTensor()(normal_img)

        # 所有几何增强共享同一个 crop/flip 参数，保证 RGB、GT、event 对齐。
        low_t, normal_t, event_t = self._aligned_preprocess(
            low_t=low_t,
            normal_t=normal_t,
            raw_or_voxel=raw_or_voxel,
            width=width,
            height=height,
        )

        # RGB tensors use torchvision normalization: [0,1] -> [-1,1] per channel.
        # Keep torch.nn.functional as F for interpolation below; its normalize API is Lp-norm,
        # not image mean/std normalization, so using F.normalize here would crash.
        low_t = TF.normalize(low_t, mean=[0.5], std=[0.5])
        normal_t = TF.normalize(normal_t, mean=[0.5], std=[0.5])
        if self.event_ablation_mode == "zero":
            event_t = torch.zeros_like(event_t)
        raw_event_t = event_t
        events_clora = self._normalize_event_voxel(event_t)
        events_refinement = self._normalize_event_voxel_refinement(event_t)

        output = {
            "conditioning_pixel_values": low_t,
            "output_pixel_values": normal_t,
            "events": events_clora,
            "events_refinement": events_refinement,
            "prompt": [self.fixed_prompt],
            "neg_prompt": self.neg_prompt,
            "scene_name": sample.scene_name,
            "frame_id": os.path.splitext(os.path.basename(sample.low_path))[0],
            "frame_index": sample.frame_index,
            "low_path": sample.low_path,
            "normal_path": sample.normal_path,
        }
        if self.return_raw_event_voxel:
            output["event_voxel_raw"] = raw_event_t
        return output

    def _shuffled_event_index(self, index):
        """batch size=1 时也使用固定、可复现且不等于自身的 dataset 错配。"""
        if len(self.samples) < 2:
            raise RuntimeError("shuffled ablation requires at least two samples.")
        offset = max(1, len(self.samples) // 2)
        donor_index = (int(index) + offset) % len(self.samples)
        if donor_index == index:
            donor_index = (int(index) + 1) % len(self.samples)
        return donor_index

    def _remap_event_extent(
        self,
        event,
        source_width,
        source_height,
        target_width,
        target_height,
    ):
        """把错配事件映射到目标图像坐标系，随后复用目标 RGB 的 crop/flip。"""
        if event.ndim == 3:
            return F.interpolate(
                event.unsqueeze(0),
                size=(target_height, target_width),
                mode="nearest",
            )[0]
        event = event.clone()
        if event.numel() > 0:
            event[:, 1] *= target_width / max(float(source_width), 1.0)
            event[:, 2] *= target_height / max(float(source_height), 1.0)
        return event

    def _scan_samples(self, root_dir):
        samples = []
        for scene_name in sorted(os.listdir(root_dir)):
            scene_dir = os.path.join(root_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            low_dir = os.path.join(scene_dir, "low")
            normal_dir = os.path.join(scene_dir, "normal")
            if not (os.path.isdir(low_dir) and os.path.isdir(normal_dir)):
                continue

            low_files = sorted(glob.glob(os.path.join(low_dir, "*.png")))
            normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
            whole_event_path = os.path.join(low_dir, "lowlight_event.npz")
            split_event_files = sorted(
                p for p in glob.glob(os.path.join(low_dir, "*.npz"))
                if os.path.basename(p) not in {"lowlight_event.npz", "normalight_event.npz"}
            )

            pair_count = min(len(low_files), len(normal_files))
            for i in range(pair_count):
                split_event_path = split_event_files[i] if i < len(split_event_files) else None
                samples.append(EventFilmSample(
                    low_path=low_files[i],
                    normal_path=normal_files[i],
                    scene_name=scene_name,
                    frame_index=i,
                    split_event_path=split_event_path,
                    whole_event_path=whole_event_path if os.path.exists(whole_event_path) else None,
                ))
        return samples

    def _load_event_for_sample(self, sample):
        if sample.split_event_path is not None:
            return self._read_npz_event(sample.split_event_path)
        if sample.whole_event_path is not None:
            events = self._read_npz_event(sample.whole_event_path)
            return self._slice_whole_event(events, sample)
        # 没有事件文件时返回全零体素，便于调试数据路径，但训练日志会显示 event_activity=0。
        with Image.open(sample.low_path) as img:
            w, h = img.size
        return torch.zeros(self.event_channels, h, w, dtype=torch.float32)

    def _read_npz_event(self, path):
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        if {"timestamp", "x", "y", "polarity"}.issubset(keys):
            event = np.stack(
                [data["timestamp"], data["x"], data["y"], data["polarity"]],
                axis=1,
            )
        else:
            arr = data["arr_0"] if "arr_0" in keys else data[keys[0]]
            if arr.dtype.fields is not None:
                event = np.stack(
                    [arr["timestamp"], arr["x"], arr["y"], arr["polarity"]],
                    axis=1,
                )
            else:
                event = arr
        data.close()
        if isinstance(event, np.ndarray) and event.ndim == 3:
            # 已经是 voxel 的情况，支持 [C,H,W] 或 [H,W,C]。
            event_t = torch.from_numpy(event).float()
            if event_t.shape[0] != self.event_channels and event_t.shape[-1] == self.event_channels:
                event_t = event_t.permute(2, 0, 1).contiguous()
            return event_t
        return torch.from_numpy(np.asarray(event)).float()

    def _slice_whole_event(self, events, sample):
        if events.ndim == 3:
            return events
        low_dir = os.path.dirname(sample.low_path)
        low_files = sorted(glob.glob(os.path.join(low_dir, "*.png")))
        idx = sample.frame_index
        if len(events) == 0:
            return events

        # EGLol 的规则：当前 frame 的事件窗口由前后相邻 frame timestamp 决定。
        start_t = events[0, 0] if idx == 0 else int(os.path.splitext(os.path.basename(low_files[idx - 1]))[0])
        end_t = events[-1, 0] if idx >= len(low_files) - 1 else int(os.path.splitext(os.path.basename(low_files[idx + 1]))[0])
        mask = (events[:, 0] > start_t) & (events[:, 0] < end_t)
        sliced = events[mask]
        return sliced if len(sliced) > 0 else events[:0]

    def _aligned_preprocess(self, low_t, normal_t, raw_or_voxel, width, height):
        if self.split == "train":
            crop_h = crop_w = self.image_size
            if height < crop_h or width < crop_w:
                scale = max(crop_h / max(height, 1), crop_w / max(width, 1))
                new_h, new_w = int(round(height * scale)), int(round(width * scale))
                low_t = F.interpolate(low_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
                normal_t = F.interpolate(normal_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
                raw_or_voxel = self._scale_event(raw_or_voxel, scale)
                height, width = new_h, new_w
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)
            do_flip = random.random() > 0.5
        else:
            crop_size = min(height, width)
            top = (height - crop_size) // 2
            left = (width - crop_size) // 2
            crop_h = crop_w = crop_size
            do_flip = False

        low_crop = low_t[:, top:top + crop_h, left:left + crop_w]
        normal_crop = normal_t[:, top:top + crop_h, left:left + crop_w]
        event_crop = self._crop_event(raw_or_voxel, top, left, crop_h, crop_w)

        if do_flip:
            low_crop = torch.flip(low_crop, dims=[2])
            normal_crop = torch.flip(normal_crop, dims=[2])
            event_crop = torch.flip(event_crop, dims=[2])

        if crop_h != self.image_size or crop_w != self.image_size:
            low_crop = F.interpolate(low_crop.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            normal_crop = F.interpolate(normal_crop.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            event_crop = F.interpolate(event_crop.unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest")[0]
        return low_crop, normal_crop, event_crop

    def _scale_event(self, event, scale):
        if event.ndim == 3:
            return F.interpolate(event.unsqueeze(0), scale_factor=scale, mode="nearest")[0]
        event = event.clone()
        if event.numel() > 0:
            event[:, 1] = event[:, 1] * scale
            event[:, 2] = event[:, 2] * scale
        return event

    def _crop_event(self, event, top, left, crop_h, crop_w):
        if event.ndim == 3:
            return event[:, top:top + crop_h, left:left + crop_w].float()
        if event.numel() == 0:
            return torch.zeros(self.event_channels, crop_h, crop_w, dtype=torch.float32)

        # raw event 格式为 [t,x,y,p]。EGLol 中 x/y 命名有些混乱，这里按常规 x=列, y=行。
        x = event[:, 1]
        y = event[:, 2]
        mask = (x >= left) & (x < left + crop_w) & (y >= top) & (y < top + crop_h)
        crop_event = event[mask].clone()
        if crop_event.numel() == 0:
            return torch.zeros(self.event_channels, crop_h, crop_w, dtype=torch.float32)
        crop_event[:, 1] -= left
        crop_event[:, 2] -= top
        return self._generate_voxel_grid(crop_event, crop_h, crop_w)

    def _generate_voxel_grid(self, event, height, width):
        event_start = event[0, 0]
        event_end = event[-1, 0]
        denom = torch.clamp(event_end - event_start, min=1.0)
        ch = ((event[:, 0].float() - event_start) / denom * self.event_bins).long()
        ch.clamp_(0, self.event_bins - 1)
        ex = event[:, 1].long().clamp_(0, width - 1)
        ey = event[:, 2].long().clamp_(0, height - 1)
        ep = event[:, 3].float()
        ep[ep == 0] = -1.0

        voxel = torch.zeros(self.event_channels, height, width, dtype=torch.float32)
        voxel.index_put_((ch, ey, ex), ep, accumulate=True)
        return voxel

    def _normalize_event_voxel(self, event):
        denom = event.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        event = event / denom
        return event.clamp(-5.0, 5.0) / 5.0

    @staticmethod
    def _normalize_event_voxel_refinement(event, eps=1e-6):
        """按预训练 EventPyramidEncoder 分布，仅标准化非零 voxel 值。"""
        output = torch.zeros_like(event)
        nonzero_mask = event != 0
        values = event[nonzero_mask]
        if values.numel() == 0:
            return output
        mean = values.mean()
        std = values.std(unbiased=False)
        if not torch.isfinite(std) or std < eps:
            # 常数事件只携带计数而没有可稳定标准化的方差，置零最安全。
            return output
        output[nonzero_mask] = (values - mean) / (std + eps)
        return output
