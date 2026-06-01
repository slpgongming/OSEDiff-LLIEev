import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _is_png(name: str) -> bool:
    return name.lower().endswith(".png")


def _frame_time(filename: str, fallback: int) -> int:
    stem = Path(filename).stem
    try:
        return int(stem)
    except ValueError:
        # SDE indoor normally uses timestamp filenames. This fallback keeps
        # smoke tests usable if a copied subset has non-timestamp names.
        return fallback


def _load_event_array(npz_path: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        return np.empty((0, 4), dtype=np.float32)

    npz = np.load(npz_path, allow_pickle=False)
    if "arr_0" in npz.files:
        events = npz["arr_0"]
    elif {"timestamp", "x", "y", "polarity"}.issubset(set(npz.files)):
        events = np.stack(
            [npz["timestamp"], npz["x"], npz["y"], npz["polarity"]],
            axis=1,
        )
    elif len(npz.files) == 1:
        events = npz[npz.files[0]]
    else:
        events = np.empty((0, 4), dtype=np.float32)

    if events.ndim == 1 and events.dtype.fields is not None:
        names = events.dtype.fields.keys()
        t_key = "timestamp" if "timestamp" in names else "t"
        p_key = "polarity" if "polarity" in names else "p"
        events = np.stack([events[t_key], events["x"], events["y"], events[p_key]], axis=1)

    if events.ndim != 2 or events.shape[-1] < 4:
        return np.empty((0, 4), dtype=np.float32)

    # SDE 的 timestamp 是 1e15 量级，float32 会丢失相邻帧的微秒级差异，
    # 因此事件切片阶段必须保留 float64/int64 精度。
    return events[:, :4].astype(np.float64, copy=False)


class SDEIndoorEGLolDataset(Dataset):
    """
    SDE indoor 数据集 loader，事件切片、裁剪和 voxel 生成参考 references/eglol.py。

    返回值同时保留 OSEDiff 训练需要的 [-1, 1] 图像，以及 LLEEncoder/可视化更直观的 [0, 1] 图像。
    low 与 normal 文件名不要求相同，严格按排序后的 index 配对。
    """

    def __init__(
        self,
        root: str,
        patch_size: int = 256,
        num_bins: int = 5,
        is_train: bool = True,
        full_resolution: bool = False,
        event_norm: str = "none",
        split_event: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.root = root
        self.patch_size = patch_size
        self.num_bins = num_bins
        self.is_train = is_train
        self.full_resolution = full_resolution
        self.event_norm = event_norm
        self.split_event = split_event
        self.samples: List[Dict] = []

        seq_names = [
            name for name in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, name))
        ] if os.path.isdir(root) else []

        for seq_name in seq_names:
            low_dir = os.path.join(root, seq_name, "low")
            normal_dir = os.path.join(root, seq_name, "normal")
            if not (os.path.isdir(low_dir) and os.path.isdir(normal_dir)):
                continue

            low_files = sorted([name for name in os.listdir(low_dir) if _is_png(name)])
            normal_files = sorted([name for name in os.listdir(normal_dir) if _is_png(name)])
            pair_count = min(len(low_files), len(normal_files))
            if pair_count == 0:
                continue

            whole_events = None
            if not split_event:
                whole_events = _load_event_array(os.path.join(low_dir, "lowlight_event.npz"))

            split_events = sorted([
                name for name in os.listdir(low_dir)
                if name.lower().endswith(".npz") and name != "lowlight_event.npz"
            ])

            for idx in range(pair_count):
                event_path = None
                if split_event and idx < len(split_events):
                    event_path = os.path.join(low_dir, split_events[idx])

                self.samples.append({
                    "seq_name": seq_name,
                    "low_dir": low_dir,
                    "normal_dir": normal_dir,
                    "low_files": low_files[:pair_count],
                    "normal_files": normal_files[:pair_count],
                    "frame_idx": idx,
                    "low_filename": low_files[idx],
                    "normal_filename": normal_files[idx],
                    "events": whole_events,
                    "event_path": event_path,
                })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _slice_events(self, sample: Dict) -> np.ndarray:
        if self.split_event and sample["event_path"] is not None:
            return _load_event_array(sample["event_path"])

        events = sample["events"]
        if events is None or len(events) == 0:
            return np.empty((0, 4), dtype=np.float32)

        idx = sample["frame_idx"]
        low_files = sample["low_files"]

        # EGLol 风格时间窗：第 0 帧从第一个事件开始，中间帧以前后 low frame timestamp 切片。
        if idx == 0:
            start_t = float(events[0, 0])
        else:
            start_t = float(_frame_time(low_files[idx - 1], idx - 1))

        if idx == len(low_files) - 1:
            end_t = float(events[-1, 0])
        else:
            end_t = float(_frame_time(low_files[idx + 1], idx + 1))

        if end_t < start_t:
            start_t, end_t = end_t, start_t

        mask = (events[:, 0] > start_t) & (events[:, 0] < end_t)
        return events[mask].copy()

    def _choose_crop(self, height: int, width: int) -> Tuple[int, int, int, int]:
        if self.full_resolution:
            return 0, 0, height, width

        crop_h = min(self.patch_size, height)
        crop_w = min(self.patch_size, width)
        if self.is_train:
            top = random.randint(0, max(0, height - crop_h))
            left = random.randint(0, max(0, width - crop_w))
        else:
            top = max(0, (height - crop_h) // 2)
            left = max(0, (width - crop_w) // 2)
        return top, left, crop_h, crop_w

    def _crop_images_and_events(
        self,
        low: np.ndarray,
        gt: np.ndarray,
        events: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = low.shape[:2]
        top, left, crop_h, crop_w = self._choose_crop(height, width)
        bottom = top + crop_h
        right = left + crop_w

        low_crop = low[top:bottom, left:right, :]
        gt_crop = gt[top:bottom, left:right, :]

        if len(events) == 0:
            return low_crop, gt_crop, np.empty((0, 4), dtype=np.float32)

        # 事件格式为 [t, x, y, polarity]，x 是水平坐标，y 是垂直坐标。
        mask = (
            (events[:, 1] >= left) & (events[:, 1] < right) &
            (events[:, 2] >= top) & (events[:, 2] < bottom)
        )
        cropped = events[mask].copy()
        if len(cropped) > 0:
            cropped[:, 1] -= left
            cropped[:, 2] -= top
        return low_crop, gt_crop, cropped

    def _generate_voxel_grid(self, events: np.ndarray, height: int, width: int) -> torch.Tensor:
        # voxel 的 channel 维是时间 bin，空间维与裁剪后的 RGB patch 对齐。
        voxel = torch.zeros((self.num_bins, height, width), dtype=torch.float32)
        if len(events) == 0:
            return voxel

        t0 = float(events[0, 0])
        t1 = float(events[-1, 0])
        if t1 <= t0:
            t1 = t0 + 1.0

        ch = ((events[:, 0] - t0) / (t1 - t0) * self.num_bins).astype(np.int64)
        ch = np.clip(ch, 0, self.num_bins - 1)
        ex = events[:, 1].astype(np.int64)
        ey = events[:, 2].astype(np.int64)
        ep = events[:, 3].astype(np.float32)
        ep[ep == 0] = -1.0

        valid = (ex >= 0) & (ex < width) & (ey >= 0) & (ey < height)
        if np.any(valid):
            voxel.index_put_(
                (
                    torch.from_numpy(ch[valid]),
                    torch.from_numpy(ey[valid]),
                    torch.from_numpy(ex[valid]),
                ),
                torch.from_numpy(ep[valid]),
                accumulate=True,
            )

        if self.event_norm == "zscore" and torch.any(voxel != 0):
            nonzero = voxel[voxel != 0]
            voxel[voxel != 0] = (nonzero - nonzero.mean()) / (nonzero.std() + 1e-8)
        return voxel

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        low_path = os.path.join(sample["low_dir"], sample["low_filename"])
        gt_path = os.path.join(sample["normal_dir"], sample["normal_filename"])

        low_bgr = cv2.imread(low_path, cv2.IMREAD_COLOR)
        gt_bgr = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if low_bgr is None or gt_bgr is None:
            raise FileNotFoundError(f"Failed to read low/GT pair: {low_path} | {gt_path}")

        low = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        events = self._slice_events(sample)
        low_crop, gt_crop, crop_events = self._crop_images_and_events(low, gt, events)

        height, width = low_crop.shape[:2]
        event_voxel = self._generate_voxel_grid(crop_events, height, width)

        low_01 = torch.from_numpy(low_crop.astype(np.float32) / 255.0).permute(2, 0, 1)
        gt_01 = torch.from_numpy(gt_crop.astype(np.float32) / 255.0).permute(2, 0, 1)

        return {
            "conditioning_pixel_values": low_01 * 2.0 - 1.0,
            "output_pixel_values": gt_01 * 2.0 - 1.0,
            "low_rgb_01": low_01,
            "gt_rgb_01": gt_01,
            "event_values": event_voxel,
            "prompt": "",
            "neg_prompt": "",
            "seq_name": sample["seq_name"],
            "frame_id": Path(sample["low_filename"]).stem,
            "low_filename": sample["low_filename"],
            "normal_filename": sample["normal_filename"],
            "save_path": f"{sample['seq_name']}_{Path(sample['low_filename']).stem}",
            "event_nonzero": int(torch.count_nonzero(event_voxel).item()),
        }
