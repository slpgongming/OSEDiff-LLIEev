import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


def _load_event_array(path: str) -> np.ndarray:
    """Load one SDE event npz as an [N, 4] array: timestamp, x, y, polarity."""
    data = np.load(path, allow_pickle=False)
    if not data.files:
        return np.empty((0, 4), dtype=np.float32)

    events = data[data.files[0]]
    if events.ndim == 1 and events.dtype.fields is not None:
        events = np.stack(
            [events["timestamp"], events["x"], events["y"], events["polarity"]],
            axis=1,
        )
    return events.astype(np.float32, copy=False)


class SDEEventDataset(Dataset):
    """
    SDE indoor dataset for Stage 3 HDRev event-latent OSEDiff.

    Returned tensors:
      conditioning_pixel_values: low RGB normalized to [-1, 1] for OSEDiff losses.
      hdrev_pixel_values: low RGB in [0, 1] for HDRevEncoder.
      event_voxel: event voxel grid [num_bins, H, W], crop-aligned with RGB.
      output_pixel_values: normal-light GT normalized to [-1, 1].
    """

    def __init__(self, split: str, args):
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.args = args
        self.root = args.train_data_root if split == "train" else getattr(args, "test_data_root", args.train_data_root)
        self.image_size = int(args.image_size)
        self.num_bins = int(getattr(args, "num_bins", 5))
        self.event_norm = bool(getattr(args, "event_norm", True))
        self.samples: List[Dict[str, str]] = []

        for scene_name in sorted(os.listdir(self.root)):
            scene_dir = os.path.join(self.root, scene_name)
            if not os.path.isdir(scene_dir):
                continue

            low_dir = os.path.join(scene_dir, "low")
            normal_dir = os.path.join(scene_dir, "normal")
            if not os.path.isdir(low_dir) or not os.path.isdir(normal_dir):
                continue

            low_files = sorted(glob.glob(os.path.join(low_dir, "*.png")))
            normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
            event_files = sorted(glob.glob(os.path.join(low_dir, "*.npz")))

            # EGLol-style datasets often provide one extra trailing event file.
            # Dropping the final file keeps frame/event pairing index-based.
            if len(event_files) == len(low_files) + 1:
                event_files = event_files[:-1]

            pair_count = min(len(low_files), len(normal_files), len(event_files))
            for idx in range(pair_count):
                self.samples.append(
                    {
                        "scene": scene_name,
                        "low": low_files[idx],
                        "normal": normal_files[idx],
                        "event": event_files[idx],
                        "index": idx,
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No SDE event samples found under {self.root}")

        print(f"[SDEEventDataset] split={split} samples={len(self.samples)} root={self.root}")

    def __len__(self):
        return len(self.samples)

    def _choose_crop(self, width: int, height: int) -> Tuple[int, int, int]:
        crop_size = min(self.image_size, width, height)
        if self.split == "train":
            top = torch.randint(0, height - crop_size + 1, (1,)).item()
            left = torch.randint(0, width - crop_size + 1, (1,)).item()
        else:
            top = (height - crop_size) // 2
            left = (width - crop_size) // 2
        return top, left, crop_size

    def _crop_events(self, events: np.ndarray, top: int, left: int, crop_size: int) -> np.ndarray:
        if events.size == 0:
            return np.empty((0, 4), dtype=np.float32)

        # Event columns are timestamp, x, y, polarity. x follows image width and
        # y follows image height, so crop offsets must use left/top respectively.
        x = events[:, 1]
        y = events[:, 2]
        mask = (x >= left) & (x < left + crop_size) & (y >= top) & (y < top + crop_size)
        cropped = events[mask].copy()
        if cropped.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        cropped[:, 1] -= left
        cropped[:, 2] -= top
        return cropped

    def _generate_voxel_grid(self, events: np.ndarray, height: int, width: int) -> torch.Tensor:
        voxel = torch.zeros((self.num_bins, height, width), dtype=torch.float32)
        if events.size == 0:
            return voxel

        t0 = float(events[0, 0])
        t1 = float(events[-1, 0])
        if t1 <= t0:
            t1 = t0 + 1.0

        # Convert timestamps to temporal bins and accumulate polarity at each
        # pixel location. Polarity 0 is represented as -1, matching EGLol.
        ch = ((events[:, 0] - t0) / (t1 - t0) * self.num_bins).astype(np.int64)
        ch = np.clip(ch, 0, self.num_bins - 1)
        ex = events[:, 1].astype(np.int64)
        ey = events[:, 2].astype(np.int64)
        ep = events[:, 3].astype(np.float32)
        ep[ep == 0] = -1.0

        valid = (ex >= 0) & (ex < width) & (ey >= 0) & (ey < height)
        if valid.any():
            voxel.index_put_(
                (
                    torch.from_numpy(ch[valid]),
                    torch.from_numpy(ey[valid]),
                    torch.from_numpy(ex[valid]),
                ),
                torch.from_numpy(ep[valid]),
                accumulate=True,
            )

        if self.event_norm and torch.count_nonzero(voxel) > 0:
            nonzero = voxel[voxel != 0]
            voxel[voxel != 0] = (nonzero - nonzero.mean()) / (nonzero.std() + 1e-8)
        return voxel

    def __getitem__(self, index):
        sample = self.samples[index]
        low_img = Image.open(sample["low"]).convert("RGB")
        normal_img = Image.open(sample["normal"]).convert("RGB")
        if low_img.size != normal_img.size:
            normal_img = normal_img.resize(low_img.size, Image.LANCZOS)

        width, height = low_img.size
        top, left, crop_size = self._choose_crop(width, height)
        low_img = TF.crop(low_img, top, left, crop_size, crop_size)
        normal_img = TF.crop(normal_img, top, left, crop_size, crop_size)
        events = self._crop_events(_load_event_array(sample["event"]), top, left, crop_size)

        if crop_size != self.image_size:
            low_img = low_img.resize((self.image_size, self.image_size), Image.LANCZOS)
            normal_img = normal_img.resize((self.image_size, self.image_size), Image.LANCZOS)

            # Event coordinates are scaled to remain aligned with resized RGB.
            if events.size > 0:
                scale = self.image_size / float(crop_size)
                events[:, 1] *= scale
                events[:, 2] *= scale

        low_01 = transforms.ToTensor()(low_img)
        normal_01 = transforms.ToTensor()(normal_img)
        event_voxel = self._generate_voxel_grid(events, self.image_size, self.image_size)

        return {
            "conditioning_pixel_values": TF.normalize(low_01, mean=[0.5], std=[0.5]),
            "hdrev_pixel_values": low_01,
            "event_voxel": event_voxel,
            "output_pixel_values": TF.normalize(normal_01, mean=[0.5], std=[0.5]),
            "prompt": self.args.fixed_prompt,
            "neg_prompt": self.args.neg_prompt,
            "scene": sample["scene"],
            "low_path": sample["low"],
            "normal_path": sample["normal"],
            "event_path": sample["event"],
            "sample_index": sample["index"],
        }
