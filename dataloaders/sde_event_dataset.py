"""
SDE Indoor Event Dataset
将低光RGB图像与逐帧事件npz文件配对，加载后concat送入OSEDiff模型。
训练：随机裁剪 + 随机水平翻转
测试：中心裁剪到最小边，再 resize 到 image_size
"""
import os
import glob
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

from .sde_event_utils import load_npz_event, generate_voxel_grid, get_image_size_from_file


class SDEEventDataset(torch.utils.data.Dataset):
    """
    SDE Indoor 数据集加载器。

    数据结构（每个场景 i_n/）：
        train_data_root/
            i_0/
                low/
                    1690097848450715.png   <- 低光图像
                    1690097848450715.npz   <- 对应事件（逐帧分割）
                    1690097848529056.png
                    1690097848529056.npz
                    ...
                    lowlight_event.npz     <- 整序列事件（可能存在）
                normal/
                    1690097848450715.png   <- Normal GT
                    1690097848529056.png
                    ...
    配对规则：low/ 和 normal/ 中的 png 文件按文件名排序后，相同 index 配对。
    """

    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        self.voxel_grid_channel = getattr(args, 'voxel_grid_channel', 5)
        self.image_size = getattr(args, 'image_size', 512)
        self.is_train = (split == 'train')

        # 扫描所有场景目录
        self.samples = []  # list of dicts
        self._scan_dataset()

        print(f"[SDEEventDataset] Loaded {len(self.samples)} samples from {args.train_data_root} (split={split})")

        # 数据增强
        if self.is_train:
            self.crop_preproc = None  # handled in __getitem__
        # test split augmentation is handled in __getitem__

    def _scan_dataset(self):
        """扫描数据集目录，构建样本列表"""
        root = self.args.train_data_root
        voxel_ch = self.voxel_grid_channel

        for scene_name in sorted(os.listdir(root)):
            scene_path = os.path.join(root, scene_name)
            if not os.path.isdir(scene_path):
                continue

            low_dir = os.path.join(scene_path, 'low')
            normal_dir = os.path.join(scene_path, 'normal')
            if not (os.path.isdir(low_dir) and os.path.isdir(normal_dir)):
                continue

            # 获取 low/ 和 normal/ 下所有 png 文件（排除 npz）
            low_png_list = sorted([
                f for f in os.listdir(low_dir)
                if f.endswith('.png')
            ])
            normal_png_list = sorted([
                f for f in os.listdir(normal_dir)
                if f.endswith('.png')
            ])

            if not low_png_list or len(low_png_list) != len(normal_png_list):
                print(f"[WARN] Scene {scene_name}: {len(low_png_list)} low vs {len(normal_png_list)} normal "
                      f"— skipped (count mismatch or empty)")
                continue

            # 获取 low/ 下所有 npz 文件（逐帧事件文件，排除整序列文件）
            low_npz_all = [f for f in os.listdir(low_dir) if f.endswith('.npz')]
            # 识别逐帧事件文件：与某个 png 文件同名（仅扩展名不同）
            frame_npz_map = {}
            for npz_f in low_npz_all:
                base = npz_f.replace('.npz', '')
                png_candidate = base + '.png'
                if png_candidate in low_png_list:
                    frame_npz_map[base] = os.path.join(low_dir, npz_f)

            # 如果没有逐帧 npz 事件文件，使用整序列事件文件
            seq_npz_path = os.path.join(low_dir, 'lowlight_event.npz')
            use_seq_npz = not frame_npz_map and os.path.exists(seq_npz_path)

            for i, (low_fname, normal_fname) in enumerate(zip(low_png_list, normal_png_list)):
                low_base = low_fname.replace('.png', '')
                normal_base = normal_fname.replace('.png', '')

                # 确定事件文件路径
                if frame_npz_map and low_base in frame_npz_map:
                    ev_path = frame_npz_map[low_base]
                    is_split = True
                elif use_seq_npz:
                    ev_path = seq_npz_path
                    is_split = False
                else:
                    # 没有可用的事件数据，跳过该帧
                    print(f"[WARN] No event file for {scene_name}/{low_fname}, skipping")
                    continue

                # 记录当前帧对应的前后帧时间戳（用于整序列 npz 模式）
                prev_t = None
                next_t = None
                if i > 0:
                    prev_t = int(low_png_list[i - 1].replace('.png', ''))
                if i < len(low_png_list) - 1:
                    next_t = int(low_png_list[i + 1].replace('.png', ''))

                self.samples.append({
                    'scene_name': scene_name,
                    'low_path': os.path.join(low_dir, low_fname),
                    'normal_path': os.path.join(normal_dir, normal_fname),
                    'event_path': ev_path,
                    'is_split_event': is_split,
                    'frame_timestamps': [int(f.replace('.png', '')) for f in low_png_list],
                    'prev_t': prev_t,
                    'next_t': next_t,
                    'frame_idx': i,
                })

    def __len__(self):
        return len(self.samples)

    def _crop_and_augment(self, img_low, img_normal, event_window, crop_size):
        """
        训练：对低光图、GT图、事件做相同的随机裁剪和水平翻转。
        测试：对低光图、GT图、事件做中心裁剪。

        Args:
            img_low: [H, W, 3] np.ndarray (RGB)
            img_normal: [H, W, 3] np.ndarray (RGB)
            event_window: [N, 4] ndarray (t, x, y, polarity)，或空 [0, 4]
            crop_size: int, 裁剪尺寸
        Returns:
            img_low_crop: [crop_size, crop_size, 3]
            img_normal_crop: [crop_size, crop_size, 3]
            event_crop: [M, 4] ndarray
        """
        H, W = img_low.shape[:2]

        # 预处理：如果图像太小，先 resize
        if min(H, W) < crop_size:
            scale = crop_size / min(H, W)
            new_w, new_h = int(W * scale), int(H * scale)
            img_low = np.array(Image.fromarray(img_low).resize((new_w, new_h), Image.LANCZOS))
            img_normal = np.array(Image.fromarray(img_normal).resize((new_w, new_h), Image.LANCZOS))
            H, W = new_h, new_w

        if self.is_train:
            # 随机裁剪
            top = random.randint(0, max(0, H - crop_size))
            left = random.randint(0, max(0, W - crop_size))
            # 随机水平翻转
            flip = random.random() > 0.5
        else:
            # 中心裁剪
            top = (H - crop_size) // 2
            left = (W - crop_size) // 2
            flip = False

        bottom = top + crop_size
        right = left + crop_size

        img_low_crop = img_low[top:bottom, left:right]
        img_normal_crop = img_normal[top:bottom, left:right]

        # 裁剪事件坐标
        if len(event_window) > 0:
            mask_x = (event_window[:, 2] >= top) & (event_window[:, 2] < bottom)
            mask_y = (event_window[:, 1] >= left) & (event_window[:, 1] < right)
            mask = mask_x & mask_y
            ev_crop = event_window[mask].copy()
            ev_crop[:, 1] -= left  # x 坐标
            ev_crop[:, 2] -= top   # y 坐标
        else:
            ev_crop = np.empty((0, 4), dtype=np.float32)

        # 水平翻转（像素和事件坐标）
        if flip:
            img_low_crop = np.flip(img_low_crop, axis=1).copy()
            img_normal_crop = np.flip(img_normal_crop, axis=1).copy()
            if len(ev_crop) > 0:
                ev_crop[:, 1] = crop_size - 1 - ev_crop[:, 1]  # 翻转 x 坐标

        return img_low_crop, img_normal_crop, ev_crop

    def __getitem__(self, idx):
        sample = self.samples[idx]
        scene_name = sample['scene_name']
        frame_idx = sample['frame_idx']

        # 1. 加载图像
        img_low = np.array(Image.open(sample['low_path']).convert('RGB'))
        img_normal = np.array(Image.open(sample['normal_path']).convert('RGB'))

        # 2. 加载并提取事件
        if sample['is_split_event']:
            # 逐帧分割的事件文件：直接加载
            events = load_npz_event(sample['event_path'])
            event_window = events
        else:
            # 整序列事件文件：根据帧时间戳提取对应窗口
            all_events = load_npz_event(sample['event_path'])
            event_window = self._extract_event_window(
                all_events,
                sample['frame_timestamps'],
                frame_idx
            )

        # 3. 裁剪与增强
        img_low_crop, img_normal_crop, ev_crop = self._crop_and_augment(
            img_low, img_normal, event_window,
            crop_size=min(self.image_size, min(img_low.shape[0], img_low.shape[1]))
        )

        # 4. 生成 voxel grid
        voxel_grid = generate_voxel_grid(
            ev_crop,
            num_bins=self.voxel_grid_channel,
            H=self.image_size,
            W=self.image_size,
            normalize=True,
        )  # [voxel_ch, H, W]

        # 5. 归一化图像到 [-1, 1]
        img_low_norm = torch.from_numpy(img_low_crop).permute(2, 0, 1).float() / 255.0
        img_low_norm = F.normalize(img_low_norm, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # GT 归一化
        img_normal_norm = torch.from_numpy(img_normal_crop).permute(2, 0, 1).float() / 255.0
        img_normal_norm = F.normalize(img_normal_norm, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # 6. Concat: [3 + voxel_ch, H, W]
        conditioning = torch.cat([img_low_norm, voxel_grid], dim=0)

        return {
            'conditioning_pixel_values': conditioning,   # [3+voxel_ch, H, W]
            'output_pixel_values': img_normal_norm,       # [3, H, W]
            'neg_prompt': getattr(self.args, 'neg_prompt', ''),
            'prompt': [getattr(self.args, 'fixed_prompt', 'high quality, bright, detailed photo')],
        }

    def _extract_event_window(self, all_events, frame_timestamps, idx):
        """
        从整序列事件中提取当前帧对应的时间窗口。
        窗口覆盖: [上一帧时间戳, 下一帧时间戳]
        """
        if len(all_events) == 0:
            return np.empty((0, 4), dtype=np.float32)

        current_t = frame_timestamps[idx]

        if idx == 0:
            start_t = all_events[0, 0]
        else:
            start_t = float(frame_timestamps[idx - 1])

        if idx == len(frame_timestamps) - 1:
            end_t = all_events[-1, 0]
        else:
            end_t = float(frame_timestamps[idx + 1])

        mask = (all_events[:, 0] > start_t) & (all_events[:, 0] < end_t)
        return all_events[mask].copy()
