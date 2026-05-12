"""
SDE Indoor 数据集加载器
继承 BaseDataset，输出键与 train.py 期望一致：pixel_images / pixel_events / gts
"""
import os
import random
import numpy as np
import cv2
import torch
from data_processing.base_dataset import BaseDataset


class SDEIndoorDataset(BaseDataset):
    """SDE Indoor 数据集：低光图像 + 正常光图像 + 事件 voxel grid"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--patch_size', type=int, default=256,
                               help='裁剪尺寸（高=宽）')
            parser.add_argument('--num_bins', type=int, default=5,
                               help='voxel grid 通道数')
            parser.add_argument('--event_norm', type=bool, default=True,
                               help='是否对 voxel grid 做均值/标准差归一化')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.is_train = opt.isTrain
        self.num_bins = opt.num_bins
        self.event_norm = getattr(opt, 'event_norm', True)

        if self.is_train:
            self.patch_size = opt.patch_size
        else:
            self.patch_size = getattr(opt, 'patch_size', 256)

        self.files, self.video_names, self.video_file = [], [], []

        # 扫描所有序列目录
        seq_names = [d for d in os.listdir(self.root)
                     if os.path.isdir(os.path.join(self.root, d))]
        seq_names.sort()

        for seq_name in seq_names:
            low_folder = os.path.join(self.root, seq_name, "low")
            normal_folder = os.path.join(self.root, seq_name, "normal")
            if not (os.path.isdir(low_folder) and os.path.isdir(normal_folder)):
                continue

            low_list = sorted([f for f in os.listdir(low_folder)
                               if f.endswith('.png')])
            normal_list = sorted([f for f in os.listdir(normal_folder)
                                  if f.endswith('.png')])
            if not low_list or len(low_list) != len(normal_list):
                continue

            # 加载事件数据（一次性加载整个序列，节省每次 __getitem__ 的 IO）
            event_file = os.path.join(low_folder, "lowlight_event.npz")
            if os.path.exists(event_file):
                evdata = np.load(event_file)
                if evdata.files:
                    events = evdata['arr_0']
                    if events.ndim == 1:
                        events = np.stack([
                            events['timestamp'],
                            events['x'],
                            events['y'],
                            events['polarity']
                        ], axis=1)
                else:
                    events = np.empty((0, 4), dtype=np.float32)
            else:
                events = np.empty((0, 4), dtype=np.float32)

            self.files.append({
                'seq_name': seq_name,
                'low_folder': low_folder,
                'normal_folder': normal_folder,
                'low_list': low_list,
                'normal_list': normal_list,
                'events': events,
                'event_start_t': float(events[0, 0]) if len(events) > 0 else 0.0,
                'event_end_t': float(events[-1, 0]) if len(events) > 0 else 0.0,
            })
            self.video_names.append(seq_name)

        # 获取图像尺寸（从第一个序列第一帧）
        H, W = 480, 640  # 默认值
        if self.files:
            first_seq = self.files[0]
            sample_path = os.path.join(first_seq['low_folder'], first_seq['low_list'][0])
            img = cv2.imread(sample_path)
            if img is not None:
                H, W = img.shape[:2]
        self.H, self.W = H, W

    def __len__(self):
        return len(self.files)

    def get_event_for_frame(self, file_info, idx):
        """根据帧索引提取对应时间窗口内的事件"""
        low_list = file_info['low_list']
        current_t = int(low_list[idx][:-4])

        if idx == 0:
            start_t = file_info['event_start_t']
        else:
            prev_t = int(low_list[idx - 1][:-4])
            start_t = (file_info['event_start_t'] + prev_t) / 2.0 if idx == 1 else prev_t

        if idx == len(low_list) - 1:
            end_t = file_info['event_end_t']
        else:
            next_t = int(low_list[idx + 1][:-4])
            end_t = (current_t + next_t) / 2.0

        events = file_info['events']
        if len(events) == 0:
            return np.empty((0, 4), dtype=np.float32)

        mask = (events[:, 0] >= start_t) & (events[:, 0] <= end_t)
        return events[mask].copy()

    def _crop(self, images, event):
        """对图像和事件进行随机/中心裁剪"""
        H, W = self.H, self.W
        ph = pw = self.patch_size

        if self.is_train:
            min_x = random.randint(0, max(0, H - ph))
            min_y = random.randint(0, max(0, W - pw))
        else:
            min_x = (H - ph) // 2
            min_y = (W - pw) // 2

        max_x = min_x + ph
        max_y = min_y + pw

        cropped_imgs = [img[min_x:max_x, min_y:max_y] for img in images]

        if len(event) > 0:
            mask_x = (event[:, 2] < max_x) & (event[:, 2] >= min_x)
            mask_y = (event[:, 1] < max_y) & (event[:, 1] >= min_y)
            mask = mask_x & mask_y
            cropped_ev = event[mask].copy()
            cropped_ev[:, 1] -= min_y
            cropped_ev[:, 2] -= min_x
        else:
            cropped_ev = np.empty((0, 4), dtype=np.float32)

        return cropped_imgs, cropped_ev

    def _generate_voxel_grid(self, event):
        """将事件点云转换为 voxel grid"""
        ph, pw = self.patch_size, self.patch_size

        if len(event) == 0:
            return torch.zeros((self.num_bins, ph, pw), dtype=torch.float32)

        ev = event.copy()
        t0, t1 = float(ev[0, 0]), float(ev[-1, 0])
        if t1 == t0:
            t1 = t0 + 1.0

        ch = ((ev[:, 0] - t0) / (t1 - t0) * self.num_bins).astype(np.int64)
        ch = np.clip(ch, 0, self.num_bins - 1)

        ex = ev[:, 1].astype(np.int64)
        ey = ev[:, 2].astype(np.int64)
        ep = ev[:, 3].astype(np.float32)
        ep[ep == 0] = -1.0

        vg = torch.zeros((self.num_bins, ph, pw), dtype=torch.float32)

        valid = (0 <= ex) & (ex < pw) & (0 <= ey) & (ey < ph)
        vg[ch[valid], ey[valid], ex[valid]] += ep[valid]

        if self.event_norm and (vg != 0).any():
            nz = vg[vg != 0]
            vg[vg != 0] = (nz - nz.mean()) / (nz.std() + 1e-8)

        return vg

    def __getitem__(self, index):
        file_info = self.files[index]

        # 随机选一帧
        frame_idx = random.randint(0, len(file_info['low_list']) - 1)

        # 加载图像
        low_path = os.path.join(file_info['low_folder'], file_info['low_list'][frame_idx])
        normal_path = os.path.join(file_info['normal_folder'], file_info['normal_list'][frame_idx])

        img_low = cv2.cvtColor(cv2.imread(low_path), cv2.COLOR_BGR2RGB)
        img_normal = cv2.cvtColor(cv2.imread(normal_path), cv2.COLOR_BGR2RGB)

        # 获取事件
        event = self.get_event_for_frame(file_info, frame_idx)

        # 裁剪
        imgs, cropped_ev = self._crop([img_low, img_normal], event)
        img_low_crop = imgs[0]
        img_normal_crop = imgs[1]

        # voxel grid
        voxel_grid = self._generate_voxel_grid(cropped_ev)

        # 归一化到 [0, 1]（线性 RGB）
        img_low_norm = img_low_crop.astype(np.float32) / 255.0
        img_normal_norm = img_normal_crop.astype(np.float32) / 255.0

        return {
            'pixel_images': torch.from_numpy(img_low_norm).permute(2, 0, 1).float(),   # 低光 [3, H, W]
            'pixel_events': voxel_grid,                                                # 事件  [num_bins, H, W]
            'gts': torch.from_numpy(img_normal_norm).permute(2, 0, 1).float(),         # 正常光 [3, H, W]
            'text': '',
            'negative_text': '',
            'seq_name': file_info['seq_name'],
            'frame_idx': frame_idx,
            'save_path': f"{file_info['seq_name']}_{frame_idx}",
        }


class SDEIndoorTestDataset(BaseDataset):
    """SDE Indoor 测试数据集：逐帧遍历，返回完整图像（不裁剪）"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_bins', type=int, default=5,
                           help='voxel grid 通道数')
        parser.add_argument('--event_norm', type=bool, default=True,
                           help='是否对 voxel grid 做均值/标准差归一化')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.num_bins = opt.num_bins
        self.event_norm = getattr(opt, 'event_norm', True)

        self.samples = []  # 存储 (seq_name, low_folder, normal_folder, low_filename, normal_filename, events, event_start_t, event_end_t)

        # 扫描所有序列目录
        seq_names = [d for d in os.listdir(self.root)
                     if os.path.isdir(os.path.join(self.root, d))]
        seq_names.sort()

        for seq_name in seq_names:
            low_folder = os.path.join(self.root, seq_name, "low")
            normal_folder = os.path.join(self.root, seq_name, "normal")
            if not (os.path.isdir(low_folder) and os.path.isdir(normal_folder)):
                continue

            low_list = sorted([f for f in os.listdir(low_folder)
                              if f.endswith('.png') and not f.endswith('event.npz')])
            normal_list = sorted([f for f in os.listdir(normal_folder)
                                if f.endswith('.png')])
            if not low_list or len(low_list) != len(normal_list):
                continue

            # 加载事件数据
            event_file = os.path.join(low_folder, "lowlight_event.npz")
            if os.path.exists(event_file):
                evdata = np.load(event_file)
                if evdata.files:
                    events = evdata['arr_0']
                    if events.ndim == 1:
                        events = np.stack([
                            events['timestamp'],
                            events['x'],
                            events['y'],
                            events['polarity']
                        ], axis=1)
                else:
                    events = np.empty((0, 4), dtype=np.float32)
            else:
                events = np.empty((0, 4), dtype=np.float32)

            event_start_t = float(events[0, 0]) if len(events) > 0 else 0.0
            event_end_t = float(events[-1, 0]) if len(events) > 0 else 0.0

            # 为每一帧创建样本
            for i, (low_f, normal_f) in enumerate(zip(low_list, normal_list)):
                self.samples.append({
                    'seq_name': seq_name,
                    'low_folder': low_folder,
                    'normal_folder': normal_folder,
                    'low_filename': low_f,
                    'normal_filename': normal_f,
                    'frame_idx': i,
                    'events': events,
                    'event_start_t': event_start_t,
                    'event_end_t': event_end_t,
                })

        # 获取图像尺寸
        if self.samples:
            first_sample = self.samples[0]
            sample_path = os.path.join(first_sample['low_folder'], first_sample['low_filename'])
            img = cv2.imread(sample_path)
            if img is not None:
                self.H, self.W = img.shape[:2]
            else:
                self.H, self.W = 480, 640
        else:
            self.H, self.W = 480, 640

    def __len__(self):
        return len(self.samples)

    def get_event_for_frame(self, sample, idx):
        """根据帧索引提取对应时间窗口内的事件"""
        low_list = [s['low_filename'] for s in self.samples if s['seq_name'] == sample['seq_name']]
        current_t = int(sample['low_filename'][:-4])

        # 找到同序列中的前后帧
        seq_samples = [s for s in self.samples if s['seq_name'] == sample['seq_name']]
        sorted_samples = sorted(seq_samples, key=lambda x: int(x['low_filename'][:-4]))

        current_idx = next(i for i, s in enumerate(sorted_samples) if s['low_filename'] == sample['low_filename'])

        if current_idx == 0:
            start_t = sample['event_start_t']
        else:
            prev_t = int(sorted_samples[current_idx - 1]['low_filename'][:-4])
            start_t = (sample['event_start_t'] + prev_t) / 2.0 if current_idx == 1 else prev_t

        if current_idx == len(sorted_samples) - 1:
            end_t = sample['event_end_t']
        else:
            next_t = int(sorted_samples[current_idx + 1]['low_filename'][:-4])
            end_t = (current_t + next_t) / 2.0

        events = sample['events']
        if len(events) == 0:
            return np.empty((0, 4), dtype=np.float32)

        mask = (events[:, 0] >= start_t) & (events[:, 0] <= end_t)
        return events[mask].copy()

    def _generate_voxel_grid(self, event):
        """将事件点云转换为 voxel grid（完整图像尺寸）"""
        H, W = self.H, self.W

        if len(event) == 0:
            return torch.zeros((self.num_bins, H, W), dtype=torch.float32)

        ev = event.copy()
        t0, t1 = float(ev[0, 0]), float(ev[-1, 0])
        if t1 == t0:
            t1 = t0 + 1.0

        ch = ((ev[:, 0] - t0) / (t1 - t0) * self.num_bins).astype(np.int64)
        ch = np.clip(ch, 0, self.num_bins - 1)

        ex = ev[:, 1].astype(np.int64)
        ey = ev[:, 2].astype(np.int64)
        ep = ev[:, 3].astype(np.float32)
        ep[ep == 0] = -1.0

        vg = torch.zeros((self.num_bins, H, W), dtype=torch.float32)

        valid = (0 <= ex) & (ex < W) & (0 <= ey) & (ey < H)
        vg[ch[valid], ey[valid], ex[valid]] += ep[valid]

        if self.event_norm and (vg != 0).any():
            nz = vg[vg != 0]
            vg[vg != 0] = (nz - nz.mean()) / (nz.std() + 1e-8)

        return vg

    def __getitem__(self, index):
        sample = self.samples[index]

        # 加载完整图像
        low_path = os.path.join(sample['low_folder'], sample['low_filename'])
        normal_path = os.path.join(sample['normal_folder'], sample['normal_filename'])

        img_low = cv2.cvtColor(cv2.imread(low_path), cv2.COLOR_BGR2RGB)
        img_normal = cv2.cvtColor(cv2.imread(normal_path), cv2.COLOR_BGR2RGB)

        # 获取当前帧的事件
        event = self.get_event_for_frame(sample, sample['frame_idx'])

        # voxel grid（完整图像尺寸）
        voxel_grid = self._generate_voxel_grid(event)

        # 归一化到 [0, 1]（线性 RGB）
        img_low_norm = img_low.astype(np.float32) / 255.0
        img_normal_norm = img_normal.astype(np.float32) / 255.0

        return {
            'pixel_images': torch.from_numpy(img_low_norm).permute(2, 0, 1).float(),   # 低光 [3, H, W]
            'pixel_events': voxel_grid,                                                 # 事件  [num_bins, H, W]
            'gts': torch.from_numpy(img_normal_norm).permute(2, 0, 1).float(),          # 正常光 [3, H, W]
            'text': '',
            'negative_text': '',
            'save_path': f"{sample['seq_name']}_{sample['frame_idx']}",
            'seq_name': sample['seq_name'],
            'frame_idx': sample['frame_idx'],
            'low_filename': sample['low_filename'],
        }
