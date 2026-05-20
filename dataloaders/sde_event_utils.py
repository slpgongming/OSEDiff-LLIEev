"""
事件体素化工具函数
从 raw event (.npz) 转换为 voxel grid，并与 RGB concat 后送入模型。
参考 references/eglol.py 中的 voxel grid 生成逻辑。
"""
import os
import numpy as np
import torch
import cv2


def load_npz_event(filepath: str) -> np.ndarray:
    """
    加载 .npz 文件中的事件数据，返回 [N, 4] float32 数组。

    Args:
        filepath: .npz 文件路径
    Returns:
        events: [N, 4] ndarray，列为 (t, x, y, polarity)
    """
    if not os.path.exists(filepath):
        return np.empty((0, 4), dtype=np.float32)

    data = np.load(filepath)
    if not data.files:
        return np.empty((0, 4), dtype=np.float32)

    arr = data['arr_0']
    if arr.ndim == 1:
        events = np.stack([
            arr['timestamp'],
            arr['x'],
            arr['y'],
            arr['polarity']
        ], axis=1).astype(np.float32)
    else:
        events = arr.astype(np.float32)

    return events


def get_event_for_frame(low_img_list: list, events: np.ndarray, idx: int) -> np.ndarray:
    """
    根据帧索引提取对应时间窗口内的事件。

    事件窗口覆盖: [上一帧时间戳, 下一帧时间戳]
    即当前帧的事件对应的是 "上一帧 → 当前帧 → 下一帧" 的时间范围。

    Args:
        low_img_list: 按文件名排序的 low png 文件名列表（不含路径）
        events: [N, 4] ndarray，(t, x, y, polarity)
        idx: 当前帧在列表中的索引
    Returns:
        event_window: 对应时间窗口内的事件 [M, 4]
    """
    if len(events) == 0:
        return np.empty((0, 4), dtype=np.float32)

    # 当前帧时间戳
    current_t = int(low_img_list[idx][:-4])  # 去掉 .png 后缀转时间戳

    # 起始时间：前一帧的时间戳（首帧则用事件序列首时间戳）
    if idx == 0:
        start_t = events[0, 0]
    else:
        start_t = int(low_img_list[idx - 1][:-4])

    # 结束时间：下一帧的时间戳（末帧则用事件序列末时间戳）
    if idx == len(low_img_list) - 1:
        end_t = events[-1, 0]
    else:
        end_t = int(low_img_list[idx + 1][:-4])

    # 提取时间窗口内的事件
    mask = (events[:, 0] > start_t) & (events[:, 0] < end_t)
    return events[mask].copy()


def generate_voxel_grid(
    events: np.ndarray,
    num_bins: int,
    H: int,
    W: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    将原始事件点云转换为 voxel grid。

    Args:
        events: [N, 4] ndarray，(t, x, y, polarity)；若无事件则为 [0, 4]
        num_bins: voxel grid 的通道数（时间方向）
        H, W: 图像高度和宽度（裁剪后）
        normalize: 是否对非零 voxel 做均值/标准差归一化
    Returns:
        voxel_grid: [num_bins, H, W] float32 tensor
    """
    vg = torch.zeros((num_bins, H, W), dtype=torch.float32)

    if len(events) == 0:
        return vg

    ev = events.copy()
    t0, t1 = float(ev[0, 0]), float(ev[-1, 0])
    if t1 == t0:
        t1 = t0 + 1.0

    # 时间归一化到 [0, num_bins)
    ch = ((ev[:, 0] - t0) / (t1 - t0) * num_bins).astype(np.int64)
    ch = np.clip(ch, 0, num_bins - 1)

    ex = ev[:, 1].astype(np.int64)
    ey = ev[:, 2].astype(np.int64)
    ep = ev[:, 3].astype(np.float32)
    ep[ep == 0] = -1.0  # 极性: 0→-1, 1→+1

    # 筛选有效坐标
    valid = (0 <= ex) & (ex < W) & (0 <= ey) & (ey < H)
    vg[ch[valid], ey[valid], ex[valid]] += ep[valid]

    # 均值/标准差归一化（非零元素）
    if normalize:
        mask_nz = vg != 0
        if mask_nz.any():
            nz_mean = vg[mask_nz].mean().item()
            nz_std = vg[mask_nz].std().item()
            vg[mask_nz] = (vg[mask_nz] - nz_mean) / (nz_std + 1e-8)

    return vg


def get_image_size_from_file(filepath: str) -> tuple:
    """
    从图像文件读取高度和宽度。

    Args:
        filepath: 图像文件路径
    Returns:
        (H, W) 高度和宽度
    """
    img = cv2.imread(filepath)
    if img is None:
        return 480, 640
    return img.shape[:2]  # H, W
