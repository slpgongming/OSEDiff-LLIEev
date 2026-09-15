"""事件 voxel 时间通道布局的轻量工具，不依赖图像读取或数据集实现。"""

import torch


def concentrate_event_voxel(event, active_bins):
    """将 ``[C,H,W]`` voxel 压缩到前 n 个 bin，尾部通道保持为零。

    输入与输出的通道数均为 C。原始第 ``i`` 个时间 bin 累加到
    ``floor(i * n / C)``，故 ``n=1`` 时所有事件严格聚集到 bin 0；这复现了
    旧 timestamp float32 精度问题产生的关键 voxel 布局，而不改变模型的 5 通道
    输入接口。
    """
    if event.ndim != 3:
        raise ValueError(f"Expected event voxel [C,H,W], got {tuple(event.shape)}.")
    channels = event.shape[0]
    active_bins = int(active_bins)
    if not 1 <= active_bins <= channels:
        raise ValueError(
            f"active_bins must be in [1, {channels}], got {active_bins}."
        )
    if active_bins == channels:
        # 标准 5-bin 设置不引入额外复制或数值差异。
        return event
    target_bins = torch.div(
        torch.arange(channels, device=event.device) * active_bins,
        channels,
        rounding_mode="floor",
    ).long()
    concentrated = torch.zeros_like(event)
    # index_add_ 汇总同一目标 bin 的来源 bin，同时保留像素位置与正负极性。
    concentrated.index_add_(0, target_bins, event)
    return concentrated
