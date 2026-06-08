import torch
import torch.nn as nn


class EventConvBlock(nn.Module):
    """
    事件特征提取的基础卷积块。

    作用：
    用两层 Conv-GN-SiLU 从稀疏事件体素中提取局部结构特征。
    GroupNorm 对 batch size 较小的双卡训练更稳，不依赖 batch 统计。
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EventEncoder(nn.Module):
    """
    轻量多尺度事件编码器。

    输入：
        events: [B, C_event, H, W]，默认 C_event=5，对应 EGLol 风格有符号时间 bin。

    输出：
        dict，其中 key 是目标空间分辨率字符串，例如 "32"/"16"/"8"/"4"。
        这些尺度会被 Event-FiLM LoRA 层按 self-attention token 尺寸选择。
    """

    def __init__(self, in_channels=5, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.stem = EventConvBlock(in_channels, base_channels, stride=2)
        self.down_64 = EventConvBlock(base_channels, 64, stride=2)
        self.down_32 = EventConvBlock(64, 64, stride=2)
        self.down_16 = EventConvBlock(64, 96, stride=2)
        self.down_8 = EventConvBlock(96, 128, stride=2)
        self.down_4 = EventConvBlock(128, 128, stride=2)
        self.out_channels_by_size = {
            "32": 64,
            "16": 96,
            "8": 128,
            "4": 128,
        }

    def forward(self, events):
        # events 的数值已在 dataset 中按样本归一化；这里不再改变事件强度分布。
        x = self.stem(events)
        x = self.down_64(x)
        feat_32 = self.down_32(x)
        feat_16 = self.down_16(feat_32)
        feat_8 = self.down_8(feat_16)
        feat_4 = self.down_4(feat_8)
        return {
            "32": feat_32,
            "16": feat_16,
            "8": feat_8,
            "4": feat_4,
        }
