"""idea8 共用的可导结构算子；训练标签和推理误差必须使用同一配置。"""

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StructureOperatorConfig:
    version: int = 1
    sigmas: tuple = (1.0, 2.0)
    magnitude_scale: float = 0.1

    def __post_init__(self):
        if self.version != 1 or not self.sigmas or any(not math.isfinite(s) or s <= 0 for s in self.sigmas):
            raise ValueError("Invalid structure operator version/scales.")
        if not math.isfinite(self.magnitude_scale) or self.magnitude_scale <= 0:
            raise ValueError("magnitude_scale must be finite and positive.")


def gaussian_blur(image, sigma):
    """逐通道 Gaussian；replicate padding 也支持小尺寸测试张量。"""
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-x.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    channels = image.shape[1]
    return F.conv2d(F.pad(image, (radius,) * 4, mode="replicate"),
                    kernel.expand(channels, 1, -1, -1), groups=channels)


def image_structure(image_01, config=None):
    """[B,3,H,W] → [B,S,H,W] 软边缘；不做逐图最大值归一化。

    正常输入域为 [0,1]。在线 raw decode 可暂时越界，由独立 range loss 约束。
    在 FP32 中计算幅值，避免 FP16 小梯度下溢。
    """
    config = config or StructureOperatorConfig()
    if image_01.ndim != 4 or image_01.shape[1] != 3:
        raise ValueError("image_structure expects [B,3,H,W].")
    image = image_01.float()
    gray = (image * image.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
    sx = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    kernels = torch.stack((sx, sx.t()))[:, None]
    result = []
    for sigma in config.sigmas:
        smooth = gaussian_blur(gray, sigma)
        gradient = F.conv2d(F.pad(smooth, (1,) * 4, mode="replicate"), kernels)
        magnitude = (gradient.square().sum(1, keepdim=True) + 1e-12).sqrt() - 1e-6
        result.append(torch.tanh(magnitude / config.magnitude_scale))
    return torch.cat(result, dim=1)


def operator_metadata(config):
    return asdict(config)
