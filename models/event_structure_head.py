"""独立事件结构预测器：不改变已有 5-bin EventEncoder 或生成器权重。"""

from dataclasses import asdict, dataclass
import torch
from torch import nn
import torch.nn.functional as F

from models.image_structure import StructureOperatorConfig


@dataclass(frozen=True)
class StructureHeadConfig:
    bins: int = 5
    base_channels: int = 32
    scales: int = 2

    def __post_init__(self):
        if self.bins < 1 or self.base_channels < 4 or self.scales < 1:
            raise ValueError("Invalid structure head dimensions.")


def _block(in_channels, out_channels, stride=1):
    # GroupNorm 避免小 batch 的 running statistics 漂移。
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                         nn.GroupNorm(1, out_channels), nn.SiLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.SiLU())


class EventStructurePredictor(nn.Module):
    """输入 [B,2*bins,H,W]；输出两组 [B,S,H,W] 的边缘和可信度。

    编码器保留 1/2 尺度跳连；置信度在结构分支冻结后单独训练。
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or StructureHeadConfig()
        c = self.config.base_channels
        self.enc1 = _block(2 * self.config.bins, c, 2)
        self.enc2 = _block(c, 2 * c, 2)
        self.enc3 = _block(2 * c, 3 * c, 2)
        self.fuse = _block(6 * c, c)
        self.edge = nn.Conv2d(c, self.config.scales, 1)
        self.confidence = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(),
                                        nn.Conv2d(c, self.config.scales, 1))

    def forward(self, events):
        if events.ndim != 4 or events.shape[1] != 2 * self.config.bins:
            raise ValueError("Structure input must be [B,2*bins,H,W].")
        x1 = self.enc1(events)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x = self.fuse(torch.cat((x1, F.interpolate(x2, x1.shape[-2:], mode="bilinear", align_corners=False),
                                F.interpolate(x3, x1.shape[-2:], mode="bilinear", align_corners=False)), dim=1))
        edge = torch.sigmoid(F.interpolate(self.edge(x), events.shape[-2:], mode="bilinear", align_corners=False))
        confidence = torch.sigmoid(F.interpolate(self.confidence(x.detach()), events.shape[-2:],
                                                 mode="bilinear", align_corners=False))
        return edge, confidence

    def set_training_phase(self, phase):
        if phase not in {"structure", "confidence"}:
            raise ValueError(phase)
        self.requires_grad_(phase == "structure")
        self.confidence.requires_grad_(phase == "confidence")


def load_structure_checkpoint(path, device="cpu", require_calibrated=True):
    """无兼容且完成校准的权重时拒绝指导；绝不使用随机结构头。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "idea8-event-structure-v1":
        raise ValueError("Not an idea8 structure checkpoint.")
    if require_calibrated and not checkpoint.get("confidence_trained", False):
        raise ValueError("Train the confidence phase before guided inference.")
    config = StructureHeadConfig(**checkpoint["head_config"])
    operator = StructureOperatorConfig(**checkpoint["operator_config"])
    if config.scales != len(operator.sigmas):
        raise ValueError("Structure head/operator scale mismatch.")
    model = EventStructurePredictor(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint, operator


def structure_checkpoint(model, operator, event_config, **metadata):
    return dict(format="idea8-event-structure-v1", head_config=asdict(model.config),
                operator_config=asdict(operator), event_config=asdict(event_config),
                state_dict=model.state_dict(), **metadata)
