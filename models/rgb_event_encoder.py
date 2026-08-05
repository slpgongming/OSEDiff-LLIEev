"""预训练 RGB-event 金字塔编码器及严格权重加载工具。"""

from dataclasses import asdict, dataclass
import hashlib
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_CHANNELS = (64, 128, 256, 512)
PRETRAINED_FILENAMES = {
    "encoder_i": "Encoder_I.pth",
    "encoder_e": "Encoder_E.pth",
    "fusion": "Merge.pth",
}


@dataclass(frozen=True)
class RGBEventEncoderConfig:
    event_channels: int = 5
    feature_channels: Tuple[int, ...] = FEATURE_CHANNELS
    rgb_input_range: str = "minus1_1_to_0_1_channel_flip"
    event_normalization: str = "nonzero_global_mean_std_v1"

    def to_dict(self):
        return asdict(self)


@dataclass
class RGBEventEncoderOutput:
    rgb_features: List[torch.Tensor]
    event_features: List[torch.Tensor]
    fused_features: List[torch.Tensor]


@dataclass(frozen=True)
class WeightLoadReport:
    component: str
    path: str
    loaded: int
    missing: Tuple[str, ...]
    unexpected: Tuple[str, ...]
    shape_mismatch: Tuple[str, ...]

    @property
    def ok(self):
        return not (self.missing or self.unexpected or self.shape_mismatch)


class ConvLayer(nn.Module):
    """与预训练权重一致的 Conv-(InstanceNorm)-ReLU 层。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        bias = norm != "BN"
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.activation = (
            getattr(torch.nn.functional, activation) if activation is not None else None
        )
        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=0.01)
        elif norm == "IN":
            # 预训练权重包含 running_mean/running_var/num_batches_tracked。
            self.norm_layer = nn.InstanceNorm2d(
                out_channels, track_running_stats=True
            )

    def forward(self, x):
        x = self.conv2d(x)
        if self.norm in {"BN", "IN"}:
            x = self.norm_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvLSTM(nn.Module):
    """单步 ConvLSTM；state 只在当前 forward 的局部变量中使用。"""

    def __init__(self, input_size, hidden_size, kernel_size=3):
        super().__init__()
        self.hidden_size = int(hidden_size)
        padding = kernel_size // 2
        self.Gates = nn.Conv2d(
            input_size + hidden_size,
            4 * hidden_size,
            kernel_size,
            padding=padding,
        )

    def forward(self, input_tensor, prev_state=None):
        batch_size, _, height, width = input_tensor.shape
        if prev_state is None:
            state_shape = (batch_size, self.hidden_size, height, width)
            prev_hidden = input_tensor.new_zeros(state_shape)
            prev_cell = input_tensor.new_zeros(state_shape)
        else:
            prev_hidden, prev_cell = (
                state.to(device=input_tensor.device, dtype=input_tensor.dtype)
                for state in prev_state
            )
        gates = self.Gates(torch.cat((input_tensor, prev_hidden), dim=1))
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, dim=1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        cell = remember_gate * prev_cell + in_gate * cell_gate
        hidden = out_gate * torch.tanh(cell)
        return hidden, cell


class RecurrentConvLayer(nn.Module):
    """5x5 stride-2 卷积后接 ConvLSTM，保持原 checkpoint 拓扑。"""

    def __init__(self, channels):
        super().__init__()
        self.recurrent_block_type = "convlstm"
        self.conv = ConvLayer(
            channels, channels, kernel_size=5, stride=2, padding=2
        )
        self.recurrent_block = ConvLSTM(channels, channels, kernel_size=3)

    def forward(self, x, prev_state=None):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        return state[0], state


class Conv2DBlock(nn.Module):
    """每个金字塔 level 的两层 Conv-IN-ReLU。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer(in_channels, out_channels, 3, padding=1, norm="IN"),
            ConvLayer(out_channels, out_channels, 3, padding=1, norm="IN"),
        )

    def forward(self, x):
        return self.conv(x)


class _PyramidEncoder(nn.Module):
    """四级编码器，输出 H/H2/H4/H8 的 64/128/256/512 通道特征。"""

    def __init__(self, in_channels):
        super().__init__()
        self.Maxpool1 = RecurrentConvLayer(64)
        self.Maxpool2 = RecurrentConvLayer(128)
        self.Maxpool3 = RecurrentConvLayer(256)
        self.Conv1 = Conv2DBlock(in_channels, 64)
        self.Conv2 = Conv2DBlock(64, 128)
        self.Conv3 = Conv2DBlock(128, 256)
        self.Conv4 = Conv2DBlock(256, 512)

    def forward(self, x):
        # Phase 1 按独立图像训练，每次调用都从空 state 开始，禁止跨 batch 泄漏。
        level0 = self.Conv1(x)
        level1, _ = self.Maxpool1(level0, None)
        level1 = self.Conv2(level1)
        level2, _ = self.Maxpool2(level1, None)
        level2 = self.Conv3(level2)
        level3, _ = self.Maxpool3(level2, None)
        level3 = self.Conv4(level3)
        return [level0, level1, level2, level3]


class RGBPyramidEncoder(_PyramidEncoder):
    def __init__(self):
        super().__init__(in_channels=3)


class EventPyramidEncoder(_PyramidEncoder):
    def __init__(self, event_channels=5):
        super().__init__(in_channels=event_channels)


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        scale = self.avg_pool(x).reshape(batch, channels)
        scale = self.fc(scale).reshape(batch, channels, 1, 1)
        return x * scale.expand_as(x)


class WeightBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)))


class MultiScaleFusion(nn.Module):
    """逐尺度融合 event/RGB；属性名保持与 48-key Merge 权重一致。"""

    def __init__(self, channels=FEATURE_CHANNELS):
        super().__init__()
        self.n_features = len(channels)
        self.w1 = nn.ModuleList()
        self.w2 = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for channel in channels:
            self.w1.append(WeightBlock(channel * 2, channel))
            self.w2.append(WeightBlock(channel * 2, channel))
            self.fuse.append(SqueezeExcitation(channel * 2))
            self.conv.append(ConvLayer(channel * 2, channel, 3, padding=1))

    def forward(self, event_features, rgb_features):
        if len(event_features) != self.n_features or len(rgb_features) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} levels, got "
                f"event={len(event_features)}, rgb={len(rgb_features)}."
            )
        outputs = []
        for index in range(self.n_features):
            if event_features[index].shape != rgb_features[index].shape:
                raise ValueError(
                    f"Level {index} RGB/event shape mismatch: "
                    f"{rgb_features[index].shape} vs {event_features[index].shape}."
                )
            joined = self.fuse[index](
                torch.cat((event_features[index], rgb_features[index]), dim=1)
            )
            event_weighted = event_features[index] * self.w1[index](joined)
            rgb_weighted = rgb_features[index] * self.w2[index](joined)
            outputs.append(
                self.conv[index](torch.cat((event_weighted, rgb_weighted), dim=1))
            )
        return outputs


def _safe_load_weights(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(payload)}.")
    return payload


def _load_component(module, path, component, strict=True):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {component} weights: {path}")
    source = _safe_load_weights(path)
    target = module.state_dict()
    compatible = {}
    unexpected = []
    shape_mismatch = []
    for key, value in source.items():
        if key not in target:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(target[key].shape):
            shape_mismatch.append(
                f"{key}: checkpoint={tuple(value.shape)}, model={tuple(target[key].shape)}"
            )
        else:
            compatible[key] = value
    missing = [key for key in target if key not in compatible]
    module.load_state_dict(compatible, strict=False)
    report = WeightLoadReport(
        component=component,
        path=os.path.abspath(path),
        loaded=len(compatible),
        missing=tuple(missing),
        unexpected=tuple(unexpected),
        shape_mismatch=tuple(shape_mismatch),
    )
    print(
        f"[RGBEventEncoder Load] {component}: loaded={report.loaded}, "
        f"missing={len(report.missing)}, unexpected={len(report.unexpected)}, "
        f"shape_mismatch={len(report.shape_mismatch)}"
    )
    if strict and not report.ok:
        raise RuntimeError(f"Strict {component} load failed: {report}")
    return report


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RGBEventEncoder(nn.Module):
    """冻结的双金字塔编码器，只为 VAE decoder refinement 产生条件。"""

    def __init__(self, event_channels=5):
        super().__init__()
        self.config = RGBEventEncoderConfig(event_channels=int(event_channels))
        self.rgb_encoder = RGBPyramidEncoder()
        self.event_encoder = EventPyramidEncoder(event_channels=event_channels)
        self.fusion = MultiScaleFusion()
        self.pretrained_manifest = None

    @staticmethod
    def adapt_rgb_input(rgb_minus1_1):
        rgb_01 = ((rgb_minus1_1 + 1.0) * 0.5).clamp(0.0, 1.0)
        return torch.flip(rgb_01, dims=[1])

    def forward(self, rgb_minus1_1, events_refinement, fusion_mode="both"):
        if fusion_mode not in {"both", "rgb", "event"}:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}.")
        rgb_features = self.rgb_encoder(self.adapt_rgb_input(rgb_minus1_1))
        event_features = self.event_encoder(events_refinement)
        fusion_rgb = rgb_features
        fusion_event = event_features
        if fusion_mode == "rgb":
            fusion_event = [torch.zeros_like(feature) for feature in event_features]
        elif fusion_mode == "event":
            fusion_rgb = [torch.zeros_like(feature) for feature in rgb_features]
        fused_features = self.fusion(fusion_event, fusion_rgb)
        return RGBEventEncoderOutput(
            rgb_features=rgb_features,
            event_features=event_features,
            fused_features=fused_features,
        )

    def load_pretrained(
        self,
        pretrained_root=None,
        encoder_i_path=None,
        encoder_e_path=None,
        fusion_path=None,
        strict=True,
    ):
        def resolve(explicit_path, key):
            if explicit_path:
                return explicit_path
            if not pretrained_root:
                raise ValueError(
                    f"Provide {key}_path or rgb_event_pretrained_root."
                )
            return os.path.join(pretrained_root, PRETRAINED_FILENAMES[key])

        paths = {
            "encoder_i": resolve(encoder_i_path, "encoder_i"),
            "encoder_e": resolve(encoder_e_path, "encoder_e"),
            "fusion": resolve(fusion_path, "fusion"),
        }
        reports = {
            "encoder_i": _load_component(
                self.rgb_encoder, paths["encoder_i"], "Encoder_I", strict=strict
            ),
            "encoder_e": _load_component(
                self.event_encoder, paths["encoder_e"], "Encoder_E", strict=strict
            ),
            "fusion": _load_component(
                self.fusion, paths["fusion"], "Merge", strict=strict
            ),
        }
        self.pretrained_manifest = {
            "encoder_i_filename": os.path.basename(paths["encoder_i"]),
            "encoder_e_filename": os.path.basename(paths["encoder_e"]),
            "fusion_filename": os.path.basename(paths["fusion"]),
            "sha256": {key: sha256_file(path) for key, path in paths.items()},
        }
        return reports, self.pretrained_manifest

    def verify_manifest(self, manifest, **load_kwargs):
        reports, actual = self.load_pretrained(**load_kwargs)
        expected_hashes = dict(manifest.get("sha256", {}))
        if expected_hashes and expected_hashes != actual["sha256"]:
            raise RuntimeError(
                "RGB-event pretrained hash mismatch: "
                f"expected={expected_hashes}, actual={actual['sha256']}"
            )
        return reports, actual

