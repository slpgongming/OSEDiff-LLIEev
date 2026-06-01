import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from references.submodules import ConvLayer, RecurrentConvLayer, SELayer, UpsampleConvLayer


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation="relu", norm=None):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm),
            ConvLayer(out_channels, out_channels, kernel_size, stride, padding, activation, norm),
        )

    def forward(self, x):
        return self.conv(x)


class WeightBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=None):
        super().__init__()
        bias = False if norm == "BN" else True
        self.norm = norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=bias)
        if norm == "BN":
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        if self.norm in ["BN", "IN"]:
            out = self.bn1(out)
        out = self.conv2(out)
        if self.norm in ["BN", "IN"]:
            out = self.bn2(out)
        return self.act(out)


class UEncoder4Recurrent(nn.Module):
    """
    HDRev 风格四层 recurrent U-Encoder。

    输入 shape: [B, C, H, W]；输出四层 feature:
    H1=[B,64,H,W], H2=[B,128,H/2,W/2], H3=[B,256,H/4,W/4], H4=[B,512,H/8,W/8]。
    """

    def __init__(self, in_channels=3):
        super().__init__()
        filters = [64, 128, 256, 512]
        # 保持与 HDRev checkpoint 一致的模块名，避免加载 Encoder_I/E 时 key 全部 miss。
        self.Maxpool1 = RecurrentConvLayer(filters[0], filters[0], kernel_size=5, padding=2, stride=2, recurrent_block_type="convlstm")
        self.Maxpool2 = RecurrentConvLayer(filters[1], filters[1], kernel_size=5, padding=2, stride=2, recurrent_block_type="convlstm")
        self.Maxpool3 = RecurrentConvLayer(filters[2], filters[2], kernel_size=5, padding=2, stride=2, recurrent_block_type="convlstm")
        self.Conv1 = Conv2DBlock(in_channels, filters[0], norm="IN")
        self.Conv2 = Conv2DBlock(filters[0], filters[1], norm="IN")
        self.Conv3 = Conv2DBlock(filters[1], filters[2], norm="IN")
        self.Conv4 = Conv2DBlock(filters[2], filters[3], norm="IN")

    def forward(self, x, prev_states=None):
        e1 = self.Conv1(x)
        if prev_states is None:
            prev_states = [None] * 3
        e2, state2 = self.Maxpool1(e1, prev_states[0])
        e2 = self.Conv2(e2)
        e3, state3 = self.Maxpool2(e2, prev_states[1])
        e3 = self.Conv3(e3)
        e4, state4 = self.Maxpool3(e3, prev_states[2])
        e4 = self.Conv4(e4)
        return [e1, e2, e3, e4], [state2, state3, state4]


class UDecoder4(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        filters = [64, 128, 256, 512]
        self.up4 = UpsampleConvLayer(filters[3], filters[2], kernel_size=5, padding=2, norm="IN")
        self.up_conv4 = Conv2DBlock(filters[3], filters[2], norm="IN")
        self.up3 = UpsampleConvLayer(filters[2], filters[1], kernel_size=5, padding=2, norm="IN")
        self.up_conv3 = Conv2DBlock(filters[2], filters[1], norm="IN")
        self.up2 = UpsampleConvLayer(filters[1], filters[0], kernel_size=5, padding=2, norm="IN")
        self.up_conv2 = Conv2DBlock(filters[1], filters[0], norm="IN")
        self.conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, features):
        e1, e2, e3, e4 = features
        d4 = self.up4(e4, e3.shape[2:])
        d4 = self.up_conv4(torch.cat((e3, d4), dim=1))
        d3 = self.up3(d4, e2.shape[2:])
        d3 = self.up_conv3(torch.cat((e2, d3), dim=1))
        d2 = self.up2(d3, e1.shape[2:])
        d2 = self.up_conv2(torch.cat((e1, d2), dim=1))
        return self.act(self.conv(d2))


class FusionLayerUnet(nn.Module):
    """逐尺度融合 event/image feature，输出 HDR feature pyramid。"""

    def __init__(self, n_features=4):
        super().__init__()
        filters = [64, 128, 256, 512]
        self.n_features = n_features
        self.w1 = nn.ModuleList()
        self.w2 = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(n_features):
            self.w1.append(WeightBlock(filters[i] * 2, filters[i]))
            self.w2.append(WeightBlock(filters[i] * 2, filters[i]))
            self.fuse.append(SELayer(channel=filters[i] * 2))
            self.conv.append(ConvLayer(filters[i] * 2, filters[i], kernel_size=3, padding=1))

    def forward(self, event_features, image_features):
        if self.n_features != len(event_features) or self.n_features != len(image_features):
            raise ValueError("FusionLayerUnet expects matching four-level event/image features.")
        out = []
        for i in range(self.n_features):
            cat = self.fuse[i](torch.cat([event_features[i], image_features[i]], dim=1))
            f_event = event_features[i] * self.w1[i](cat)
            f_image = image_features[i] * self.w2[i](cat)
            out.append(self.conv[i](torch.cat([f_event, f_image], dim=1)))
        return out


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "module"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _load_matching_state(module: nn.Module, state_dict: Dict, prefix: str = "") -> Dict[str, List[str]]:
    current = module.state_dict()
    loadable = {}
    shape_mismatched = []
    unexpected = []
    for key, value in state_dict.items():
        clean_key = key[len(prefix):] if prefix and key.startswith(prefix) else key
        clean_key = clean_key[7:] if clean_key.startswith("module.") else clean_key
        if clean_key not in current:
            unexpected.append(key)
        elif tuple(current[clean_key].shape) != tuple(value.shape):
            shape_mismatched.append(f"{clean_key}: ckpt={tuple(value.shape)} model={tuple(current[clean_key].shape)}")
        else:
            loadable[clean_key] = value

    missing = [key for key in current.keys() if key not in loadable]
    module.load_state_dict(loadable, strict=False)
    return {
        "loaded": list(loadable.keys()),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatched": shape_mismatched,
    }


class LLEEncoder(nn.Module):
    """
    Event-image low-light enhancement encoder.

    主路径只返回多尺度 HDR features，不默认使用 decoder 生成 restored image；
    decoder 仅保留给 RestCond 消融或可视化。
    """

    feature_channels = (64, 128, 256, 512)

    def __init__(self, num_bins=5, use_decoder=False):
        super().__init__()
        self.netEncoder_I = UEncoder4Recurrent(in_channels=3)
        self.netEncoder_E = UEncoder4Recurrent(in_channels=num_bins)
        self.netMerge = FusionLayerUnet(n_features=4)
        self.netDecoder = UDecoder4(out_channels=3) if use_decoder else None

    def forward(self, low_rgb, event_voxel, return_img=False):
        # low_rgb 使用 [0,1] RGB；HDRev 原实现 flip 到 BGR，这里保留该权重兼容路径。
        image_features, _ = self.netEncoder_I(torch.flip(low_rgb, dims=[1]), None)
        event_features, _ = self.netEncoder_E(event_voxel, None)
        hdr_features = self.netMerge(event_features, image_features)
        if return_img:
            if self.netDecoder is None:
                raise RuntimeError("LLEEncoder was built without decoder; set use_decoder=True for RestCond visualization.")
            restored = torch.flip(self.netDecoder(hdr_features), dims=[1])
            return hdr_features, restored
        return hdr_features

    def load_pretrained(self, pretrained_root: Optional[str], logger=print) -> Dict[str, Dict[str, List[str]]]:
        if not pretrained_root:
            return {}

        paths = {
            "netEncoder_I": os.path.join(pretrained_root, "Encoder_I.pth"),
            "netEncoder_E": os.path.join(pretrained_root, "Encoder_E.pth"),
            "netMerge": os.path.join(pretrained_root, "Merge.pth"),
        }
        if self.netDecoder is not None:
            paths["netDecoder"] = os.path.join(pretrained_root, "Decoder.pth")

        report = {}
        modules = {
            "netEncoder_I": self.netEncoder_I,
            "netEncoder_E": self.netEncoder_E,
            "netMerge": self.netMerge,
            "netDecoder": self.netDecoder,
        }
        for name, path in paths.items():
            if not os.path.exists(path):
                logger(f"[LLEEncoder] Skip missing pretrained file: {path}")
                continue
            state_dict = _extract_state_dict(_torch_load(path))
            module_report = _load_matching_state(modules[name], state_dict)
            report[name] = module_report
            logger(
                f"[LLEEncoder] {name}: loaded={len(module_report['loaded'])}, "
                f"missing={len(module_report['missing'])}, unexpected={len(module_report['unexpected'])}, "
                f"shape_mismatched={len(module_report['shape_mismatched'])}"
            )
        return report
