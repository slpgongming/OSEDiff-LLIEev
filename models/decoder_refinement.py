"""VAE decoder 的四级显式条件融合模块。"""

from dataclasses import dataclass
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.autoencoders.vae import Decoder


REFINEMENT_VARIANTS = ("legacy_fused", "cgdrs")
CGDRS_ACTIVITY_MODES = ("abs_mean", "abs_sum")


def normalize_refinement_variant(value, allow_auto=False):
    """统一 decoder refinement variant 名称，避免 checkpoint/CLI 使用不同拼写。"""
    value = str(value or "").strip().lower()
    allowed = set(REFINEMENT_VARIANTS)
    if allow_auto:
        allowed.add("auto")
    if value not in allowed:
        raise ValueError(
            f"Unsupported decoder refinement variant={value!r}; "
            f"expected one of {sorted(allowed)}."
        )
    return value


def normalize_refinement_levels(value):
    """解析 CLI 的逗号列表或 Python 序列，返回去重后的 level tuple。"""
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    levels = tuple(sorted({int(level) for level in value}))
    return levels


def normalize_activity_mode(value):
    """统一 CGDRS event activity 聚合方式。"""
    value = str(value or "abs_mean").strip().lower()
    if value not in CGDRS_ACTIVITY_MODES:
        raise ValueError(
            f"Unsupported CGDRS activity mode={value!r}; "
            f"expected one of {CGDRS_ACTIVITY_MODES}."
        )
    return value


@dataclass
class DecoderRefinementCondition:
    """decoder refinement 的多模态条件；legacy 只需要 fused_features。"""

    fused_features: Optional[List[torch.Tensor]] = None
    rgb_features: Optional[List[torch.Tensor]] = None
    event_features: Optional[List[torch.Tensor]] = None
    event_activity: Optional[List[torch.Tensor]] = None


def _largest_divisor_at_most(channels, upper=32):
    for groups in range(min(int(channels), int(upper)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FeatureFusionResBlock(nn.Module):
    """GroupNorm-SiLU-Conv 残差块，支持输入/输出通道投影。"""

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = int(out_channels or in_channels)
        in_channels = int(in_channels)
        self.norm = nn.GroupNorm(
            num_groups=_largest_divisor_at_most(in_channels),
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        hidden = self.norm(x)
        hidden = hidden * torch.sigmoid(hidden)
        hidden = self.conv(hidden)
        return hidden + residual


class DecoderFeatureFusionBlock(nn.Module):
    """把一个 RGB-event pyramid level 融合到对应 decoder up-block 输入。"""

    def __init__(
        self,
        condition_channels,
        decoder_channels,
        fusion_weight=1.0,
        init="near_zero",
        allow_interpolation=True,
    ):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.decoder_channels = int(decoder_channels)
        self.fusion_weight = float(fusion_weight)
        self.allow_interpolation = bool(allow_interpolation)
        self.condition_projection = nn.Sequential(
            nn.Conv2d(self.condition_channels, self.condition_channels, 5, padding=2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(self.condition_channels, self.decoder_channels, 3, padding=1),
        )
        self.fusion_resblock = FeatureFusionResBlock(
            self.decoder_channels * 2, self.decoder_channels
        )
        self.output_resblock = FeatureFusionResBlock(
            self.decoder_channels, self.decoder_channels
        )
        self.last_stats = {}
        self.forward_count = 0
        if init != "near_zero":
            raise ValueError(f"Unsupported decoder fusion init: {init}")
        self._near_zero_init()

    def _near_zero_init(self):
        # 1e-6 近零卷积让初始 decoder 基本退化为原 VAE，同时保留非零梯度。
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=1e-6)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, condition, decoder_feature, fusion_weight=None):
        if condition.ndim != 4 or decoder_feature.ndim != 4:
            raise ValueError(
                "Decoder fusion expects BCHW tensors, got "
                f"condition={condition.shape}, decoder={decoder_feature.shape}."
            )
        if condition.shape[0] != decoder_feature.shape[0]:
            raise ValueError(
                f"Decoder fusion batch mismatch: {condition.shape[0]} vs "
                f"{decoder_feature.shape[0]}."
            )
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"Decoder condition channel mismatch: expected={self.condition_channels}, "
                f"got={condition.shape[1]}."
            )
        if decoder_feature.shape[1] != self.decoder_channels:
            raise ValueError(
                f"Decoder feature channel mismatch: expected={self.decoder_channels}, "
                f"got={decoder_feature.shape[1]}."
            )
        if condition.shape[-2:] != decoder_feature.shape[-2:]:
            if not self.allow_interpolation:
                raise ValueError(
                    "Decoder condition spatial mismatch: "
                    f"condition={condition.shape[-2:]}, decoder={decoder_feature.shape[-2:]}."
                )
            warnings.warn(
                "Interpolating decoder condition from "
                f"{condition.shape[-2:]} to {decoder_feature.shape[-2:]}",
                RuntimeWarning,
                stacklevel=2,
            )
            condition = F.interpolate(
                condition,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        condition = condition.to(
            device=decoder_feature.device, dtype=decoder_feature.dtype
        )
        projected = self.condition_projection(condition)
        delta = self.fusion_resblock(
            torch.cat((projected, decoder_feature), dim=1)
        )
        delta = self.output_resblock(delta)
        weight = self.fusion_weight if fusion_weight is None else float(fusion_weight)
        output = decoder_feature + weight * delta
        self.forward_count += 1
        with torch.no_grad():
            decoder_norm = decoder_feature.detach().float().norm().clamp_min(1e-12)
            self.last_stats = {
                "decoder_before_mean": decoder_feature.detach().float().mean().item(),
                "decoder_before_std": decoder_feature.detach().float().std(unbiased=False).item(),
                "decoder_after_mean": output.detach().float().mean().item(),
                "decoder_after_std": output.detach().float().std(unbiased=False).item(),
                "fusion_residual_norm": delta.detach().float().norm().item(),
                "fusion_residual_ratio": (
                    delta.detach().float().norm() / decoder_norm
                ).item(),
            }
        return output


class _CGDRSBranch(nn.Module):
    """CGDRS 的轻量卷积残差分支；仅最后一层零初始化。"""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(
                num_groups=_largest_divisor_at_most(hidden_channels),
                num_channels=hidden_channels,
                eps=1e-6,
            ),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(hidden_channels, out_channels, 1)
        # 只将最终输出置零，保证初始时不改变原 decoder，同时让前级投影有梯度。
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(self.body(x))


class ConfidenceGatedDualResidualSkipBlock(nn.Module):
    """RGB 主导、event 可信度门控的双残差 decoder skip。

    ``rgb_features`` 负责内容和颜色细化；``event_features`` 只能通过 spatial
    gate 控制的残差进入 decoder。event activity 为零时，event residual 被严格
    屏蔽，便于 zero/shuffled event 消融并保护原始 VAE 分布。
    """

    def __init__(
        self,
        condition_channels,
        decoder_channels,
        fusion_weight=1.0,
        gate_bias=-2.0,
        event_enabled=True,
        allow_interpolation=True,
    ):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.decoder_channels = int(decoder_channels)
        self.fusion_weight = float(fusion_weight)
        self.gate_bias = float(gate_bias)
        self.event_enabled = bool(event_enabled)
        self.allow_interpolation = bool(allow_interpolation)
        self.hidden_channels = min(128, max(32, self.decoder_channels))

        hidden = self.hidden_channels
        self.decoder_projection = nn.Conv2d(self.decoder_channels, hidden, 1)
        self.rgb_projection = nn.Conv2d(self.condition_channels, hidden, 1)
        self.event_projection = nn.Conv2d(self.condition_channels, hidden, 1)

        self.rgb_branch = _CGDRSBranch(hidden * 2, hidden, self.decoder_channels)
        self.event_branch = _CGDRSBranch(hidden * 3, hidden, self.decoder_channels)
        self.gate_body = nn.Sequential(
            nn.Conv2d(hidden * 3 + 1, hidden, 3, padding=1),
            nn.GroupNorm(
                num_groups=_largest_divisor_at_most(hidden),
                num_channels=hidden,
                eps=1e-6,
            ),
            nn.SiLU(),
        )
        self.gate_output = nn.Conv2d(hidden, 1, 1)
        # gate 初始为 sigmoid(gate_bias)，且不被随机纹理驱动。
        nn.init.zeros_(self.gate_output.weight)
        nn.init.constant_(self.gate_output.bias, self.gate_bias)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.last_stats = {}
        self.forward_count = 0

    @staticmethod
    def _resize_like(condition, decoder_feature, allow_interpolation):
        if condition is None:
            return None
        if condition.shape[-2:] == decoder_feature.shape[-2:]:
            return condition
        if not allow_interpolation:
            raise ValueError(
                "CGDRS condition spatial mismatch: "
                f"condition={condition.shape[-2:]}, decoder={decoder_feature.shape[-2:]}"
            )
        return F.interpolate(
            condition,
            size=decoder_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        rgb_condition,
        event_condition,
        decoder_feature,
        event_activity=None,
        fusion_weight=None,
        fusion_mode="both",
    ):
        if decoder_feature.ndim != 4:
            raise ValueError(
                f"CGDRS decoder feature must be BCHW, got {decoder_feature.shape}."
            )
        if fusion_mode not in {"both", "rgb", "event"}:
            raise ValueError(f"Unsupported CGDRS fusion_mode={fusion_mode}.")
        if rgb_condition is None or event_condition is None:
            raise ValueError("CGDRS requires separate RGB and event conditions.")
        for name, condition in (("rgb", rgb_condition), ("event", event_condition)):
            if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
                raise ValueError(
                    f"CGDRS {name} condition channel mismatch: "
                    f"expected BCHW with C={self.condition_channels}, got {condition.shape}."
                )
            if condition.shape[0] != decoder_feature.shape[0]:
                raise ValueError(
                    f"CGDRS {name} condition batch mismatch: "
                    f"{condition.shape[0]} vs {decoder_feature.shape[0]}."
                )
        rgb_condition = self._resize_like(
            rgb_condition, decoder_feature, self.allow_interpolation
        )
        event_condition = self._resize_like(
            event_condition, decoder_feature, self.allow_interpolation
        )
        if event_activity is None:
            event_activity = torch.ones(
                decoder_feature.shape[0],
                1,
                *decoder_feature.shape[-2:],
                device=decoder_feature.device,
                dtype=decoder_feature.dtype,
            )
        else:
            event_activity = self._resize_like(
                event_activity, decoder_feature, self.allow_interpolation
            )
            if event_activity.shape[1] != 1:
                raise ValueError(
                    f"CGDRS activity must have one channel, got {event_activity.shape}."
                )
            if event_activity.shape[0] != decoder_feature.shape[0]:
                raise ValueError(
                    "CGDRS activity batch mismatch: "
                    f"{event_activity.shape[0]} vs {decoder_feature.shape[0]}."
                )
            event_activity = event_activity.to(
                device=decoder_feature.device, dtype=decoder_feature.dtype
            )

        decoder_feature = decoder_feature
        rgb_projected = self.rgb_projection(
            rgb_condition.to(device=decoder_feature.device, dtype=decoder_feature.dtype)
        )
        event_projected = self.event_projection(
            event_condition.to(device=decoder_feature.device, dtype=decoder_feature.dtype)
        )
        decoder_projected = self.decoder_projection(decoder_feature)
        if fusion_mode == "rgb":
            event_projected = torch.zeros_like(event_projected)
            event_activity = torch.zeros_like(event_activity)
        elif fusion_mode == "event":
            rgb_projected = torch.zeros_like(rgb_projected)

        rgb_delta = self.rgb_branch(torch.cat((decoder_projected, rgb_projected), dim=1))
        event_delta = self.event_branch(
            torch.cat((decoder_projected, rgb_projected, event_projected), dim=1)
        )
        gate_input = torch.cat(
            (decoder_projected, rgb_projected, event_projected, event_activity), dim=1
        )
        gate = torch.sigmoid(self.gate_output(self.gate_body(gate_input)))
        gate = gate * event_activity.clamp(0.0, 1.0)
        if fusion_mode == "rgb" or not self.event_enabled:
            event_delta = torch.zeros_like(event_delta)
            gate = torch.zeros_like(gate)
        if fusion_mode == "event":
            rgb_delta = torch.zeros_like(rgb_delta)

        weight = self.fusion_weight if fusion_weight is None else float(fusion_weight)
        delta = self.alpha.to(dtype=decoder_feature.dtype) * (
            rgb_delta + gate * event_delta
        )
        output = decoder_feature + weight * delta
        self.forward_count += 1
        with torch.no_grad():
            decoder_norm = decoder_feature.detach().float().norm().clamp_min(1e-12)
            self.last_stats = {
                "decoder_before_mean": decoder_feature.detach().float().mean().item(),
                "decoder_before_std": decoder_feature.detach().float().std(unbiased=False).item(),
                "decoder_after_mean": output.detach().float().mean().item(),
                "decoder_after_std": output.detach().float().std(unbiased=False).item(),
                "gate_mean": gate.detach().float().mean().item(),
                "gate_std": gate.detach().float().std(unbiased=False).item(),
                "activity_mean": event_activity.detach().float().mean().item(),
                "rgb_residual_ratio": (rgb_delta.detach().float().norm() / decoder_norm).item(),
                "event_residual_ratio": (
                    (gate * event_delta).detach().float().norm() / decoder_norm
                ).item(),
                "output_delta_ratio": (delta.detach().float().norm() / decoder_norm).item(),
                "alpha": self.alpha.detach().float().item(),
            }
        return output


def decoder_input_channels(block_out_channels):
    """返回每个 up-block 之前的 decoder feature 通道。"""
    reversed_channels = list(reversed(tuple(int(c) for c in block_out_channels)))
    output_channel = reversed_channels[0]
    inputs = []
    for index in range(len(reversed_channels)):
        prev_output_channel = output_channel
        output_channel = reversed_channels[index]
        inputs.append(prev_output_channel)
    return tuple(inputs)


class FeatureAwareDecoder(Decoder):
    """保持原 Decoder backbone key/shape，只新增四级 feature fusion blocks。"""

    def __init__(
        self,
        *args,
        condition_channels=(64, 128, 256, 512),
        fusion_weight=1.0,
        fusion_init="near_zero",
        refinement_variant="legacy_fused",
        cgdrs_event_levels=(1, 2, 3),
        cgdrs_gate_bias=-2.0,
        cgdrs_activity_mode="abs_mean",
        allow_condition_interpolation=True,
        **kwargs,
    ):
        block_out_channels = tuple(kwargs.get("block_out_channels", (64,)))
        super().__init__(*args, **kwargs)
        decoder_channels = decoder_input_channels(block_out_channels)
        condition_channels = tuple(int(c) for c in condition_channels)
        if len(condition_channels) != len(self.up_blocks):
            raise ValueError(
                f"condition levels={len(condition_channels)} do not match "
                f"decoder up-blocks={len(self.up_blocks)}."
            )
        self.condition_channels = condition_channels
        self.decoder_channels = decoder_channels
        self.fusion_weight = float(fusion_weight)
        self.refinement_variant = normalize_refinement_variant(refinement_variant)
        self.cgdrs_event_levels = normalize_refinement_levels(cgdrs_event_levels)
        if any(level < 0 or level >= len(condition_channels) for level in self.cgdrs_event_levels):
            raise ValueError(
                f"Invalid CGDRS event levels={self.cgdrs_event_levels}; "
                f"expected [0, {len(condition_channels) - 1}]."
            )
        self.cgdrs_gate_bias = float(cgdrs_gate_bias)
        self.cgdrs_activity_mode = normalize_activity_mode(cgdrs_activity_mode)
        if self.refinement_variant == "legacy_fused":
            self.feature_fusion_blocks = nn.ModuleList(
                DecoderFeatureFusionBlock(
                    condition_channels=cond_channels,
                    decoder_channels=dec_channels,
                    fusion_weight=fusion_weight,
                    init=fusion_init,
                    allow_interpolation=allow_condition_interpolation,
                )
                for cond_channels, dec_channels in zip(
                    reversed(condition_channels), decoder_channels
                )
            )
        else:
            # decoder 从 deepest level 开始，condition 也需按同样顺序反转。
            self.feature_fusion_blocks = nn.ModuleList(
                ConfidenceGatedDualResidualSkipBlock(
                    condition_channels=cond_channels,
                    decoder_channels=dec_channels,
                    fusion_weight=fusion_weight,
                    gate_bias=cgdrs_gate_bias,
                    event_enabled=level in self.cgdrs_event_levels,
                    allow_interpolation=allow_condition_interpolation,
                )
                for level, (cond_channels, dec_channels) in zip(
                    reversed(range(len(condition_channels))),
                    zip(reversed(condition_channels), decoder_channels),
                )
            )

    @staticmethod
    def _checkpoint(module, *inputs):
        def custom_forward(*values):
            return module(*values)

        try:
            return torch.utils.checkpoint.checkpoint(
                custom_forward, *inputs, use_reentrant=False
            )
        except TypeError:
            return torch.utils.checkpoint.checkpoint(custom_forward, *inputs)

    def _validate_conditions(self, fused_features, rgb_features, event_features, event_activity, fusion_enabled):
        if not fusion_enabled:
            return None
        if self.refinement_variant == "legacy_fused":
            if fused_features is None:
                raise ValueError("legacy_fused requires fused_features.")
            if len(fused_features) != len(self.feature_fusion_blocks):
                raise ValueError(
                    f"Expected {len(self.feature_fusion_blocks)} fused feature levels, "
                    f"got {len(fused_features)}."
                )
            return {"fused": list(reversed(fused_features))}
        for name, features in (("rgb", rgb_features), ("event", event_features)):
            if features is None or len(features) != len(self.feature_fusion_blocks):
                raise ValueError(
                    f"cgdrs requires {len(self.feature_fusion_blocks)} {name} feature levels, "
                    f"got {None if features is None else len(features)}."
                )
        if event_activity is None or len(event_activity) != len(self.feature_fusion_blocks):
            raise ValueError(
                f"cgdrs requires {len(self.feature_fusion_blocks)} activity levels, "
                f"got {None if event_activity is None else len(event_activity)}."
            )
        return {
            "rgb": list(reversed(rgb_features)),
            "event": list(reversed(event_features)),
            "activity": list(reversed(event_activity)),
        }

    def forward(
        self,
        sample,
        latent_embeds=None,
        fused_features=None,
        rgb_features=None,
        event_features=None,
        event_activity=None,
        fusion_enabled=False,
        fusion_weight=None,
        fusion_mode="both",
    ):
        conditions = self._validate_conditions(
            fused_features,
            rgb_features,
            event_features,
            event_activity,
            fusion_enabled,
        )
        sample = self.conv_in(sample)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        if self.training and self.gradient_checkpointing:
            sample = self._checkpoint(self.mid_block, sample, latent_embeds)
            sample = sample.to(upscale_dtype)
            for index, up_block in enumerate(self.up_blocks):
                if conditions is not None:
                    if self.refinement_variant == "legacy_fused":
                        sample = self.feature_fusion_blocks[index](
                            conditions["fused"][index], sample, fusion_weight=fusion_weight
                        )
                    else:
                        sample = self.feature_fusion_blocks[index](
                            conditions["rgb"][index],
                            conditions["event"][index],
                            sample,
                            event_activity=conditions["activity"][index],
                            fusion_weight=fusion_weight,
                            fusion_mode=fusion_mode,
                        )
                sample = self._checkpoint(up_block, sample, latent_embeds)
        else:
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)
            for index, up_block in enumerate(self.up_blocks):
                if conditions is not None:
                    if self.refinement_variant == "legacy_fused":
                        sample = self.feature_fusion_blocks[index](
                            conditions["fused"][index], sample, fusion_weight=fusion_weight
                        )
                    else:
                        sample = self.feature_fusion_blocks[index](
                            conditions["rgb"][index],
                            conditions["event"][index],
                            sample,
                            event_activity=conditions["activity"][index],
                            fusion_weight=fusion_weight,
                            fusion_mode=fusion_mode,
                        )
                sample = up_block(sample, latent_embeds)

        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        return self.conv_out(sample)

    def refinement_config(self):
        return {
            "condition_channels": list(self.condition_channels),
            "decoder_channels": list(self.decoder_channels),
            "fusion_blocks": len(self.feature_fusion_blocks),
            "fusion_weight": self.fusion_weight,
            "fusion_init": "near_zero",
            "variant": self.refinement_variant,
            "config_version": 2,
            "cgdrs_event_levels": list(self.cgdrs_event_levels),
            "cgdrs_gate_bias": self.cgdrs_gate_bias,
            "cgdrs_activity_mode": self.cgdrs_activity_mode,
            "cgdrs_activity_clip": 4.0,
        }

    def debug_stats(self):
        return {
            f"level_{index}/{key}": value
            for index, block in enumerate(self.feature_fusion_blocks)
            for key, value in block.last_stats.items()
        }
