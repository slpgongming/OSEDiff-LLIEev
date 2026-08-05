"""VAE decoder 的四级显式条件融合模块。"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.autoencoders.vae import Decoder


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

    def _validate_conditions(self, fused_features, fusion_enabled):
        if not fusion_enabled:
            return None
        if fused_features is None:
            raise ValueError("fusion_enabled=True requires fused_features.")
        if len(fused_features) != len(self.feature_fusion_blocks):
            raise ValueError(
                f"Expected {len(self.feature_fusion_blocks)} fused feature levels, "
                f"got {len(fused_features)}."
            )
        return list(reversed(fused_features))

    def forward(
        self,
        sample,
        latent_embeds=None,
        fused_features=None,
        fusion_enabled=False,
        fusion_weight=None,
    ):
        conditions = self._validate_conditions(fused_features, fusion_enabled)
        sample = self.conv_in(sample)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        if self.training and self.gradient_checkpointing:
            sample = self._checkpoint(self.mid_block, sample, latent_embeds)
            sample = sample.to(upscale_dtype)
            for index, up_block in enumerate(self.up_blocks):
                if conditions is not None:
                    sample = self.feature_fusion_blocks[index](
                        conditions[index], sample, fusion_weight=fusion_weight
                    )
                sample = self._checkpoint(up_block, sample, latent_embeds)
        else:
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)
            for index, up_block in enumerate(self.up_blocks):
                if conditions is not None:
                    sample = self.feature_fusion_blocks[index](
                        conditions[index], sample, fusion_weight=fusion_weight
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
        }

    def debug_stats(self):
        return {
            f"level_{index}/{key}": value
            for index, block in enumerate(self.feature_fusion_blocks)
            for key, value in block.last_stats.items()
        }
