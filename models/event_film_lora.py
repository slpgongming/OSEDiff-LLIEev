import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


EVENT_LORA_VARIANTS = (
    "film_affine",
    "cbam_lowrank_residual",
    "cbam_output_gate",
)


@dataclass(frozen=True)
class EventLoRAConfig:
    """Event-LoRA 的结构配置；训练、推理和 regularizer 必须使用同一份语义。"""

    variant: str = "film_affine"
    rank: int = 4
    alpha: float = 4.0
    mod_channels: int = 32
    modulation_scale: float = 0.1
    cbam_reduction: int = 8
    cbam_spatial_kernel: int = 7
    output_gate_channels: str = "spatial"
    zero_init_last: bool = True

    def __post_init__(self):
        if self.variant not in EVENT_LORA_VARIANTS:
            raise ValueError(f"Unsupported Event-LoRA variant: {self.variant}")
        if self.rank <= 0 or self.mod_channels <= 0:
            raise ValueError("rank and mod_channels must be positive.")
        if self.cbam_reduction <= 0:
            raise ValueError("cbam_reduction must be positive.")
        if self.cbam_spatial_kernel not in {3, 7}:
            raise ValueError("cbam_spatial_kernel must be 3 or 7.")
        if self.output_gate_channels not in {"spatial", "channel_spatial"}:
            raise ValueError("output_gate_channels must be spatial or channel_spatial.")

    def to_dict(self):
        return asdict(self)


class ChannelAttention(nn.Module):
    """
    CBAM 通道注意力。

    `reduction` 控制通道 MLP 的压缩比例：隐藏通道数约为
    `channels / reduction`。值越大，参数量和计算量越小，但通道间关系的
    表达能力也更弱；本项目默认 8，在 mod_channels=32 时隐藏通道为 4。
    avg/max pooling 分别概括整体响应和最强响应，二者共享同一组 MLP 权重。
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg_score = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_score = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg_score + max_score)


class SpatialAttention(nn.Module):
    """
    CBAM 空间注意力。

    `kernel_size` 决定空间注意力的局部感受野。7x7 能利用更大范围的事件边缘
    与暗区上下文，通常比 3x3 更适合当前 4~32 尺度特征；3x3 更省计算，
    可用于高分辨率或显存紧张实验。输入先沿通道做 mean/max 压缩，因此卷积
    始终只处理 2 个输入通道，kernel 增大不会像普通高维卷积那样显著增参。
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        attention = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention


class CBAM(nn.Module):
    """
    Channel Attention -> Spatial Attention。

    `channels` 等于融合头的 `mod_channels`，它决定 CBAM 中间特征宽度：
    较大时能承载更丰富的 RGB-LoRA/event 交互，但所有 3x3 融合卷积的开销
    也随通道数近似平方增长。CBAM 只帮助生成 delta/gate，不替代低秩 LoRA。
    """

    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class _BaseEventModulator(nn.Module):
    def __init__(self, rank, event_channels_by_size, config):
        super().__init__()
        self.rank = rank
        self.config = config
        self.event_proj = nn.ModuleDict({
            key: nn.Conv2d(channels, config.mod_channels, kernel_size=1)
            for key, channels in event_channels_by_size.items()
        })

    def project_event(self, event_feat, event_key, spatial_size, dtype):
        if event_feat.shape[-2:] != spatial_size:
            event_feat = F.interpolate(
                event_feat, size=spatial_size, mode="bilinear", align_corners=False
            )
        return self.event_proj[event_key](event_feat.to(dtype=dtype))

    @staticmethod
    def _zero_init(module):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FiLMAffineModulator(_BaseEventModulator):
    """在 LoRA bottleneck z 上预测空间相关 dgamma/dbeta。"""

    def __init__(self, rank, event_channels_by_size, config):
        super().__init__(rank, event_channels_by_size, config)
        self.z_proj = nn.Conv2d(rank, config.mod_channels, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(config.mod_channels * 2, config.mod_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.mod_channels, rank * 2, kernel_size=1),
        )
        if config.zero_init_last:
            self._zero_init(self.fusion[-1])

    def forward(self, z_map, event_feat, event_key, u_map=None):
        event_proj = self.project_event(
            event_feat, event_key, z_map.shape[-2:], z_map.dtype
        )
        fused = torch.cat([self.z_proj(z_map), event_proj], dim=1)
        dgamma, dbeta = self.fusion(fused).chunk(2, dim=1)
        scale = self.config.modulation_scale
        z_mod = (1.0 + scale * torch.tanh(dgamma)) * z_map + scale * dbeta
        return {
            "z_mod": z_mod,
            "u_mod": None,
            "aux": {"dgamma": dgamma, "dbeta": dbeta},
        }


class CBAMLowRankResidualModulator(_BaseEventModulator):
    """
    在低秩空间预测 additive residual `delta_z`。

    z/event 先分别投影到 `mod_channels`，融合后通过 CBAM 选择有用的通道和
    空间位置，最后投影回 rank 通道。末层 zero-init 令初始 delta_z=0，
    因而安装后与原普通 LoRA 严格保持同一计算形式。
    """

    def __init__(self, rank, event_channels_by_size, config):
        super().__init__(rank, event_channels_by_size, config)
        self.z_proj = nn.Conv2d(rank, config.mod_channels, kernel_size=1)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(config.mod_channels * 2, config.mod_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.cbam = CBAM(
            config.mod_channels,
            reduction=config.cbam_reduction,
            spatial_kernel=config.cbam_spatial_kernel,
        )
        self.delta_z_head = nn.Conv2d(config.mod_channels, rank, kernel_size=1)
        if config.zero_init_last:
            self._zero_init(self.delta_z_head)

    def forward(self, z_map, event_feat, event_key, u_map=None):
        event_proj = self.project_event(
            event_feat, event_key, z_map.shape[-2:], z_map.dtype
        )
        fused = self.fuse_conv(torch.cat([self.z_proj(z_map), event_proj], dim=1))
        delta_z = self.delta_z_head(self.cbam(fused))
        z_mod = z_map + self.config.modulation_scale * delta_z
        return {
            "z_mod": z_mod,
            "u_mod": None,
            "aux": {"delta_z": delta_z},
        }


class CBAMOutputGateModulator(_BaseEventModulator):
    """在 LoRA 输出 u=B(Ax) 上预测受限的乘性 gate。"""

    def __init__(self, rank, out_features, event_channels_by_size, config):
        super().__init__(rank, event_channels_by_size, config)
        self.u_proj = nn.Conv2d(out_features, config.mod_channels, kernel_size=1)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(config.mod_channels * 2, config.mod_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.cbam = CBAM(
            config.mod_channels,
            reduction=config.cbam_reduction,
            spatial_kernel=config.cbam_spatial_kernel,
        )
        # spatial 只输出一个共享空间 gate，远比为 320/640/1280 通道各预测 gate 轻量。
        gate_channels = 1 if config.output_gate_channels == "spatial" else out_features
        self.gate_head = nn.Conv2d(config.mod_channels, gate_channels, kernel_size=1)
        if config.zero_init_last:
            self._zero_init(self.gate_head)

    def forward(self, z_map, event_feat, event_key, u_map=None):
        if u_map is None:
            raise ValueError("cbam_output_gate requires u_map.")
        event_proj = self.project_event(
            event_feat, event_key, u_map.shape[-2:], u_map.dtype
        )
        fused = self.fuse_conv(torch.cat([self.u_proj(u_map), event_proj], dim=1))
        gate = torch.tanh(self.gate_head(self.cbam(fused)))
        u_mod = u_map * (1.0 + self.config.modulation_scale * gate)
        return {"z_mod": None, "u_mod": u_mod, "aux": {"gate": gate}}


class SpatialRGBEventLoRALinear(nn.Module):
    """
    用 Event-LoRA 替换 UNet self-attention `attn1.to_v`。

    输入为 `[B,N,C]` 图像 token。FiLM/low-rank residual 在 `lora_up` 前
    修改 rank 通道特征；output gate 在 `lora_up` 后修改 LoRA residual。
    base Linear 始终冻结，且没有事件特征时会退化为普通 LoRA。
    """

    def __init__(self, base_linear, config, event_channels_by_size):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.requires_grad_(False)
        self.config = config
        self.rank = config.rank
        self.lora_alpha = config.alpha
        self.scaling = self.lora_alpha / max(self.rank, 1)
        self.event_channels_by_size = event_channels_by_size
        self.event_feats = None
        self.modulation_enabled = True
        self.last_debug = {}
        self.last_aux = {}

        self.lora_down = nn.Linear(base_linear.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        if config.variant == "film_affine":
            self.modulator = FiLMAffineModulator(
                self.rank, event_channels_by_size, config
            )
        elif config.variant == "cbam_lowrank_residual":
            self.modulator = CBAMLowRankResidualModulator(
                self.rank, event_channels_by_size, config
            )
        else:
            self.modulator = CBAMOutputGateModulator(
                self.rank, base_linear.out_features, event_channels_by_size, config
            )

    @property
    def variant(self):
        return self.config.variant

    def set_event_feats(self, event_feats):
        self.event_feats = event_feats

    def clear_event_feats(self):
        self.event_feats = None

    def set_modulation_enabled(self, enabled):
        self.modulation_enabled = bool(enabled)

    def forward(self, hidden_states):
        base_out = self.base_linear(hidden_states)
        z = self.lora_down(hidden_states)
        if not self.modulation_enabled or self.event_feats is None or hidden_states.dim() != 3:
            lora_out = self.lora_up(z) * self.scaling
            self._record_debug({}, base_out, lora_out)
            return base_out + lora_out

        bsz, token_count, _ = hidden_states.shape
        spatial = int(math.sqrt(token_count))
        if spatial * spatial != token_count:
            raise ValueError(
                f"Event-LoRA 当前假设 self-attention token 为方形网格，收到 N={token_count}。"
            )
        event_key = self._select_event_key(spatial)
        event_feat = self.event_feats[event_key]
        z_map = z.transpose(1, 2).reshape(bsz, self.rank, spatial, spatial)

        if self.variant == "cbam_output_gate":
            u_tokens = self.lora_up(z) * self.scaling
            u_map = u_tokens.transpose(1, 2).reshape(
                bsz, self.base_linear.out_features, spatial, spatial
            )
            result = self.modulator(z_map, event_feat, event_key, u_map=u_map)
            lora_out = result["u_mod"].flatten(2).transpose(1, 2).contiguous()
        else:
            result = self.modulator(z_map, event_feat, event_key)
            z_tokens = result["z_mod"].flatten(2).transpose(1, 2).contiguous()
            lora_out = self.lora_up(z_tokens) * self.scaling

        self._record_debug(result["aux"], base_out, lora_out)
        return base_out + lora_out

    def _select_event_key(self, spatial):
        key = str(spatial)
        if key in self.event_feats and key in self.event_channels_by_size:
            return key
        available = [
            k for k in self.event_feats.keys() if k in self.event_channels_by_size
        ]
        if not available:
            raise RuntimeError("Event-LoRA 没有可用的事件特征尺度。")
        return min(available, key=lambda k: abs(int(k) - spatial))

    def _record_debug(self, aux, base_out, lora_out):
        self.last_aux = aux
        debug = {
            "base_norm": float(base_out.detach().float().norm().item()),
            "lora_norm": float(lora_out.detach().float().norm().item()),
        }
        for name, tensor in aux.items():
            detached = tensor.detach().float()
            if name == "gate":
                debug.update({
                    "gate_mean": float(detached.mean().item()),
                    "gate_std": float(detached.std(unbiased=False).item()),
                    "gate_min": float(detached.min().item()),
                    "gate_max": float(detached.max().item()),
                })
            else:
                debug[f"{name}_abs_mean"] = float(detached.abs().mean().item())
                debug[f"{name}_abs_max"] = float(detached.abs().max().item())
        self.last_debug = debug


# 兼容旧 import；新代码统一使用 SpatialRGBEventLoRALinear。
SpatialRGBEventFiLMLoRALinear = SpatialRGBEventLoRALinear
