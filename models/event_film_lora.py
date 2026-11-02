import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


EVENT_LORA_VARIANTS = ("residual_cross_attn",)

EVENT_LORA_VARIANT_ALIASES = {"film": "film_affine"}
EVENT_LORA_TARGETS = ("q", "k", "v", "out")


def parse_event_lora_targets(value, allow_auto=False):
    """Normalize the selected self-attention projections in a stable order."""
    if not isinstance(value, str) and not value:
        return ()
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
    else:
        parts = [str(part).strip().lower() for part in value]
    if len(parts) == 1 and parts[0] == "auto" and allow_auto:
        return "auto"
    if len(parts) == 1 and parts[0] == "all":
        return EVENT_LORA_TARGETS
    if len(parts) == 1 and parts[0] == "none":
        return ()
    if not parts or any(part not in EVENT_LORA_TARGETS for part in parts) or len(parts) != len(set(parts)):
        raise ValueError(f"Invalid Event-LoRA targets {value!r}; use none, all, auto, or a unique subset of q,k,v,out.")
    return tuple(target for target in EVENT_LORA_TARGETS if target in parts)


def normalize_event_lora_variant(variant):
    """把旧 CLI/checkpoint 中的 ``film`` 统一映射到新名称。"""
    return EVENT_LORA_VARIANT_ALIASES.get(str(variant), str(variant))


@dataclass(frozen=True)
class EventLoRAConfig:
    """Event-LoRA 的结构配置；训练、推理和 regularizer 必须使用同一份语义。"""

    variant: str = "residual_cross_attn"
    targets: tuple = ("v",)
    rank: int = 4
    alpha: float = 4.0
    mod_channels: int = 64
    modulation_scale: float = 0.1
    # 仅保留旧 residual-cross-attn 配置的序列化字段，不参与计算。
    cbam_reduction: int = 8
    cbam_spatial_kernel: int = 7
    output_gate_channels: str = "spatial"
    mod_num_heads: int = 4
    mod_window_size: int = 8
    use_dwconv_pe: bool = True
    zero_init_last: bool = True

    def __post_init__(self):
        object.__setattr__(self, "targets", parse_event_lora_targets(self.targets))
        normalized_variant = normalize_event_lora_variant(self.variant)
        object.__setattr__(self, "variant", normalized_variant)
        if normalized_variant not in EVENT_LORA_VARIANTS:
            raise ValueError(f"Unsupported Event-LoRA variant: {normalized_variant}")
        if self.rank <= 0 or self.mod_channels <= 0:
            raise ValueError("rank and mod_channels must be positive.")
        if self.cbam_reduction <= 0:
            raise ValueError("cbam_reduction must be positive.")
        if self.cbam_spatial_kernel not in {3, 7}:
            raise ValueError("cbam_spatial_kernel must be 3 or 7.")
        if self.output_gate_channels not in {"spatial", "channel_spatial"}:
            raise ValueError("output_gate_channels must be spatial or channel_spatial.")
        if self.mod_num_heads <= 0 or self.mod_channels % self.mod_num_heads != 0:
            raise ValueError("mod_channels must be divisible by mod_num_heads.")
        if self.mod_window_size <= 0:
            raise ValueError("mod_window_size must be positive.")

    def to_dict(self):
        return asdict(self)


class WindowRGBEventCrossAttentionCore(nn.Module):
    """在局部窗口内让 RGB/LoRA query 检索同尺度事件 token。"""

    def __init__(
        self,
        query_channels,
        event_channels,
        dim=64,
        num_heads=4,
        window_size=8,
        use_dwconv_pe=True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.query_channels = int(query_channels)
        self.event_channels = int(event_channels)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.use_dwconv_pe = bool(use_dwconv_pe)

        self.z_proj = nn.Conv2d(self.query_channels, self.dim, kernel_size=1)
        self.event_proj = nn.Conv2d(self.event_channels, self.dim, kernel_size=1)
        if self.use_dwconv_pe:
            self.z_pos = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim)
            self.event_pos = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.attn_out = nn.Linear(self.dim, self.dim)

    @staticmethod
    def _partition(x, window_size):
        bsz, channels, height, width = x.shape
        x = x.reshape(
            bsz,
            channels,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.reshape(-1, window_size * window_size, channels)

    @staticmethod
    def _reverse(windows, window_size, bsz, height, width):
        channels = windows.shape[-1]
        x = windows.reshape(
            bsz,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(bsz, channels, height, width)

    def forward(self, query_map, event_feat):
        if query_map.shape[1] != self.query_channels:
            raise ValueError(
                "Cross-attention query channel mismatch: "
                f"expected={self.query_channels}, got={query_map.shape[1]}."
            )
        bsz, _, height, width = query_map.shape
        if event_feat.shape[-2:] != (height, width):
            event_feat = F.interpolate(
                event_feat, size=(height, width), mode="bilinear", align_corners=False
            )
        event_feat = event_feat.to(device=query_map.device, dtype=query_map.dtype)

        window_size = min(self.window_size, height, width)
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        query_pad = F.pad(query_map, (0, pad_w, 0, pad_h))
        event_pad = F.pad(event_feat, (0, pad_w, 0, pad_h))
        padded_h, padded_w = query_pad.shape[-2:]

        query_feature = self.z_proj(query_pad)
        event_feature = self.event_proj(event_pad)
        if self.use_dwconv_pe:
            query_feature = query_feature + self.z_pos(query_feature)
            event_feature = event_feature + self.event_pos(event_feature)

        query_tokens = self._partition(query_feature, window_size)
        event_tokens = self._partition(event_feature, window_size)
        query = self.q_proj(query_tokens)
        key = self.k_proj(event_tokens)
        value = self.v_proj(event_tokens)

        window_count, token_count, dim = query.shape
        head_dim = dim // self.num_heads
        query = query.reshape(window_count, token_count, self.num_heads, head_dim).transpose(1, 2)
        key = key.reshape(window_count, token_count, self.num_heads, head_dim).transpose(1, 2)
        value = value.reshape(window_count, token_count, self.num_heads, head_dim).transpose(1, 2)
        attention = torch.matmul(query.float(), key.float().transpose(-2, -1)) * (head_dim ** -0.5)
        attention = attention.softmax(dim=-1).to(dtype=value.dtype)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(window_count, token_count, dim)
        attended = self.attn_out(attended)
        attended = self._reverse(attended, window_size, bsz, padded_h, padded_w)
        return attended[:, :, :height, :width]


class ResidualEventCrossAttentionHead(nn.Module):
    """从 ``A(x)`` 与事件特征预测低秩空间残差 ``delta_z``。"""

    def __init__(self, rank, event_channels, config):
        super().__init__()
        self.core = WindowRGBEventCrossAttentionCore(
            query_channels=rank,
            event_channels=event_channels,
            dim=config.mod_channels,
            num_heads=config.mod_num_heads,
            window_size=config.mod_window_size,
            use_dwconv_pe=config.use_dwconv_pe,
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(config.mod_channels, config.mod_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.mod_channels, rank, 1),
        )
        if config.zero_init_last:
            nn.init.zeros_(self.out_proj[-1].weight)
            nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, query_map, event_feat):
        return self.out_proj(self.core(query_map, event_feat))


class SharedResidualEventHeads(nn.Module):
    """同一空间尺度的 UNet self-attention 层共享一个 residual head。"""

    def __init__(self, config, event_channels_by_size, query_channels_by_size=None):
        super().__init__()
        self.variant = config.variant
        self.heads = nn.ModuleDict()
        for size, event_channels in event_channels_by_size.items():
            head = ResidualEventCrossAttentionHead(
                rank=config.rank, event_channels=event_channels, config=config
            )
            self.heads[str(size)] = head

    def forward(self, query_map, event_feats):
        spatial = query_map.shape[-2]
        key = str(spatial)
        if key not in self.heads or key not in event_feats:
            candidates = [item for item in self.heads if item in event_feats]
            if not candidates:
                raise RuntimeError("Event-LoRA 没有可用的同尺度事件特征。")
            key = min(candidates, key=lambda item: abs(int(item) - spatial))
        return self.heads[key](query_map, event_feats[key])


class SpatialRGBEventLoRALinear(nn.Module):
    """
    用 Event-LoRA 替换选中的 UNet self-attention Q/K/V/out 投影。

    输入为 `[B,N,C]` 图像 token。cross-attention 在 `lora_up` 前
    对 rank 通道的 A(x) 添加事件残差。
    base Linear 始终冻结，且没有事件特征时会退化为普通 LoRA。
    """

    def __init__(
        self,
        base_linear,
        config,
        event_channels_by_size,
        shared_residual_heads=None,
        projection="v",
    ):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.requires_grad_(False)
        self.config = config
        self.projection = projection
        self.rank = config.rank
        self.lora_alpha = config.alpha
        self.scaling = self.lora_alpha / max(self.rank, 1)
        self.event_channels_by_size = event_channels_by_size
        self.event_feats = None
        self.modulation_enabled = True
        self.last_debug = {}
        self.last_aux = {}
        self.last_visuals = {}
        self.last_z_delta = None

        self.lora_down = nn.Linear(base_linear.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        if shared_residual_heads is None:
            raise ValueError(f"{config.variant} requires shared_residual_heads.")
        # 共享 head 已注册在 UNet 顶层；这里只保留非注册引用，避免 state_dict 重复。
        object.__setattr__(self, "shared_residual_heads", shared_residual_heads)
        self.modulator = None

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
            self.last_z_delta = None
            self._record_debug({}, base_out, lora_out)
            return base_out + lora_out

        bsz, token_count, _ = hidden_states.shape
        spatial = int(math.sqrt(token_count))
        if spatial * spatial != token_count:
            raise ValueError(
                f"Event-LoRA 当前假设 self-attention token 为方形网格，收到 N={token_count}。"
            )
        z_map = z.transpose(1, 2).reshape(bsz, self.rank, spatial, spatial)

        delta_z = self.shared_residual_heads(z_map, self.event_feats)
        z_mod = z_map + self.config.modulation_scale * delta_z
        self.last_z_delta = (z_mod - z_map).detach()
        z_tokens = z_mod.flatten(2).transpose(1, 2).contiguous()
        lora_out = self.lora_up(z_tokens) * self.scaling
        result = {"aux": {"z_event": delta_z, "delta_z": z_mod - z_map}}

        self._record_debug(result["aux"], base_out, lora_out)
        return base_out + lora_out


    def _record_debug(self, aux, base_out, lora_out):
        self.last_aux = aux
        self.last_visuals = {}
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
                self.last_visuals[f"{name}_abs"] = detached.abs().mean(
                    dim=1, keepdim=True
                )
        self.last_debug = debug


# 兼容旧 import；新代码统一使用 SpatialRGBEventLoRALinear。
SpatialRGBEventFiLMLoRALinear = SpatialRGBEventLoRALinear
