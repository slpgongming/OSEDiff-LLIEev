import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


EVENT_LORA_VARIANTS = (
    "film_affine",
    "cbam_lowrank_residual",
    "cbam_output_gate",
    "residual_cross_attn",
    "residual_cross_attn_bax",
)

EVENT_LORA_VARIANT_ALIASES = {"film": "film_affine"}


def normalize_event_lora_variant(variant):
    """把旧 CLI/checkpoint 中的 ``film`` 统一映射到新名称。"""
    return EVENT_LORA_VARIANT_ALIASES.get(str(variant), str(variant))


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
    mod_num_heads: int = 4
    mod_window_size: int = 8
    use_dwconv_pe: bool = True
    zero_init_last: bool = True

    def __post_init__(self):
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
            _BaseEventModulator._zero_init(self.out_proj[-1])

    def forward(self, query_map, event_feat):
        return self.out_proj(self.core(query_map, event_feat))


class ResidualBAxEventCrossAttentionHead(nn.Module):
    """从 ``B(A(x))`` 与事件特征预测 LoRA 输出空间残差。"""

    def __init__(self, query_channels, event_channels, config):
        super().__init__()
        self.core = WindowRGBEventCrossAttentionCore(
            query_channels=query_channels,
            event_channels=event_channels,
            dim=config.mod_channels,
            num_heads=config.mod_num_heads,
            window_size=config.mod_window_size,
            use_dwconv_pe=config.use_dwconv_pe,
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(config.mod_channels, config.mod_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.mod_channels, query_channels, 1),
        )
        if config.zero_init_last:
            _BaseEventModulator._zero_init(self.out_proj[-1])

    def forward(self, query_map, event_feat):
        return self.out_proj(self.core(query_map, event_feat))


class SharedResidualEventHeads(nn.Module):
    """同一空间尺度的 UNet self-attention 层共享一个 residual head。"""

    def __init__(self, config, event_channels_by_size, query_channels_by_size=None):
        super().__init__()
        self.variant = config.variant
        self.heads = nn.ModuleDict()
        for size, event_channels in event_channels_by_size.items():
            if config.variant == "residual_cross_attn":
                head = ResidualEventCrossAttentionHead(
                    rank=config.rank, event_channels=event_channels, config=config
                )
            elif config.variant == "residual_cross_attn_bax":
                if query_channels_by_size is None or str(size) not in query_channels_by_size:
                    raise ValueError(
                        "residual_cross_attn_bax requires query_channels_by_size "
                        f"for size={size}."
                    )
                head = ResidualBAxEventCrossAttentionHead(
                    query_channels=query_channels_by_size[str(size)],
                    event_channels=event_channels,
                    config=config,
                )
            else:
                raise ValueError(f"Shared residual heads do not support {config.variant}.")
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
    用 Event-LoRA 替换 UNet self-attention `attn1.to_v`。

    输入为 `[B,N,C]` 图像 token。FiLM/low-rank residual 在 `lora_up` 前
    修改 rank 通道特征；output gate 在 `lora_up` 后修改 LoRA residual。
    base Linear 始终冻结，且没有事件特征时会退化为普通 LoRA。
    """

    def __init__(
        self,
        base_linear,
        config,
        event_channels_by_size,
        shared_residual_heads=None,
    ):
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
        self.last_visuals = {}
        self.last_z_delta = None

        self.lora_down = nn.Linear(base_linear.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        if config.variant in {"residual_cross_attn", "residual_cross_attn_bax"}:
            if shared_residual_heads is None:
                raise ValueError(f"{config.variant} requires shared_residual_heads.")
            # 共享 head 已注册在 UNet 顶层；这里只保留非注册引用，避免 state_dict 重复。
            object.__setattr__(self, "shared_residual_heads", shared_residual_heads)
            self.modulator = None
        elif config.variant == "film_affine":
            self.modulator = FiLMAffineModulator(
                self.rank, event_channels_by_size, config
            )
        elif config.variant == "cbam_lowrank_residual":
            self.modulator = CBAMLowRankResidualModulator(
                self.rank, event_channels_by_size, config
            )
        elif config.variant == "cbam_output_gate":
            self.modulator = CBAMOutputGateModulator(
                self.rank, base_linear.out_features, event_channels_by_size, config
            )
        else:
            raise ValueError(f"Unsupported Event-LoRA variant: {config.variant}")

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
        event_key = self._select_event_key(spatial)
        event_feat = self.event_feats[event_key]
        z_map = z.transpose(1, 2).reshape(bsz, self.rank, spatial, spatial)

        if self.variant == "residual_cross_attn":
            delta_z = self.shared_residual_heads(z_map, self.event_feats)
            z_mod = z_map + self.config.modulation_scale * delta_z
            self.last_z_delta = (z_mod - z_map).detach()
            z_tokens = z_mod.flatten(2).transpose(1, 2).contiguous()
            lora_out = self.lora_up(z_tokens) * self.scaling
            result = {"aux": {"z_event": delta_z, "delta_z": z_mod - z_map}}
        elif self.variant == "residual_cross_attn_bax":
            bax_tokens = self.lora_up(z)
            bax_map = bax_tokens.transpose(1, 2).reshape(
                bsz, self.base_linear.out_features, spatial, spatial
            )
            delta_bax = self.shared_residual_heads(bax_map, self.event_feats)
            bax_mod = bax_map + self.config.modulation_scale * delta_bax
            self.last_z_delta = (bax_mod - bax_map).detach()
            lora_out = bax_mod.flatten(2).transpose(1, 2).contiguous() * self.scaling
            result = {"aux": {"bax_event": delta_bax, "bax_delta": bax_mod - bax_map}}
        elif self.variant == "cbam_output_gate":
            u_tokens = self.lora_up(z) * self.scaling
            u_map = u_tokens.transpose(1, 2).reshape(
                bsz, self.base_linear.out_features, spatial, spatial
            )
            result = self.modulator(z_map, event_feat, event_key, u_map=u_map)
            self.last_z_delta = (result["u_mod"] - u_map).detach()
            lora_out = result["u_mod"].flatten(2).transpose(1, 2).contiguous()
        else:
            result = self.modulator(z_map, event_feat, event_key)
            self.last_z_delta = (result["z_mod"] - z_map).detach()
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
