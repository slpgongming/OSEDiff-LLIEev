"""idea6：冻结 RGB/事件 LoRA，仅学习事件残差的空间选择与层强度。"""

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.event_film_lora import EventLoRAConfig, SharedResidualEventHeads


@dataclass(frozen=True)
class MaskFusionConfig:
    mask_channels: int = 16
    max_strength: float = 2.0
    # 旧 idea4 checkpoint 只保存全局 alpha=rank，而实际层复制 PEFT alpha=8。
    # 因此必须显式记录 legacy alpha，不能误用全局配置改变 90 号残差幅度。
    legacy_event_lora_alpha: float = 8.0

    def __post_init__(self):
        if self.mask_channels < 1:
            raise ValueError("mask_channels must be positive")
        if not math.isfinite(self.max_strength) or self.max_strength <= 0:
            raise ValueError("max_strength must be finite and positive")
        if not math.isfinite(self.legacy_event_lora_alpha) or self.legacy_event_lora_alpha <= 0:
            raise ValueError("legacy_event_lora_alpha must be finite and positive")

    def to_dict(self):
        return asdict(self)


def load_exact(module, state, label):
    """关键参数必须逐项匹配；不接受随机初始化分支被静默遗漏。"""
    target = module.state_dict()
    missing, extra = sorted(set(target) - set(state)), sorted(set(state) - set(target))
    shapes = [k for k in target.keys() & state.keys() if target[k].shape != state[k].shape]
    if missing or extra or shapes:
        raise ValueError(f"{label}: missing={missing}, extra={extra}, shape_mismatch={shapes}")
    module.load_state_dict(state, strict=True)


class SpatialResidualMask(nn.Module):
    """[B,C_rgb,H,W] 与 [B,C_event,H,W] -> [B,1,H,W]，不访问 GT。"""

    def __init__(self, rgb_channels, event_channels, width):
        super().__init__()
        self.rgb_proj = nn.Conv2d(rgb_channels, width, 1)
        self.event_proj = nn.Conv2d(event_channels, width, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 1, 1)
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)  # 初始 mask=0.5；alpha=0 仍可获得梯度。

    def forward(self, rgb, event):
        event = F.interpolate(event, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        event = event.to(device=rgb.device, dtype=rgb.dtype)
        return self.fuse(torch.cat((self.rgb_proj(rgb), self.event_proj(event)), dim=1)).sigmoid()


class RGBEventMaskLinear(nn.Module):
    """保留原 PEFT RGB 层，额外注入 alpha*M*s90*B90[H(A90h,E)-H(A90h,0)]。

    90 号无条件 B90*A90*h 不进入输出；H 的参数冻结，但对输入 h 的梯度保留。
    同尺度 H 仅注册一次，避免重复保存。实际零输入先经过 EventEncoder，不能
    用全零 feature 替代 f(0)。本实验限定 256 方形输入及 residual_cross_attn。
    """

    def __init__(self, rgb_layer, down, up, head, event_channels, scaling, modulation_scale, config):
        super().__init__()
        self.rgb_layer = rgb_layer
        self.event_down = nn.Linear(down.shape[1], down.shape[0], bias=False)
        self.event_up = nn.Linear(up.shape[1], up.shape[0], bias=False)
        with torch.no_grad():
            self.event_down.weight.copy_(down)
            self.event_up.weight.copy_(up)
        object.__setattr__(self, "event_head", head)
        self.mask = SpatialResidualMask(down.shape[1], event_channels, config.mask_channels)
        self.raw_strength = nn.Parameter(torch.zeros(()))
        self.max_strength = config.max_strength
        self.event_scale = float(scaling * modulation_scale)
        self.context = None
        self.last_stats = {}
        self.last_mask = None
        self.requires_grad_(False)
        self.mask.requires_grad_(True)
        self.raw_strength.requires_grad_(True)

    @property
    def strength(self):
        # 允许有界的正/负修正；tanh(0) 保证初始精确回到 RGB 路径。
        return self.max_strength * self.raw_strength.tanh()

    def forward(self, hidden_states, *args, **kwargs):
        rgb_out = self.rgb_layer(hidden_states, *args, **kwargs)
        if self.context is None:
            raise RuntimeError("idea6 event context missing; use fusion_context for UNet forward")
        event, zero, enabled, collect_stats = self.context
        if not enabled:
            return rgb_out
        b, n, channels = hidden_states.shape
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError(f"idea6 requires square token grids, got N={n}")
        query = self.event_down(hidden_states).transpose(1, 2).reshape(b, -1, side, side)
        # 不能 detach/no_grad：较早层的 mask 需要通过当前冻结 head 获得梯度。
        delta = self.event_head(query, event) - self.event_head(query, zero)
        delta = self.event_up(delta.flatten(2).transpose(1, 2)) * self.event_scale
        rgb_map = hidden_states.transpose(1, 2).reshape(b, channels, side, side)
        mask = self.mask(rgb_map, event - zero)
        correction = self.strength.to(delta.dtype) * mask.flatten(2).transpose(1, 2) * delta
        if collect_stats:
            self.last_mask = mask.detach()
            # 缓存 detached 标量张量，日志阶段统一同步，避免每层 .item() 同步 GPU。
            self.last_stats = {
                "mask_mean": mask.detach().float().mean(),
                "mask_std": mask.detach().float().std(unbiased=False),
                "strength": self.strength.detach().float(),
                "event_delta_abs": delta.detach().float().abs().mean(),
                "injection_abs": correction.detach().float().abs().mean(),
            }
        return rgb_out + correction.to(rgb_out.dtype)


def iter_mask_layers(unet):
    for name, module in unet.named_modules():
        if isinstance(module, RGBEventMaskLinear):
            yield name, module


def install_mask_fusion(unet, event_checkpoint, event_channels_by_size, config):
    """只迁移 90 号的 A/B 与共享 H；不加载其普通 LoRA、conv_in 或 VAE。"""
    saved = event_checkpoint.get("event_lora_config")
    if not saved:
        raise ValueError("90 checkpoint must contain event_lora_config")
    event_config = EventLoRAConfig(**saved)
    if event_config.variant != "residual_cross_attn":
        raise ValueError("idea6 pilot supports 90's residual_cross_attn checkpoint only")
    if event_config.modulation_scale <= 0:
        raise ValueError("event modulation_scale must be positive")
    source = event_checkpoint["state_dict_unet"]
    shared_prefix = "_event_lora_shared_heads_residual_cross_attn."
    heads = SharedResidualEventHeads(event_config, event_channels_by_size)
    load_exact(heads, {k[len(shared_prefix):]: v for k, v in source.items()
                       if k.startswith(shared_prefix)}, "90 shared event heads")
    heads.requires_grad_(False)
    targets = [(n, m) for n, m in unet.named_modules() if n.endswith("attn1.to_v")]
    if not targets or any(isinstance(m, RGBEventMaskLinear) for _, m in targets):
        raise ValueError("Expected unmodified RGB UNet attn1.to_v layers")
    expected = {f"{n}.{suffix}.weight" for n, _ in targets for suffix in ("lora_down", "lora_up")}
    actual = {k for k in source if k.endswith((".lora_down.weight", ".lora_up.weight"))}
    if expected != actual:
        raise ValueError(f"90 to_v mapping mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    unet.add_module("idea6_event_heads", heads)
    block_channels = list(unet.config.block_out_channels)
    sizes = sorted(event_channels_by_size, key=int, reverse=True)
    layer_configs = event_checkpoint.get("event_lora_layer_configs", {})
    for name, rgb_layer in targets:
        # SD UNet down i / up reversed-i / mid-last 的事件尺度，避免同通道层映射歧义。
        if name.startswith("down_blocks."):
            level = int(name.split(".")[1])
        elif name.startswith("up_blocks."):
            level = len(block_channels) - 1 - int(name.split(".")[1])
        elif name.startswith("mid_block."):
            level = len(block_channels) - 1
        else:
            raise ValueError(f"Unknown attention block: {name}")
        key = sizes[level]
        down, up = source[f"{name}.lora_down.weight"], source[f"{name}.lora_up.weight"]
        base = getattr(rgb_layer, "base_layer", rgb_layer)
        if tuple(down.shape) != (event_config.rank, base.in_features) or tuple(up.shape) != (base.out_features, event_config.rank):
            raise ValueError(f"90 rank/channel mismatch at {name}")
        alpha = float(layer_configs.get(name, {}).get("alpha", config.legacy_event_lora_alpha))
        wrapped = RGBEventMaskLinear(rgb_layer, down, up, heads.heads[key],
                                     event_channels_by_size[key], alpha / event_config.rank,
                                     event_config.modulation_scale, config)
        wrapped.event_key = key
        parent = unet.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], wrapped)
    return [n for n, _ in targets]


@contextmanager
def fusion_context(unet, event_features, zero_features, enabled=True, collect_stats=False):
    """每次前向显式设置/清理条件；checkpoint 重算也必须重新进入此作用域。"""
    layers = list(iter_mask_layers(unet))
    if any(layer.context is not None for _, layer in layers):
        raise RuntimeError("Nested/stale idea6 event context")
    try:
        for _, layer in layers:
            key = layer.event_key
            layer.context = (event_features[key], zero_features[key], enabled, collect_stats)
        yield
    finally:
        for _, layer in layers:
            layer.context = None


def mask_state_dict(unet):
    return {k: v.detach().cpu() for k, v in unet.state_dict().items()
            if ".mask." in k or k.endswith(".raw_strength")}


def load_mask_state(unet, state):
    expected = mask_state_dict(unet)
    if set(expected) != set(state) or any(expected[k].shape != state[k].shape for k in expected.keys() & state.keys()):
        raise ValueError("idea6 mask checkpoint keys/shapes do not match")
    unet.load_state_dict(state, strict=False)
