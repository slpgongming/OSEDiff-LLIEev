import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialRGBEventFiLMLoRALinear(nn.Module):
    """
    用事件特征调制 self-attention to_v 的 LoRA Linear。

    作用：
    - base_linear 保留 Stable Diffusion 原始 to_v 权重，并保持冻结；
    - lora_down / lora_up 表示普通 LoRA 的 A/B 矩阵；
    - event_proj、z_proj 和 modulation_head 在 LoRA 低秩空间预测空间相关 gamma/beta；
    - 当 event_feats=None 或 modulation 关闭时，退化为普通 LoRA。

    输入 hidden_states:
        [B, N, C]，其中 N=H*W 是 self-attention 的图像 token 数。
    """

    def __init__(
        self,
        base_linear,
        rank=4,
        lora_alpha=None,
        modulation_scale=0.1,
        mod_channels=32,
        event_channels_by_size=None,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.requires_grad_(False)
        self.rank = int(rank)
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else rank)
        self.scaling = self.lora_alpha / max(self.rank, 1)
        self.modulation_scale = float(modulation_scale)
        self.mod_channels = int(mod_channels)
        self.event_channels_by_size = event_channels_by_size or {
            "32": 64,
            "16": 96,
            "8": 128,
            "4": 128,
        }
        self.event_feats = None
        self.modulation_enabled = True
        self.last_debug = {}

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_down = nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, out_features, bias=False)
        self.z_proj = nn.Conv2d(self.rank, self.mod_channels, kernel_size=1)
        self.event_proj = nn.ModuleDict({
            k: nn.Conv2d(v, self.mod_channels, kernel_size=1)
            for k, v in self.event_channels_by_size.items()
        })
        self.modulation_head = nn.Sequential(
            nn.Conv2d(self.mod_channels * 2, self.mod_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.mod_channels, self.rank * 2, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        nn.init.kaiming_uniform_(self.z_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.z_proj.bias)
        for proj in self.event_proj.values():
            nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))
            nn.init.zeros_(proj.bias)
        # 最后一层 zero-init：Stage2 刚加载时 dgamma/dbeta=0，输出等价普通 LoRA。
        last = self.modulation_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def set_event_feats(self, event_feats):
        self.event_feats = event_feats

    def clear_event_feats(self):
        self.event_feats = None

    def set_modulation_enabled(self, enabled):
        self.modulation_enabled = bool(enabled)

    def forward(self, hidden_states):
        base_out = self.base_linear(hidden_states)
        z = self.lora_down(hidden_states)

        if (
            not self.modulation_enabled
            or self.event_feats is None
            or hidden_states.dim() != 3
        ):
            lora_out = self.lora_up(z) * self.scaling
            self._record_debug(None, None, base_out, lora_out)
            return base_out + lora_out

        bsz, token_count, _ = hidden_states.shape
        spatial = int(math.sqrt(token_count))
        if spatial * spatial != token_count:
            raise ValueError(
                f"Event-FiLM 当前假设 self-attention token 为方形网格，收到 N={token_count}。"
            )

        z_map = z.transpose(1, 2).reshape(bsz, self.rank, spatial, spatial)
        key = self._select_event_key(spatial)
        event_feat = self.event_feats[key]
        if event_feat.shape[-2:] != (spatial, spatial):
            event_feat = F.interpolate(event_feat, size=(spatial, spatial), mode="bilinear", align_corners=False)

        event_proj = self.event_proj[key](event_feat.to(dtype=z_map.dtype))
        z_proj = self.z_proj(z_map)
        fused = torch.cat([event_proj, z_proj], dim=1)
        dgamma, dbeta = self.modulation_head(fused).chunk(2, dim=1)

        # 在低秩空间中做空间 affine 调制，tanh 限制 gamma 幅度以降低伪纹理风险。
        z_mod = (1.0 + self.modulation_scale * torch.tanh(dgamma)) * z_map
        z_mod = z_mod + self.modulation_scale * dbeta
        z_mod = z_mod.flatten(2).transpose(1, 2).contiguous()
        lora_out = self.lora_up(z_mod) * self.scaling
        self._record_debug(dgamma, dbeta, base_out, lora_out)
        return base_out + lora_out

    def _select_event_key(self, spatial):
        key = str(spatial)
        if key in self.event_feats and key in self.event_proj:
            return key
        available = [k for k in self.event_feats.keys() if k in self.event_proj]
        if not available:
            raise RuntimeError("Event-FiLM 没有可用的事件特征尺度。")
        return min(available, key=lambda k: abs(int(k) - spatial))

    def _record_debug(self, dgamma, dbeta, base_out, lora_out):
        if dgamma is None or dbeta is None:
            self.last_debug = {
                "dgamma_abs_mean": 0.0,
                "dbeta_abs_mean": 0.0,
                "dgamma_abs_max": 0.0,
                "dbeta_abs_max": 0.0,
                "base_norm": float(base_out.detach().float().norm().item()),
                "lora_norm": float(lora_out.detach().float().norm().item()),
            }
            return
        self.last_debug = {
            "dgamma_abs_mean": float(dgamma.detach().abs().mean().item()),
            "dbeta_abs_mean": float(dbeta.detach().abs().mean().item()),
            "dgamma_abs_max": float(dgamma.detach().abs().max().item()),
            "dbeta_abs_max": float(dbeta.detach().abs().max().item()),
            "base_norm": float(base_out.detach().float().norm().item()),
            "lora_norm": float(lora_out.detach().float().norm().item()),
        }
