import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowRGBEventCrossAttentionHead(nn.Module):
    """
    Window RGB-Event Cross-Attention 调制头。

    作用：
    - 将 LoRA 低秩特征 z_map 作为 Query；
    - 将同尺度 event feature 作为 Key/Value；
    - 在局部 window 内检索事件结构信息；
    - 输出低秩空间的 dgamma/dbeta，用于 Spatial Event-FiLM。

    输入:
        z_map:      [B, rank, H, W]
        event_feat: [B, C_event, H, W]
    输出:
        dgamma/dbeta: [B, rank, H, W]
    """

    def __init__(
        self,
        rank,
        event_channels,
        dim=64,
        num_heads=4,
        window_size=8,
        use_dwconv_pe=True,
        use_local_event_concat=True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"mod_channels/dim={dim} must be divisible by num_heads={num_heads}.")

        self.rank = int(rank)
        self.event_channels = int(event_channels)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.use_dwconv_pe = bool(use_dwconv_pe)
        self.use_local_event_concat = bool(use_local_event_concat)

        # z_proj 将 LoRA 低秩空间投影到 attention dim；event_proj 对齐事件特征通道。
        self.z_proj = nn.Conv2d(self.rank, self.dim, kernel_size=1)
        self.event_proj = nn.Conv2d(self.event_channels, self.dim, kernel_size=1)

        if self.use_dwconv_pe:
            # Depthwise Conv PE 只注入局部位置信息，不改变通道数。
            self.z_pos = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim)
            self.event_pos = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim)

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.attn_out = nn.Linear(self.dim, self.dim)

        fuse_in = self.dim * 3 if self.use_local_event_concat else self.dim * 2
        self.fuse_head = nn.Sequential(
            nn.Conv2d(fuse_in, self.dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.dim, self.rank * 2, kernel_size=1),
        )
        self.zero_init_last()

    def zero_init_last(self):
        # zero-init 保证 Stage2 初始化时 dgamma=dbeta=0，Event-FiLM 等价普通 LoRA。
        last = self.fuse_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @staticmethod
    def window_partition(x, window_size):
        """
        将特征图切成局部 window。

        输入:
            x: [B, C, H, W]，H/W 已经 padding 到能被 window_size 整除。
        输出:
            windows: [B*nW, M*M, C]
        """
        bsz, channels, height, width = x.shape
        m = window_size
        x = x.contiguous().view(bsz, channels, height // m, m, width // m, m)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, m * m, channels)

    @staticmethod
    def window_reverse(windows, window_size, bsz, height, width):
        """
        将 window tokens 还原回空间特征图。

        输入:
            windows: [B*nW, M*M, C]
        输出:
            x: [B, C, H, W]
        """
        m = window_size
        channels = windows.shape[-1]
        x = windows.view(bsz, height // m, width // m, m, m, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(bsz, channels, height, width)

    def forward(self, z_map, event_feat):
        bsz, _, height, width = z_map.shape
        if event_feat.shape[-2:] != (height, width):
            # 事件编码器输出是固定多尺度特征；这里按 LoRA token 的真实空间尺度对齐。
            event_feat = F.interpolate(
                event_feat,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        event_feat = event_feat.to(device=z_map.device, dtype=z_map.dtype)
        m = min(self.window_size, height, width)
        pad_h = (m - height % m) % m
        pad_w = (m - width % m) % m
        if pad_h > 0 or pad_w > 0:
            z_pad = F.pad(z_map, (0, pad_w, 0, pad_h))
            e_pad = F.pad(event_feat, (0, pad_w, 0, pad_h))
        else:
            z_pad = z_map
            e_pad = event_feat

        padded_h, padded_w = z_pad.shape[-2:]
        fz = self.z_proj(z_pad)
        fe = self.event_proj(e_pad)
        if self.use_dwconv_pe:
            fz = fz + self.z_pos(fz)
            fe = fe + self.event_pos(fe)

        # window 内每个 LoRA token 都只和同一局部区域的 event tokens 做 cross-attention。
        q_tokens = self.window_partition(fz, m)
        e_tokens = self.window_partition(fe, m)
        q = self.q_proj(q_tokens)
        k = self.k_proj(e_tokens)
        v = self.v_proj(e_tokens)

        b_windows, token_count, dim = q.shape
        head_dim = dim // self.num_heads
        q = q.view(b_windows, token_count, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(b_windows, token_count, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(b_windows, token_count, self.num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1).to(dtype=v.dtype)
        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).contiguous().view(b_windows, token_count, dim)
        attended = self.attn_out(attended)
        fa = self.window_reverse(attended, m, bsz, padded_h, padded_w)

        if pad_h > 0 or pad_w > 0:
            fz = fz[:, :, :height, :width]
            fe = fe[:, :, :height, :width]
            fa = fa[:, :, :height, :width]

        if self.use_local_event_concat:
            fuse = torch.cat([fz, fa, fe], dim=1)
        else:
            fuse = torch.cat([fz, fa], dim=1)
        dgamma, dbeta = self.fuse_head(fuse).chunk(2, dim=1)
        return dgamma, dbeta


class SharedWindowModulationHeads(nn.Module):
    """
    同尺度共享的 Window Cross-Attention 调制头容器。

    作用：
    - 不为每个 attn1.to_v 单独创建 head；
    - 相同空间分辨率共享同一个 modulation head；
    - 降低参数量，并让 Stage2 event branch 更稳定。
    """

    def __init__(
        self,
        rank,
        event_channels_by_size=None,
        dim=64,
        num_heads=4,
        window_size=8,
    ):
        super().__init__()
        self.event_channels_by_size = event_channels_by_size or {
            "32": 64,
            "16": 96,
            "8": 128,
            "4": 128,
        }
        self.heads = nn.ModuleDict({
            str(size): WindowRGBEventCrossAttentionHead(
                rank=rank,
                event_channels=channels,
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                use_dwconv_pe=True,
                use_local_event_concat=True,
            )
            for size, channels in self.event_channels_by_size.items()
        })

    def _select_key(self, spatial, event_feats):
        key = str(spatial)
        if key in self.heads and key in event_feats:
            return key
        candidates = [k for k in self.heads.keys() if k in event_feats]
        if not candidates:
            raise RuntimeError("Event-FiLM 没有可用的同尺度事件特征。")
        return min(candidates, key=lambda k: abs(int(k) - spatial))

    def forward(self, z_map, event_feats):
        spatial = z_map.shape[-2]
        key = self._select_key(spatial, event_feats)
        return self.heads[key](z_map, event_feats[key])


class SpatialRGBEventFiLMLoRALinear(nn.Module):
    """
    用事件特征调制 self-attention to_v 的 LoRA Linear。

    作用：
    - base_linear 保留 Stable Diffusion 原始 to_v 权重，并保持冻结；
    - lora_down / lora_up 表示普通 LoRA 的 A/B 矩阵；
    - shared_modulation_heads 使用 window cross-attention 从 event 中检索结构先验；
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
        mod_channels=64,
        event_channels_by_size=None,
        mod_num_heads=4,
        mod_window_size=8,
        shared_modulation_heads=None,
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
        self.last_visuals = {}

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_down = nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, out_features, bias=False)

        if shared_modulation_heads is None:
            self.shared_modulation_heads = SharedWindowModulationHeads(
                rank=self.rank,
                event_channels_by_size=self.event_channels_by_size,
                dim=self.mod_channels,
                num_heads=mod_num_heads,
                window_size=mod_window_size,
            )
        else:
            # shared_modulation_heads 已作为 UNet 顶层子模块注册；这里保存引用即可，
            # 避免同一个 head 被每个 LoRA 层重复注册和重复保存。
            object.__setattr__(self, "shared_modulation_heads", shared_modulation_heads)

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

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

        # LoRA A 的低秩 token 恢复成空间图，之后所有事件调制都发生在低秩空间。
        z_map = z.transpose(1, 2).reshape(bsz, self.rank, spatial, spatial)
        dgamma, dbeta = self.shared_modulation_heads(z_map, self.event_feats)

        # Spatial Event-FiLM：gamma 用 tanh 限幅，beta 直接平移，降低事件噪声导致伪纹理的风险。
        scale = self.modulation_scale
        z_mod = (1.0 + scale * torch.tanh(dgamma)) * z_map
        z_mod = z_mod + scale * dbeta
        z_mod = z_mod.flatten(2).transpose(1, 2).contiguous()
        lora_out = self.lora_up(z_mod) * self.scaling
        self._record_debug(dgamma, dbeta, base_out, lora_out)
        return base_out + lora_out

    def _record_debug(self, dgamma, dbeta, base_out, lora_out):
        if dgamma is None or dbeta is None:
            zero = base_out.detach().float().new_tensor(0.0)
            self.last_debug = {
                "dgamma_abs_mean": zero,
                "dbeta_abs_mean": zero,
                "dgamma_abs_max": zero,
                "dbeta_abs_max": zero,
                "base_norm": base_out.detach().float().norm(),
                "lora_norm": lora_out.detach().float().norm(),
            }
            self.last_visuals = {}
            return

        dgamma_abs = dgamma.detach().abs()
        dbeta_abs = dbeta.detach().abs()
        self.last_debug = {
            "dgamma_abs_mean": dgamma_abs.mean(),
            "dbeta_abs_mean": dbeta_abs.mean(),
            "dgamma_abs_max": dgamma_abs.max(),
            "dbeta_abs_max": dbeta_abs.max(),
            "base_norm": base_out.detach().float().norm(),
            "lora_norm": lora_out.detach().float().norm(),
        }
        # 只缓存通道均值热力图，供 W&B 可视化；不保存计算图，也不落本地文件。
        self.last_visuals = {
            "dgamma_abs": dgamma_abs.mean(dim=1, keepdim=True),
            "dbeta_abs": dbeta_abs.mean(dim=1, keepdim=True),
        }
