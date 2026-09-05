"""Event Self-Swap Guidance (ESG) 的纯推理特征扰动工具。"""

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F


_SWAP_MODES = {"spatial", "channel", "both"}
_PAIR_POLICIES = {"dissimilar", "random", "similar"}
_EXECUTION_MODES = {"batched", "sequential"}


@dataclass(frozen=True)
class EventSelfSwapConfig:
    """控制 ESG 负分支的 event-token swap 与 guidance 强度。"""

    guidance_scale: float = 0.0
    swap_ratio: float = 0.1
    swap_mode: str = "spatial"
    pair_policy: str = "dissimilar"
    feature_scales: Tuple[str, ...] = ("32", "16", "8")
    seed: int = 42
    execution_mode: str = "batched"

    def __post_init__(self):
        scales = tuple(str(scale) for scale in self.feature_scales)
        object.__setattr__(self, "feature_scales", scales)
        if self.guidance_scale < 0:
            raise ValueError("ESG guidance_scale must be non-negative.")
        if not 0.0 <= self.swap_ratio <= 1.0:
            raise ValueError("ESG swap_ratio must be in [0, 1].")
        if self.swap_mode not in _SWAP_MODES:
            raise ValueError(
                f"Unsupported ESG swap_mode={self.swap_mode}; expected {_SWAP_MODES}."
            )
        if self.pair_policy not in _PAIR_POLICIES:
            raise ValueError(
                f"Unsupported ESG pair_policy={self.pair_policy}; expected {_PAIR_POLICIES}."
            )
        if self.execution_mode not in _EXECUTION_MODES:
            raise ValueError(
                "Unsupported ESG execution_mode="
                f"{self.execution_mode}; expected {_EXECUTION_MODES}."
            )
        if not scales:
            raise ValueError("ESG feature_scales must contain at least one scale.")
        if len(scales) != len(set(scales)):
            raise ValueError(f"ESG feature_scales contains duplicates: {scales}")


def _candidate_pairs(token_count: int, device: torch.device) -> torch.Tensor:
    """返回不含对角线的上三角 token 对 `[2, num_pairs]`。"""
    return torch.triu_indices(token_count, token_count, offset=1, device=device)


def _select_disjoint_pairs(
    tokens: torch.Tensor,
    pair_count: int,
    policy: str,
    seed: int,
) -> Tuple[torch.Tensor, float]:
    """按 cosine similarity 贪心选择互不重叠的 token 对。"""
    token_count = tokens.shape[0]
    if pair_count <= 0 or token_count < 2:
        return torch.empty((0, 2), dtype=torch.long, device=tokens.device), 0.0

    candidates = _candidate_pairs(token_count, tokens.device)
    candidate_left = candidates[0].detach().cpu().tolist()
    candidate_right = candidates[1].detach().cpu().tolist()
    if policy == "random":
        # 在 CPU 上生成顺序，使 CUDA/CPU 与 batched/sequential 路径共享确定性索引。
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        order = torch.randperm(candidates.shape[1], generator=generator).tolist()
        normalized = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    else:
        normalized = F.normalize(tokens.float(), dim=-1, eps=1e-6)
        similarities = (normalized @ normalized.transpose(0, 1))[
            candidates[0], candidates[1]
        ]
        order = torch.argsort(
            similarities, descending=(policy == "similar")
        ).tolist()

    used = set()
    selected = []
    selected_similarities = []
    for candidate_index in order:
        # pair 坐标一次性搬到 CPU，避免 Python greedy 循环逐元素触发 CUDA 同步。
        left = candidate_left[candidate_index]
        right = candidate_right[candidate_index]
        if left in used or right in used:
            continue
        selected.append((left, right))
        used.add(left)
        used.add(right)
        selected_similarities.append(
            float((normalized[left] * normalized[right]).sum().item())
        )
        if len(selected) >= pair_count:
            break

    if not selected:
        return torch.empty((0, 2), dtype=torch.long, device=tokens.device), 0.0
    return (
        torch.tensor(selected, dtype=torch.long, device=tokens.device),
        sum(selected_similarities) / len(selected_similarities),
    )


def _swap_token_axis(
    tokens: torch.Tensor,
    swap_ratio: float,
    policy: str,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """交换 `[B,T,D]` 的 T 维 token，并返回实际比例与 pair similarity。"""
    batch_size, token_count, _ = tokens.shape
    requested_pairs = min(int(swap_ratio * token_count / 2), token_count // 2)
    output = tokens.clone()
    total_pairs = 0
    similarities = []
    for batch_index in range(batch_size):
        pairs, mean_similarity = _select_disjoint_pairs(
            tokens[batch_index],
            requested_pairs,
            policy,
            seed + batch_index * 1000003,
        )
        if pairs.numel() == 0:
            continue
        left, right = pairs[:, 0], pairs[:, 1]
        output[batch_index, left] = tokens[batch_index, right]
        output[batch_index, right] = tokens[batch_index, left]
        total_pairs += pairs.shape[0]
        similarities.append(mean_similarity)
    denominator = max(batch_size * token_count, 1)
    return output, {
        "pair_count": float(total_pairs),
        "swap_fraction": float(2 * total_pairs / denominator),
        "pair_similarity": float(sum(similarities) / max(len(similarities), 1)),
    }


def _spatial_swap(
    feature: torch.Tensor, config: EventSelfSwapConfig, seed: int
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size, channels, height, width = feature.shape
    tokens = feature.flatten(2).transpose(1, 2).contiguous()
    swapped, stats = _swap_token_axis(
        tokens, config.swap_ratio, config.pair_policy, seed
    )
    return (
        swapped.transpose(1, 2).reshape(batch_size, channels, height, width),
        stats,
    )


def _channel_swap(
    feature: torch.Tensor, config: EventSelfSwapConfig, seed: int
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size, channels, height, width = feature.shape
    tokens = feature.flatten(2)
    swapped, stats = _swap_token_axis(
        tokens, config.swap_ratio, config.pair_policy, seed
    )
    return swapped.reshape(batch_size, channels, height, width), stats


def swap_event_feature_tokens(
    event_features: Mapping[str, torch.Tensor],
    config: EventSelfSwapConfig,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    只对指定尺度执行 event feature token swap。

    未选中的 feature 保持原 tensor 引用；选中尺度先 clone，绝不原地修改正分支。
    """
    missing = [scale for scale in config.feature_scales if scale not in event_features]
    if missing:
        raise KeyError(
            f"ESG feature scales {missing} are unavailable; "
            f"available={sorted(event_features)}."
        )
    swapped_features = dict(event_features)
    stats: Dict[str, float] = {}
    for scale_index, scale in enumerate(config.feature_scales):
        feature = event_features[scale]
        if feature.ndim != 4:
            raise ValueError(
                f"ESG event feature {scale} must be [B,C,H,W], got {tuple(feature.shape)}."
            )
        swapped = feature.clone()
        scale_seed = int(config.seed) + scale_index * 100003
        if config.swap_mode in {"spatial", "both"}:
            swapped, mode_stats = _spatial_swap(swapped, config, scale_seed)
            for name, value in mode_stats.items():
                stats[f"scale_{scale}/spatial_{name}"] = value
        if config.swap_mode in {"channel", "both"}:
            swapped, mode_stats = _channel_swap(
                swapped, config, scale_seed + 50000017
            )
            for name, value in mode_stats.items():
                stats[f"scale_{scale}/channel_{name}"] = value
        swapped_features[scale] = swapped
    return swapped_features, stats
