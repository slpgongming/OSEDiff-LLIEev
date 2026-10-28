"""结构头的区域监督与验证统计；不读取预测置信度，不改动共用结构算子。"""

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class StructureSupervisionConfig:
    version: int = 1
    mode: str = "smooth_l1"
    # 在固定的 [0,1] 软结构域分区；这是可配置初值，正式 H0/H1 应先按训练 GT 统计锁定。
    thresholds: tuple = (0.05, 0.2)
    region_weights: tuple = (1.0, 1.0, 1.0)  # 平坦、弱边缘、强边缘。
    beta: float = 0.1

    def __post_init__(self):
        object.__setattr__(self, "thresholds", tuple(self.thresholds))
        object.__setattr__(self, "region_weights", tuple(self.region_weights))
        if self.version != 1 or self.mode not in {"smooth_l1", "region_balanced"}:
            raise ValueError("Invalid structure supervision version/mode.")
        if len(self.thresholds) != 2 or any(not math.isfinite(v) for v in self.thresholds):
            raise ValueError("Two finite region thresholds are required.")
        if not 0 < self.thresholds[0] < self.thresholds[1] < 1:
            raise ValueError("Region thresholds must satisfy 0 < weak < strong < 1.")
        if len(self.region_weights) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.region_weights):
            raise ValueError("Flat/weak/strong region weights must all be finite and positive.")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("Smooth L1 beta must be finite and positive.")


def _regional_means(values, target, config):
    """[B,S,H,W] → 区域均值与像素数 [B,S,3]；每张图每尺度独立归一化。"""
    weak, strong = config.thresholds
    label = target.detach()
    masks = torch.stack((label < weak, (label >= weak) & (label < strong), label >= strong), dim=2)
    counts = masks.sum((-2, -1))
    means = (values.unsqueeze(2) * masks).sum((-2, -1)) / counts.clamp_min(1)
    return means, counts


def _balanced_mean(means, counts, weights):
    """空区域不参与分母；保留其他区域的相对权重，防止空强边缘区域稀释损失。"""
    weights = means.new_tensor(weights) * (counts > 0)
    return (means * weights).sum(-1) / weights.sum(-1).clamp_min(1e-12)


def structure_supervision_loss(prediction, target, config=None):
    config = config or StructureSupervisionConfig()
    if config.mode == "smooth_l1":
        # 原 H0 默认公式逐字保留；新增统计和 best 保存不能改变原训练梯度。
        return F.smooth_l1_loss(prediction, target, beta=config.beta)
    residual = F.smooth_l1_loss(prediction, target, beta=config.beta, reduction="none")
    means, counts = _regional_means(residual, target, config)
    return _balanced_mean(means, counts, config.region_weights).mean()


class StructureValidationAccumulator:
    """可合并的逐样本/尺度统计；不在不等长 DDP 验证循环里执行 collective。

    每个区域指标平均的是存在该区域的样本/尺度均值，不把区域像素数当作样本权重。
    balanced_mae 始终采用区域等权，与训练选择 smooth_l1 或 region_balanced 无关。
    """
    metric_names = ("balanced_mae", "flat_mae", "weak_mae", "strong_mae", "weak_bias", "strong_bias",
                    "flat_response", "edge_precision", "edge_recall", "edge_f1", "spatial_correlation")

    def __init__(self, scales, config=None, scenes=(), device="cpu", border=4, tolerance=2):
        if scales < 1 or any(not isinstance(v, int) or v < 0 for v in (border, tolerance)):
            raise ValueError("Invalid validation scales/border/tolerance.")
        self.scales = scales
        self.config = config or StructureSupervisionConfig()
        self.border, self.tolerance = border, tolerance
        self.scenes = tuple(sorted(scenes))
        if len(set(self.scenes)) != len(self.scenes):
            raise ValueError("Validation scene names must be unique.")
        self.names = list(self.metric_names) + [f"scale_{i}_{name}" for i in range(scales) for name in self.metric_names]
        # 第 0 个槽存样本数，其余存各指标的 sum/count；rank 无样本时自然保持全零。
        self.totals = torch.zeros(1 + len(self.scenes), 1 + len(self.names), 2, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, prediction, target, scene_names=None):
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != self.scales:
            raise ValueError("Validation expects matching [B,S,H,W] structure maps.")
        if self.border and min(prediction.shape[-2:]) <= 2 * self.border:
            raise ValueError("Validation border removes the entire structure map.")
        prediction, target = prediction.detach().float(), target.detach().float()
        if self.border:
            b = self.border
            prediction, target = prediction[..., b:-b, b:-b], target[..., b:-b, b:-b]
        error = prediction - target
        mae, counts = _regional_means(error.abs(), target, self.config)
        bias, _ = _regional_means(error, target, self.config)
        response, _ = _regional_means(prediction, target, self.config)
        present = counts > 0
        always = torch.ones_like(present[..., 0], dtype=torch.bool)
        stats = {"balanced_mae": (_balanced_mean(mae, counts, (1., 1., 1.)), always),
                 "flat_mae": (mae[..., 0], present[..., 0]),
                 "weak_mae": (mae[..., 1], present[..., 1]),
                 "strong_mae": (mae[..., 2], present[..., 2]),
                 "weak_bias": (bias[..., 1], present[..., 1]),
                 "strong_bias": (bias[..., 2], present[..., 2]),
                 "flat_response": (response[..., 0], present[..., 0])}
        # 容忍固定像素位移的双向边缘匹配；全无边缘且无误报时 F1=1。
        truth_edge = target >= self.config.thresholds[1]
        predicted_edge = prediction >= self.config.thresholds[1]
        kernel = 2 * self.tolerance + 1
        near_truth = F.max_pool2d(truth_edge.float(), kernel, stride=1, padding=self.tolerance) > 0
        near_prediction = F.max_pool2d(predicted_edge.float(), kernel, stride=1, padding=self.tolerance) > 0
        n_truth, n_pred = truth_edge.sum((-2, -1)), predicted_edge.sum((-2, -1))
        precision = torch.where(n_pred > 0, (predicted_edge & near_truth).sum((-2, -1)) / n_pred.clamp_min(1),
                                (n_truth == 0).float())
        recall = torch.where(n_truth > 0, (truth_edge & near_prediction).sum((-2, -1)) / n_truth.clamp_min(1),
                             torch.ones_like(precision))
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        stats.update(edge_precision=(precision, always), edge_recall=(recall, always), edge_f1=(f1, always))
        x = prediction - prediction.mean((-2, -1), keepdim=True)
        y = target - target.mean((-2, -1), keepdim=True)
        vx, vy = x.square().sum((-2, -1)), y.square().sum((-2, -1))
        valid_corr = (vx > 1e-12) & (vy > 1e-12)
        corr = (x * y).sum((-2, -1)) / (vx * vy).sqrt().clamp_min(1e-12)
        stats["spatial_correlation"] = (corr, valid_corr)
        batch = prediction.shape[0]
        rows = [torch.stack((prediction.new_ones(batch), prediction.new_zeros(batch)), dim=-1)]
        for scale in (None, *range(self.scales)):
            for name in self.metric_names:
                values, valid = stats[name]
                if scale is None:
                    numerator, denominator = (values * valid).sum(1), valid.sum(1)
                else:
                    numerator, denominator = values[:, scale] * valid[:, scale], valid[:, scale]
                rows.append(torch.stack((numerator, denominator), dim=-1))
        rows = torch.stack(rows, dim=1).double()
        self.totals[0] += rows.sum(0)
        if self.scenes:
            if scene_names is None or len(scene_names) != batch or set(scene_names) - set(self.scenes):
                raise ValueError("Validation scene metadata is missing or inconsistent.")
            for index, scene in enumerate(self.scenes, 1):
                subset = [i for i, value in enumerate(scene_names) if value == scene]
                if subset:
                    self.totals[index] += rows[subset].sum(0)

    def synchronize(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.totals)

    def compute(self):
        result = []
        for total in self.totals.cpu().tolist():
            row = {name: numerator / denominator if denominator else None
                   for name, (numerator, denominator) in zip(self.names, total[1:])}
            row["samples"] = int(total[0][0])
            row["metric_counts"] = {name: int(values[1]) for name, values in zip(self.names, total[1:])}
            result.append(row)
        return dict(overall=result[0], by_scene=dict(zip(self.scenes, result[1:])))
