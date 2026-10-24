"""idea7：冻结网络，仅对一个完整 up-block 的全通道特征做推理期修正。"""

from dataclasses import dataclass
import math
from typing import Callable, Optional

import torch


class SkipFeatureDPS(RuntimeError):
    """观测工厂发现有效区域不足等可恢复情况，保留本次 baseline。"""


@dataclass(frozen=True)
class FeatureDPSConfig:
    steps: int = 1
    up_block_index: Optional[int] = None
    step_rel: float = 0.01
    radius_rel: float = 0.05
    eps: float = 1e-12
    gradient_eps: float = 1e-12

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps not in (0, 1, 2):
            raise ValueError("Feature DPS MVP supports steps=0, 1 or 2.")
        if self.up_block_index is not None and (
            isinstance(self.up_block_index, bool)
            or not isinstance(self.up_block_index, int)
            or self.up_block_index < 0
        ):
            raise ValueError("up_block_index must be a nonnegative integer or None.")
        for name in ("step_rel", "radius_rel", "eps", "gradient_eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.step_rel == 0 or self.eps == 0:
            raise ValueError("step_rel and eps must be positive.")


def _rms(tensor):
    # float32 下统计相对尺度；不让 fp16 的平方/求和溢出或吞掉小梯度。
    return tensor.float().square().mean().sqrt()


class FeatureCorrectionState:
    """单样本临时状态；不继承 nn.Module，不注册参数或写入 checkpoint。"""

    def __init__(self, config):
        self.config = config
        self.u = None
        self.scale = None
        self.feature_shape = None
        self.feature_dtype = None
        self.block_index = None
        self._num_blocks = None
        self._hits = 0

    def begin_forward(self):
        self._hits = 0

    def bind_num_up_blocks(self, count):
        index = count - 2 if self.config.up_block_index is None else self.config.up_block_index
        if count < 2 or not 0 <= index < count - 1:
            raise ValueError("Feature correction must leave at least one full up-block afterwards.")
        if self._num_blocks is not None and count != self._num_blocks:
            raise ValueError("UNet up-block count changed between candidates.")
        self._num_blocks, self.block_index = count, index

    def apply(self, block_index, features):
        if self.block_index is None:
            raise RuntimeError("Call bind_num_up_blocks before applying feature correction.")
        if block_index != self.block_index:
            return features
        self._hits += 1
        if self._hits != 1:
            raise RuntimeError("Target up-block was visited more than once in one forward.")
        if features.ndim != 4 or features.shape[0] != 1 or not features.is_floating_point():
            raise ValueError("Feature DPS requires floating [1,C,H,W] up-block features.")
        if not torch.isfinite(features).all():
            raise FloatingPointError("Uncorrected up-block feature is not finite.")
        if self.u is None:
            # 首轮走到注入点时创建叶变量，零注入同时生成 baseline，无额外捕获前向。
            self.scale = _rms(features.detach()).clamp_min(self.config.eps).detach()
            if not torch.isfinite(self.scale):
                raise FloatingPointError("Feature RMS is not finite.")
            self.u = torch.zeros_like(features, dtype=torch.float32, requires_grad=True)
            self.feature_shape = tuple(features.shape)
            self.feature_dtype = features.dtype
        elif (
            tuple(features.shape) != self.feature_shape
            or features.device != self.u.device
            or features.dtype != self.feature_dtype
        ):
            raise ValueError("Feature shape, device or dtype changed between candidates.")
        # h0 每轮由固定原始输入重新算出；仅 u 累计更新，不能累计已修正后的 h。
        return features + (self.scale * self.u).to(dtype=features.dtype)

    def finish_forward(self):
        if self._hits != 1 or self.u is None:
            raise RuntimeError("Configured feature correction point was not visited exactly once.")

    def diagnostics(self):
        if self.u is None:
            return {"feature_shape": None, "up_block_index": self.block_index,
                    "relative_rms": 0.0, "relative_max": 0.0}
        value = self.u.detach()
        return {
            "feature_shape": list(self.feature_shape),
            "up_block_index": self.block_index,
            "relative_rms": float(_rms(value).item()),
            "relative_max": float(value.abs().max().item()),
        }

    def update(self, gradient):
        rms = _rms(gradient)
        value = self.u.detach() - self.config.step_rel * gradient.detach().float() / rms.clamp_min(self.config.eps)
        # 硬投影限制相对 RMS 总改动；这不是对每个位置分别限制最大值。
        factor = (self.config.radius_rel / _rms(value).clamp_min(self.config.eps)).clamp(max=1.0)
        self.u = (value * factor).detach().requires_grad_(True)

    def clear(self):
        self.u = None
        self.scale = None
        self.feature_shape = None
        self.feature_dtype = None
        self._hits = 0


@dataclass
class FeatureDPSResult:
    baseline: torch.Tensor
    image: torch.Tensor
    stats: dict
    history: list


def freeze_for_feature_dps(model):
    """在权重加载和 adapter 激活之后调用；保留对输入特征求导的能力。"""
    model.eval()
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
        if hasattr(module, "use_tiling"):
            module.use_tiling = False
    return model


class FeatureDPSRunner:
    """generate(state)->[0,1] RGB；loss_factory(baseline)->固定 scalar energy 函数。

    generate 负责一次完整生成，不能缓存修正点后输出。工厂只调用一次，允许根据
    初始 baseline 创建固定 mask；它不接收 GT。cleanup 必须幂等，用于释放模型诊断缓存。
    """

    def __init__(self, config=None):
        self.config = config or FeatureDPSConfig()

    def run(self, generate: Callable, loss_factory=None, *, skip_reason=None,
            cleanup=None, cuda_devices=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Feature DPS must run outside torch.inference_mode().")
        active = self.config.steps > 0 and self.config.radius_rel > 0 and not skip_reason
        if active and loss_factory is None:
            raise ValueError("Active feature DPS requires a loss_factory.")
        if cuda_devices is None:
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        state = FeatureCorrectionState(self.config) if active else None
        baseline = best = None
        best_energy = None
        stats = {
            "selected_iteration": 0, "skip_reason": skip_reason or "",
            "forward_calls": 0, "backward_calls": 0,
            "energy_before": None, "energy_after": None,
            "feature_shape": None, "up_block_index": None,
            "relative_rms": 0.0, "relative_max": 0.0,
        }
        history = []
        try:
            # 每个候选复用同一 Torch RNG 状态（包括 scheduler 的潜在随机项）。
            # fork_rng 在成功/异常时均恢复调用者 RNG，条件编码应在进入 runner 前固定。
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                cpu_rng = torch.get_rng_state()
                gpu_rng = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}
                energy_function = None
                iterations = self.config.steps if active else 0
                for index in range(iterations + 1):
                    torch.set_rng_state(cpu_rng)
                    for device, rng in gpu_rng.items():
                        torch.cuda.set_rng_state(rng, device)
                    wants_gradient = active and index < self.config.steps
                    if state is not None:
                        state.begin_forward()
                    try:
                        with torch.set_grad_enabled(wants_gradient):
                            stats["forward_calls"] += 1
                            try:
                                image = generate(state)
                            except FloatingPointError:
                                if baseline is None:
                                    raise
                                stats["skip_reason"] = "non_finite_candidate"
                                break
                            if state is not None:
                                state.finish_forward()
                            if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[:2] != (1, 3):
                                raise ValueError("generate must return RGB [1,3,H,W].")
                            if not torch.isfinite(image).all():
                                if baseline is None:
                                    raise FloatingPointError("Baseline image is not finite.")
                                stats["skip_reason"] = "non_finite_candidate"
                                break
                            if image.detach().min() < -1e-6 or image.detach().max() > 1.0 + 1e-6:
                                raise ValueError("generate must return image values in [0,1].")
                            if baseline is None:
                                baseline = image.detach().clone()
                                best = baseline
                            if not active:
                                stats["skip_reason"] = skip_reason or ("steps_zero" if self.config.steps == 0 else "radius_zero")
                                break
                            diagnostic = state.diagnostics()
                            if energy_function is None:
                                try:
                                    energy_function = loss_factory(baseline)
                                except SkipFeatureDPS as exc:
                                    stats["skip_reason"] = str(exc) or "observation_unavailable"
                                    stats.update(diagnostic)
                                    break
                                if not callable(energy_function):
                                    raise TypeError("loss_factory must return a callable energy function.")
                            energy = energy_function(image)
                            if not isinstance(energy, torch.Tensor) or energy.numel() != 1:
                                raise ValueError("Observation energy must be a scalar Tensor.")
                            if not torch.isfinite(energy).all():
                                stats["skip_reason"] = "non_finite_energy"
                                break
                            value = float(energy.detach().item())
                            row = {"iteration": index, "energy": value, **diagnostic}
                            history.append(row)
                            if index == 0:
                                stats["energy_before"] = value
                                best_energy = value
                                stats.update(diagnostic)
                            if value < best_energy:
                                best_energy = value
                                best = image.detach().clone()
                                stats["selected_iteration"] = index
                                stats.update(diagnostic)
                            stats["energy_after"] = best_energy
                            if not wants_gradient:
                                continue
                            if not energy.requires_grad:
                                stats["skip_reason"] = "disconnected_energy"
                                break
                            stats["backward_calls"] += 1
                            gradient = torch.autograd.grad(energy, state.u, allow_unused=True)[0]
                            if gradient is None:
                                stats["skip_reason"] = "disconnected_energy"
                                break
                            if not torch.isfinite(gradient).all():
                                stats["skip_reason"] = "non_finite_gradient"
                                break
                            gradient_rms = float(_rms(gradient).item())
                            row["gradient_rms"] = gradient_rms
                            if not math.isfinite(gradient_rms):
                                stats["skip_reason"] = "non_finite_gradient"
                                break
                            if gradient_rms <= self.config.gradient_eps:
                                stats["skip_reason"] = "zero_gradient"
                                break
                            state.update(gradient)
                    finally:
                        # 每轮反传完成后立即释放 last_aux 等可能持图的诊断，异常也执行。
                        if cleanup is not None:
                            cleanup()
                if active and not stats["skip_reason"] and stats["selected_iteration"] == 0:
                    stats["skip_reason"] = "no_energy_improvement"
            return FeatureDPSResult(baseline=baseline, image=best, stats=stats, history=history)
        finally:
            if state is not None:
                state.clear()
            if cleanup is not None:
                cleanup()
