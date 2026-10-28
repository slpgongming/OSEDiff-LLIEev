"""固定结构观测，在推理时优化完整通道主干特征；默认观测仅依赖事件。"""

from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F

from models.image_structure import StructureOperatorConfig, gaussian_blur, image_structure


@dataclass(frozen=True)
class GuidanceConfig:
    steps: int = 0
    learning_rate: float = 0.01
    trust_radius: float = 0.03
    lambda_low: float = 1.0
    lambda_keep: float = 0.5
    lambda_range: float = 1.0
    max_low_change: float = 0.02
    max_outside_change: float = 0.02
    backtracking_steps: int = 3
    min_confidence: float = 0.25
    protection_mode: str = "quadratic"
    trace_candidates: bool = True

    def __post_init__(self):
        if self.protection_mode not in {"quadratic", "l1"}:
            raise ValueError("protection_mode must be quadratic or l1.")
        if not isinstance(self.trace_candidates, bool):
            raise ValueError("trace_candidates must be bool.")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in (self.steps, self.backtracking_steps)):
            raise ValueError("Guidance step counts must be nonnegative integers.")
        for name in ("learning_rate", "trust_radius", "lambda_low", "lambda_keep", "lambda_range",
                     "max_low_change", "max_outside_change", "min_confidence"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid guidance {name}={value}")
        if self.learning_rate == 0 or self.trust_radius == 0 or self.min_confidence > 1:
            raise ValueError("Positive learning rate/trust radius and confidence <= 1 required.")


@dataclass(frozen=True)
class StructureObservation:
    edges: torch.Tensor
    confidence: torch.Tensor
    # 独立控制结构误差、允许修改的区域、需要保护的区域；None 保留原推理行为。
    feature_mask: torch.Tensor = None
    protection_mask: torch.Tensor = None

    def validate(self):
        if self.edges.ndim != 4 or not self.edges.numel() or self.edges.shape != self.confidence.shape:
            raise ValueError("Observation expects equal [B,S,H,W] edge/confidence maps.")
        for tensor in (self.edges, self.confidence):
            if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
                raise ValueError("Structure observations must be finite and in [0,1].")
        expected = (self.edges.shape[0], 1, *self.edges.shape[-2:])
        for tensor in (self.feature_mask, self.protection_mask):
            if tensor is not None and (tuple(tensor.shape) != expected or
                    not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1):
                raise ValueError("Independent guidance masks must be finite [B,1,H,W] maps in [0,1].")


def freeze_observation_masks(observation):
    """从原事件置信度固定 M0/K0；随后替换目标或误差权重不会改变它们。"""
    mean_confidence = observation.confidence.detach().mean(1, keepdim=True)
    return replace(observation,
                   feature_mask=mean_confidence if observation.feature_mask is None else observation.feature_mask.detach(),
                   protection_mask=1 - mean_confidence if observation.protection_mask is None else observation.protection_mask.detach())


@torch.no_grad()
def observe_events(predictor, events, min_confidence=0.25):
    """目标只依赖事件；无事件区域不作为平滑图像的负证据。"""
    edges, confidence = predictor(events.float())
    support = (events.abs().sum(1, keepdim=True) > 0).float()
    support = F.max_pool2d(support, 3, stride=1, padding=1)
    confidence = confidence * support * (confidence >= min_confidence)
    observation = StructureObservation(edges.detach(), confidence.detach())
    observation.validate()
    return observation


class FullFeatureCorrection:
    """一次样本的临时 [B,C,H,W] leaf；不是 nn.Module/Parameter，不进入 state_dict。"""
    def __init__(self, image_mask):
        self.image_mask = image_mask.detach().float()
        self.h0 = self.delta = self.mask = self.channel_scale = None

    def apply(self, feature):
        if feature.ndim != 4:
            raise ValueError("Feature correction requires [B,C,H,W].")
        if self.h0 is None:
            self.h0 = feature.detach()
            self.mask = F.interpolate(self.image_mask, feature.shape[-2:], mode="area")
            self.channel_scale = self.h0.float().square().mean((2, 3), keepdim=True).sqrt().clamp_min(1e-3)
            self.delta = torch.zeros_like(feature, dtype=torch.float32, requires_grad=True)
        if feature.shape != self.h0.shape:
            raise ValueError("Feature shape changed within a fixed sample.")
        # 转换 dtype 保留对 FP32 delta 的梯度；全部 C 通道均可调。
        return self.h0 + (self.mask * self.delta).to(feature.dtype)

    def relative_norm(self, delta):
        change = self.mask * delta / self.channel_scale
        denominator = self.mask.square().sum() * delta.shape[1]
        return (change.square().sum() / denominator.clamp_min(1e-8)).sqrt()

    def project(self, delta, radius):
        ratio = (radius / self.relative_norm(delta).clamp_min(1e-8)).clamp(max=1)
        return (delta * ratio).detach().requires_grad_(True)

    def clear(self):
        self.h0 = self.delta = self.mask = self.channel_scale = None


def _losses(raw, baseline, observation, operator, config):
    image = raw.float() * 0.5 + 0.5
    base = baseline.float() * 0.5 + 0.5
    edges = image_structure(image, operator)
    confidence = observation.confidence
    if edges.shape != observation.edges.shape:
        raise ValueError("Image/observation structure shape mismatch.")
    residual = F.huber_loss(edges, observation.edges, reduction="none", delta=0.1)
    structure = (confidence * residual).sum() / confidence.sum().clamp_min(1e-8)
    low_difference = gaussian_blur(image, 3.0) - gaussian_blur(base, 3.0)
    difference = image - base
    low = low_difference.abs().mean()
    outside = observation.protection_mask
    if outside is None:
        outside = 1 - confidence.mean(1, keepdim=True)
    denominator = (3 * outside.sum()).clamp_min(1e-8)
    keep = (outside * difference.abs()).sum() / denominator
    # 软惩罚与硬预算分离：二次项在零点附近平滑，避免 L1 成本压过微小结构收益。
    # low/keep 始终保留 L1 含义，不能因更换软惩罚而放宽 0.02 的验收阈值。
    if config.protection_mode == "quadratic":
        penalty_low = low_difference.square().mean()
        penalty_keep = (outside * difference.square()).sum() / denominator
    else:
        penalty_low, penalty_keep = low, keep
    protection = config.lambda_low * penalty_low + config.lambda_keep * penalty_keep
    range_loss = (F.relu(-image).square() + F.relu(image-1).square()).mean()
    total = structure + protection + config.lambda_range * range_loss
    return total, dict(structure=structure, low=low, keep=keep, range=range_loss,
                      penalty_low=penalty_low, penalty_keep=penalty_keep, protection=protection)


def _loss_numbers(values):
    """只保存有限标量；异常候选使用 null，避免生成含 NaN/Infinity 的诊断 JSON。"""
    numbers = {key: float(value.detach()) for key, value in values.items()}
    return {key: value if math.isfinite(value) else None for key, value in numbers.items()}


def optimize_features(render, observation, config, operator=None):
    """render(correction) 返回 [-1,1] 定义下的 raw RGB；只对 delta 反传。

    render 必须使用固定 latent/条件/RNG，且权重冻结。所有接受判断在最终 clamp
    后图像上进行；优化 raw 域以免饱和像素断梯度。重试计入 forward 次数。
    """
    operator = operator or StructureOperatorConfig()
    stats = dict(forward_calls=0, backward_calls=0, accepted_steps=0, reason="disabled")
    if config.trace_candidates:
        stats["candidate_history"] = []
        stats["gradient_history"] = []
    if config.steps == 0:
        with torch.no_grad():
            image = render(None).detach().clamp(-1, 1)
        stats["forward_calls"] = 1
        return image, stats
    if observation is None:
        raise ValueError("Positive guidance steps require a fixed event observation.")
    observation.validate()
    observation = replace(observation, edges=observation.edges.detach(), confidence=observation.confidence.detach(),
                          feature_mask=observation.feature_mask.detach() if observation.feature_mask is not None else None,
                          protection_mask=observation.protection_mask.detach() if observation.protection_mask is not None else None)
    stats.update(mean_confidence=float(observation.confidence.mean()),
                 supported_fraction=float((observation.confidence.amax(1) > 0).float().mean()))
    if observation.edges.shape[0] != 1:
        raise ValueError("Initial idea8 guidance supports batch size 1.")
    if not observation.confidence.any():
        with torch.no_grad():
            image = render(None).detach().clamp(-1, 1)
        stats.update(forward_calls=1, reason="no_event_support")
        return image, stats
    feature_mask = observation.feature_mask
    if feature_mask is None:
        feature_mask = observation.confidence.mean(1, keepdim=True)
    correction = FullFeatureCorrection(feature_mask)
    try:
        with torch.enable_grad():
            raw = render(correction)
            stats["forward_calls"] += 1
            if correction.delta is None:
                raise RuntimeError("UNet did not execute the requested feature correction hook.")
            if not torch.isfinite(raw).all():
                raise RuntimeError("Baseline itself is non-finite; cannot provide a safe fallback.")
            baseline = raw.detach().clamp(-1, 1)
            result = baseline
            with torch.no_grad():
                score, initial_values = _losses(baseline, baseline, observation, operator, config)
                current_score = float(score)
                current_structure = float(initial_values["structure"])
            stats.update(initial_objective=current_score, reason="completed", relative_delta=0.0,
                         feature_shape=list(correction.h0.shape),
                         protection_mode=config.protection_mode, initial_losses=_loss_numbers(initial_values))
            for step in range(config.steps):
                loss, raw_values = _losses(raw, baseline, observation, operator, config)
                if config.trace_candidates:
                    stats["gradient_history"].append(dict(step=step + 1,
                        raw_losses=_loss_numbers(dict(total=loss, **raw_values))))
                if not torch.isfinite(loss):
                    stats["reason"] = "nonfinite_loss"
                    break
                # FP16 后段中的梯度可能小于可表示范围；先放大再以 FP32 还原。
                loss_scale = 1024.0 if raw.dtype == torch.float16 else 1.0
                gradient, = torch.autograd.grad(loss * loss_scale, correction.delta)
                gradient = gradient.float() / loss_scale
                stats["backward_calls"] += 1
                stats["gradient_loss_scale"] = loss_scale
                if not torch.isfinite(gradient).all():
                    stats["reason"] = "nonfinite_gradient"
                    break
                norm = gradient.square().mean().sqrt()
                if config.trace_candidates:
                    stats["gradient_history"][-1]["gradient_rms"] = float(norm)
                if float(norm) < 1e-12:
                    stats["reason"] = "zero_gradient"
                    break
                old_delta = correction.delta.detach()
                direction = gradient / norm.clamp_min(1e-12) * correction.channel_scale
                accepted = False
                for retry in range(config.backtracking_steps + 1):
                    step_size = config.learning_rate * (0.5 ** retry)
                    proposal = old_delta - step_size * direction
                    correction.delta = correction.project(proposal, config.trust_radius)
                    candidate_raw = render(correction)
                    stats["forward_calls"] += 1
                    if not torch.isfinite(candidate_raw).all():
                        if config.trace_candidates:
                            stats["candidate_history"].append(dict(step=step + 1, retry=retry,
                                learning_rate=step_size, accepted=False, rejection_reasons=["nonfinite_candidate"]))
                        continue
                    candidate = candidate_raw.detach().clamp(-1, 1)
                    with torch.no_grad():
                        score, values = _losses(candidate, baseline, observation, operator, config)
                        reasons = []
                        if not torch.isfinite(score):
                            reasons.append("nonfinite_objective")
                        elif float(score) >= current_score:
                            reasons.append("objective_not_lower")
                        if float(values["low"]) > config.max_low_change:
                            reasons.append("low_budget")
                        if float(values["keep"]) > config.max_outside_change:
                            reasons.append("keep_budget")
                        allowed = not reasons
                        if config.trace_candidates:
                            raw_score, raw_parts = _losses(candidate_raw, baseline, observation, operator, config)
                            improvement = current_structure - float(values["structure"])
                            stats["candidate_history"].append(dict(step=step + 1, retry=retry,
                                learning_rate=step_size, accepted=allowed, rejection_reasons=reasons,
                                objective_before=current_score, losses=_loss_numbers(dict(total=score, **values)),
                                raw_losses=_loss_numbers(dict(total=raw_score, **raw_parts)),
                                structure_improvement=improvement if math.isfinite(improvement) else None,
                                relative_delta=float(correction.relative_norm(correction.delta))))
                    if allowed:
                        raw, result, current_score = candidate_raw, candidate, float(score)
                        current_structure = float(values["structure"])
                        stats["accepted_steps"] += 1
                        stats["relative_delta"] = float(correction.relative_norm(correction.delta).detach())
                        accepted = True
                        break
                if not accepted:
                    stats["reason"] = "no_acceptable_candidate"
                    break
            with torch.no_grad():
                _, final_values = _losses(result, baseline, observation, operator, config)
            stats.update(final_objective=current_score,
                         final_losses=_loss_numbers(final_values),
                         saturated_fraction=float((result.abs() >= 1).float().mean()))
            return result, stats
    finally:
        correction.clear()


def restore_sample(gen, context, timesteps, rgb, config, observation=None, operator=None,
                   events_refinement=None, refinement_enabled=False):
    """生成器桥接：固定 RNG，恢复模式/梯度开关，清理缓存。接口无 GT。"""
    from models.event_lora_utils import clear_event_feats_for_all_event_lora, iter_event_lora_modules

    if config.steps and (refinement_enabled or getattr(gen, "decoder_refinement_enabled", False)):
        raise ValueError("Positive guidance steps currently require decoder refinement off.")
    if config.steps and (rgb.shape[0] != 1 or rgb.shape[-2:] != (256, 256)):
        raise ValueError("Positive guidance steps initially support [1,3,256,256].")
    # baseline 按既有路径执行一次，不改变全局随机序列/训练开关。
    if config.steps == 0:
        with torch.no_grad():
            latent, _ = gen.predict_denoised_latent_from_context(context, timesteps)
            image = gen.decode_denoised_latent(latent, rgb, events_refinement=events_refinement,
                                              decoder_fusion_enabled=refinement_enabled,
                                              refinement_fusion_mode="both")
        return image, dict(forward_calls=1, backward_calls=0, accepted_steps=0, reason="disabled")
    parameters = [(p, p.requires_grad) for p in gen.parameters()]
    modes = [(module, module.training) for module in gen.modules()]
    checkpointing = [(module, module.gradient_checkpointing) for module in gen.modules()
                     if hasattr(module, "gradient_checkpointing")]
    devices = [rgb.device.index if rgb.device.index is not None else torch.cuda.current_device()] if rgb.is_cuda else []
    cpu_state = torch.get_rng_state()
    cuda_states = {device: torch.cuda.get_rng_state(device) for device in devices}

    def render(correction):
        # fork_rng 保护调用方 RNG，同时每个候选使用相同 scheduler 噪声。
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_state)
            for device, state in cuda_states.items():
                torch.cuda.set_rng_state(state, device)
            latent, _ = gen.predict_denoised_latent_from_context(
                context, timesteps, feature_guidance=correction)
            return gen.decode_denoised_latent(latent, rgb, decoder_fusion_enabled=False, clamp_output=False)
    try:
        gen.eval().requires_grad_(False)
        for module, _ in checkpointing:
            module.gradient_checkpointing = False
        return optimize_features(render, observation, config, operator)
    finally:
        clear_event_feats_for_all_event_lora(gen.unet)
        # 原训练路径的 last_aux 用于损失，不能全局 detach；只在在线优化结束时释放。
        for _, layer in iter_event_lora_modules(gen.unet):
            layer.last_aux = {}
            layer.last_visuals = {}
            layer.last_z_delta = None
        for param, enabled in parameters:
            param.requires_grad_(enabled)
        for module, mode in modes:
            module.training = mode
        for module, enabled in checkpointing:
            module.gradient_checkpointing = enabled
