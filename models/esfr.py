"""E-SFR 一步修正的可微展开：外层图像损失回传到结构头和 decoder fusion。"""
import torch
from models.event_structure_guidance import (FullFeatureCorrection, StructureObservation,
    freeze_observation_masks, observe_events, _losses)


def joint_observation(head, frozen_teacher, events, threshold):
    """结构目标可训练；置信度、事件支持域、保护掩码由独立冻结教师固定。"""
    edges, _ = head(events.float())
    teacher = observe_events(frozen_teacher, events, threshold)
    return freeze_observation_masks(StructureObservation(edges, teacher.confidence)), edges


def differentiable_guidance(render, observation, config, operator, *, create_graph=True):
    """无 GT 接口；保留一步归一化梯度/信赖域投影的图，硬验收只作离散选择。

    create_graph=True 保留混合二阶项，不将接受的 latent/image detach。
    渲染器须固定 RNG 与条件；验证使用同一接受规则但不保留二阶图。
    """
    if config.steps != 1:
        raise ValueError('E-SFR supports exactly one guidance step.')
    observation.validate()
    correction = FullFeatureCorrection(observation.feature_mask)
    raw = render(correction)
    baseline = raw.detach().clamp(-1, 1)
    stats = dict(accepted_steps=0, forward_calls=1, uses_gt_for_guidance=False,
                 higher_order=create_graph)
    if not bool(observation.confidence.any()):
        return raw.clamp(-1, 1), stats
    with torch.no_grad():
        initial, _ = _losses(baseline, baseline, observation, operator, config)
    loss, _ = _losses(raw, baseline, observation, operator, config)
    scale = 1024. if raw.dtype == torch.float16 else 1.
    gradient, = torch.autograd.grad(loss * scale, correction.delta,
                                    create_graph=create_graph, retain_graph=True)
    gradient = gradient.float() / scale
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite inner gradient')
    norm = gradient.square().mean().sqrt()
    stats['gradient_rms'] = float(norm.detach())
    if float(norm.detach()) < 1e-12:
        return raw.clamp(-1, 1), stats
    direction = gradient / norm.clamp_min(1e-12) * correction.channel_scale
    for retry in range(config.backtracking_steps + 1):
        proposal = -config.learning_rate * (0.5 ** retry) * direction
        # 与原 project 数值一致，但训练时不能 detach，否则 H1 收不到最终图像梯度。
        ratio = (config.trust_radius / correction.relative_norm(proposal).clamp_min(1e-8)).clamp(max=1)
        correction.delta = proposal * ratio
        candidate = render(correction).clamp(-1, 1)
        stats['forward_calls'] += 1
        with torch.no_grad():
            score, parts = _losses(candidate, baseline, observation, operator, config)
            allowed = (torch.isfinite(score) and score < initial and
                parts['low'] <= config.max_low_change and parts['keep'] <= config.max_outside_change)
        if allowed:
            stats.update(accepted_steps=1, retry=retry, initial_objective=float(initial),
                         final_objective=float(score), relative_delta=float(correction.relative_norm(correction.delta).detach()))
            return candidate, stats
    return raw.clamp(-1, 1), stats


def guided_forward(gen, head, teacher, sample, config, operator, *, enabled=True, create_graph=True):
    """固定 Stage3 前缀及条件；GT 留在调用方，仅外层监督可读取。"""
    rgb=sample['conditioning_pixel_values'][None].cuda()
    events=sample['events'][None].cuda()
    evref=sample['events_refinement'][None].cuda()
    evhead=sample['events_structure'][None].cuda()
    assert not torch.count_nonzero(evref[:,1:])
    prompts=dict(prompt=[sample['prompt']],neg_prompt=[sample['neg_prompt']])
    with torch.no_grad():
        context=gen.prepare_denoising_context(rgb.float(),events.float(),prompts)
        with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
            condition=gen.prepare_decoder_refinement_condition(rgb,evref,'both')
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    def render(correction):
        with torch.random.fork_rng(devices=[rgb.device.index]):
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
            z,_=gen.predict_denoised_latent_from_context(context,
                torch.tensor([gen.args.generator_timestep],device=rgb.device),feature_guidance=correction)
            with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
                return gen.decode_denoised_latent(z,rgb,events_refinement=evref,
                    decoder_fusion_enabled=True,refinement_fusion_mode='both',clamp_output=False,
                    refinement_condition=condition)
    if not enabled:
        return render(None).clamp(-1,1), None, dict(accepted_steps=0)
    observation,edges=joint_observation(head,teacher,evhead,config.min_confidence)
    image,stats=differentiable_guidance(render,observation,config,operator,create_graph=create_graph)
    return image,edges,stats
