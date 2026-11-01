"""冻结 H1 内层特征修正，再对接受的 latent 训练 decoder refinement。"""
from dataclasses import asdict
import hashlib
import torch

from models.event_structure_guidance import (GuidanceConfig, observe_events,
                                            freeze_observation_masks, optimize_features)
from models.event_structure_head import load_structure_checkpoint
from models.event_lora_utils import clear_event_feats_for_all_event_lora, iter_event_lora_modules
from osediff_event_film import OSEDiffEventFiLMGen


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def guided_latents(gen, rgb, events, events_structure, prompts, predictor, operator, config):
    """按样本求 delta；[B,4,H/8,W/8] 输出完全 detach，外层梯度只经过后续 decoder。

    H1 的目标、内层损失与接受条件均不接收 GT。候选使用相同 context/RNG，
    on_accept 只保留通过原硬预算验收的 latent，拒绝候选不能泄漏到外层训练。
    """
    if rgb.shape[0] != events.shape[0] or rgb.shape[0] != events_structure.shape[0]:
        raise ValueError('H1 training views must have the same batch size.')
    if rgb.shape[-2:] != (256, 256):
        raise ValueError('H1 Stage4 requires 256-square aligned inputs.')
    if any(p.requires_grad for name, p in gen.named_parameters()
           if 'vae.decoder.feature_fusion_blocks.' not in name):
        raise RuntimeError('H1 inner optimization requires a frozen generator and predictor.')
    latents, diagnostics = [], []
    devices = [rgb.device.index] if rgb.is_cuda else []
    try:
        # 外层 fp16 仅用于 refinement 训练；H1 保持已验证的 FP32 数值路径。
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            for index in range(rgb.shape[0]):
                image = rgb[index:index+1].float()
                with torch.no_grad():
                    context = gen.prepare_denoising_context(image, events[index:index+1].float(),
                        dict(prompt=[prompts['prompt'][index]], neg_prompt=[prompts['neg_prompt'][index]]))
                    observation = freeze_observation_masks(observe_events(
                        predictor, events_structure[index:index+1].float(), config.min_confidence))
                cpu_state = torch.get_rng_state()
                cuda_state = torch.cuda.get_rng_state(rgb.device) if rgb.is_cuda else None
                current = {}

                def render(correction):
                    with torch.random.fork_rng(devices=devices):
                        torch.set_rng_state(cpu_state)
                        if rgb.is_cuda:
                            torch.cuda.set_rng_state(cuda_state, rgb.device)
                        latent, _ = gen.predict_denoised_latent_from_context(context,
                            torch.tensor([gen.args.generator_timestep], device=rgb.device, dtype=torch.long),
                            feature_guidance=correction)
                        current['last'] = latent.detach()
                        # 直接禁用 fusion 解码，避免内层调用 REEncoder 或依赖可训练 fusion。
                        return gen.vae.decode(latent / gen.vae.config.scaling_factor,
                                              fusion_enabled=False).sample

                def accept():
                    current['accepted'] = current['last']

                _, stats = optimize_features(render, observation, config, operator, on_accept=accept)
                if 'accepted' not in current:
                    raise RuntimeError('No accepted baseline latent was captured.')
                latents.append(current['accepted']); diagnostics.append(stats)
                # 不保留内层 autograd 图及 Event-LoRA 中间统计，防止跨样本累积显存。
                clear_event_feats_for_all_event_lora(gen.unet)
                for _, layer in iter_event_lora_modules(gen.unet):
                    layer.last_aux = {}; layer.last_visuals = {}; layer.last_z_delta = None
                del context, observation, current
    finally:
        clear_event_feats_for_all_event_lora(gen.unet)
    return torch.cat(latents).detach(), diagnostics


def refinement_guided_sample(gen, rgb, events, events_structure, prompts, predictor, operator,
                            config, *, events_refinement, refinement_fusion_mode="both",
                            refinement_autocast_dtype=None, preserve_rng=True):
    """冻结完整 Stage4，在最终 decoder 输出上优化/验收临时特征；接口不接收 GT。

    返回接受的 (clamped RGB, detached latent, diagnostics)。模型权重、原有参数梯度、
    train/eval 和 checkpointing 开关均保留；输入与条件 detach，仅 delta 可求导。
    decoder 的当前层 mask/全局强度直接生效，调用方必须为每个设置独立求解 delta。
    """
    if rgb.shape != (1, 3, 256, 256):
        raise ValueError("Full-refinement H1 requires [1,3,256,256] RGB.")
    if not getattr(gen, 'decoder_refinement_enabled', False):
        raise ValueError("Full-refinement H1 requires an installed Stage4 decoder.")
    for tensor in (events, events_structure, events_refinement):
        if tensor is None or tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("Full-refinement H1 requires aligned per-sample event views.")
        if tensor.device != rgb.device:
            raise ValueError("All H1 inputs must use the same device.")
    if set(prompts) != {'prompt', 'neg_prompt'}:
        raise ValueError("Pass only prompt/neg_prompt; GT and training batches are forbidden.")
    if refinement_autocast_dtype not in (None, torch.float16, torch.bfloat16):
        raise ValueError("Unsupported refinement autocast dtype.")
    # 冻结权重并不关闭 autograd：UNet suffix + 完整 VAE/fusion 仍须向 delta 传梯度。
    modules = list(dict.fromkeys([*gen.modules(), *predictor.modules()]))
    parameters = list(dict.fromkeys([*gen.parameters(), *predictor.parameters()]))
    flags = [(parameter, parameter.requires_grad) for parameter in parameters]
    modes = [(module, module.training) for module in modules]
    checkpointing = [(module, module.gradient_checkpointing) for module in modules
                     if hasattr(module, 'gradient_checkpointing')]
    devices = [rgb.device.index] if rgb.is_cuda else []
    rgb, events, events_structure, events_refinement = (
        tensor.detach() for tensor in (rgb, events, events_structure, events_refinement))
    accepted = {}
    next_context_rng = {}

    def decoder_precision():
        return torch.autocast(device_type=rgb.device.type, dtype=refinement_autocast_dtype,
                              enabled=refinement_autocast_dtype is not None, cache_enabled=False)

    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        for module in modules:
            module.training = False
        for module, _ in checkpointing:
            module.gradient_checkpointing = False
        # 准备和候选均保护调用方 RNG；同一设置所有候选共用 scheduler 随机数。
        with torch.random.fork_rng(devices=devices), torch.autocast(rgb.device.type, enabled=False):
            with torch.no_grad():
                context = gen.prepare_denoising_context(rgb.float(), events.float(), prompts)
                # 训练维持逐样本 VAE encode 的随机序列；候选 replay 本身不推进 RNG。
                if not preserve_rng:
                    next_context_rng['cpu'] = torch.get_rng_state()
                    if rgb.is_cuda:
                        next_context_rng['cuda'] = torch.cuda.get_rng_state(rgb.device)
                observation = freeze_observation_masks(observe_events(
                    predictor, events_structure.float(), config.min_confidence))
                with decoder_precision():
                    condition = gen.prepare_decoder_refinement_condition(
                        rgb, events_refinement, refinement_fusion_mode)
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(rgb.device) if rgb.is_cuda else None
            current = {}

            def render(correction):
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_state)
                    if rgb.is_cuda:
                        torch.cuda.set_rng_state(cuda_state, rgb.device)
                    latent, _ = gen.predict_denoised_latent_from_context(
                        context, torch.tensor([gen.args.generator_timestep], device=rgb.device),
                        feature_guidance=correction)
                    current['latent'] = latent.detach()
                    with decoder_precision():
                        # raw 输出用于梯度；optimize_features 在相同最终图像的 clamp 域验收。
                        return gen.decode_denoised_latent(latent, rgb,
                            decoder_fusion_enabled=True, refinement_fusion_mode=refinement_fusion_mode,
                            clamp_output=False, refinement_condition=condition)

            def on_accept():
                accepted['latent'] = current['latent']

            image, stats = optimize_features(render, observation, config, operator, on_accept=on_accept)
            stats.update(decoder_path='full_refinement', uses_gt_for_guidance=False,
                         decoder_fusion_levels=getattr(gen.vae.decoder, 'inference_fusion_levels', None),
                         decoder_fusion_weight=gen.decoder_fusion_weight,
                         refinement_autocast=str(refinement_autocast_dtype))
            # 直接返回已经验收的图像；禁止另换 mask/强度/精度重新解码后冒充同一结果。
            return image.detach(), accepted['latent'], stats
    finally:
        if next_context_rng:
            torch.set_rng_state(next_context_rng['cpu'])
            if rgb.is_cuda:
                torch.cuda.set_rng_state(next_context_rng['cuda'], rgb.device)
        clear_event_feats_for_all_event_lora(gen.unet)
        for _, layer in iter_event_lora_modules(gen.unet):
            layer.last_aux = {}; layer.last_visuals = {}; layer.last_z_delta = None
        for parameter, enabled in flags:
            parameter.requires_grad_(enabled)
        for module, mode in modes:
            module.training = mode
        for module, enabled in checkpointing:
            module.gradient_checkpointing = enabled


class H1DecoderRefinementGen(OSEDiffEventFiLMGen):
    """与 Stage4 DDP 接口一致；只开放 feature_fusion_blocks 的外层梯度。"""
    supports_h1_decoder_refinement = True

    def __init__(self, args):
        super().__init__(args)
        head, meta, operator = load_structure_checkpoint(args.h1_structure_checkpoint, require_calibrated=True)
        if meta.get('image_size') != args.image_size or args.image_size != 256:
            raise ValueError('H1 checkpoint/image-size mismatch.')
        self.h1_predictor = head
        self.h1_metadata = meta
        self.h1_operator = operator
        self.h1_config = GuidanceConfig(steps=args.h1_guidance_steps, learning_rate=args.h1_guidance_lr,
            trust_radius=args.h1_guidance_trust_radius, min_confidence=args.h1_min_confidence)
        self.h1_pipeline = dict(version=1, mode='frozen_h1_then_decoder_refinement',
            h1_sha256=file_sha256(args.h1_structure_checkpoint),
            stage3_sha256=file_sha256(args.event_lora_checkpoint),
            guidance=asdict(self.h1_config), event_config=meta['event_config'],
            clora_event_active_bins=args.clora_event_active_bins,
            rgb_event_active_bins=args.rgb_event_active_bins,
            ordinary_unet_lora_parts=list(args.unet_lora_active_parts),
            image_size=args.image_size, generator_timestep=args.generator_timestep,
            decoder_refinement_variant=args.decoder_refinement_variant)
        self.h1_decoder_path = getattr(args, 'h1_decoder_path', 'legacy_stage3')
        if self.h1_decoder_path == 'full_refinement':
            # v2 区分新的内层目标，禁止把旧优化器/训练语义静默当作续训恢复。
            self.h1_pipeline.update(version=2, mode='frozen_full_refinement_h1_then_decoder_training',
                decoder_path='full_refinement', refinement_precision=args.mixed_precision,
                decoder_fusion_weight=args.decoder_fusion_weight,
                refinement_fusion_mode=args.refinement_fusion_mode, decoder_levels=[0,1,2,3])
        self.last_h1_stats = {}

    def load_event_checkpoint(self, checkpoint, *args, **kwargs):
        saved = checkpoint.get('additional_info', {}).get('h1_pipeline')
        if saved is not None and saved != self.h1_pipeline:
            raise ValueError('H1 Stage4 resume must preserve head, Stage3, bins, guidance and refinement settings.')
        return super().load_event_checkpoint(checkpoint, *args, **kwargs)

    def forward(self, c_t, events, timesteps, batch=None, events_refinement=None,
                decoder_fusion_enabled=None, refinement_fusion_mode=None):
        if batch is None or 'events_structure' not in batch:
            raise ValueError('H1 Stage4 requires aligned structure events.')
        if not torch.all(timesteps == self.args.generator_timestep):
            raise ValueError('H1 Stage4 uses the fixed generator timestep.')
        if self.h1_decoder_path == 'full_refinement':
            if decoder_fusion_enabled is False or events_refinement is None:
                raise ValueError('Full-refinement H1 training needs enabled fusion and aligned events.')
            if getattr(self.vae.decoder, 'inference_fusion_levels', None) not in (None, (0,1,2,3)):
                raise ValueError('Full-refinement training must use all four decoder layers.')
            precision = {'no': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[self.args.mixed_precision]
            values, records = [], []
            for index in range(c_t.shape[0]):
                # 内层冻结包括当前 fusion 的全部权重；恢复开关后，外层只对 fusion 求导。
                _, value, stats = refinement_guided_sample(self, c_t[index:index+1], events[index:index+1],
                    batch['events_structure'][index:index+1],
                    dict(prompt=[batch['prompt'][index]], neg_prompt=[batch['neg_prompt'][index]]),
                    self.h1_predictor, self.h1_operator, self.h1_config,
                    events_refinement=events_refinement[index:index+1],
                    refinement_fusion_mode=refinement_fusion_mode or self.refinement_fusion_mode,
                    refinement_autocast_dtype=precision, preserve_rng=False)
                values.append(value); records.append(stats)
            latent = torch.cat(values).detach()
        else:
            latent, records = guided_latents(self, c_t, events, batch['events_structure'],
                batch, self.h1_predictor, self.h1_operator, self.h1_config)
        self.last_h1_stats = {
            'h1/accepted_fraction': sum(r['accepted_steps'] > 0 for r in records) / len(records),
            'h1/mean_forward_calls': sum(r['forward_calls'] for r in records) / len(records),
            'h1/mean_relative_delta': sum(r.get('relative_delta', 0.) for r in records) / len(records),
            'h1/mean_confidence': sum(r.get('mean_confidence', 0.) for r in records) / len(records),
        }
        prediction = self.decode_denoised_latent(latent, c_t, events_refinement=events_refinement,
            decoder_fusion_enabled=decoder_fusion_enabled, refinement_fusion_mode=refinement_fusion_mode)
        return prediction, latent, None, None

    def debug_stats(self):
        return dict(super().debug_stats(), **self.last_h1_stats)

    def save_model(self, outf, **kwargs):
        info = dict(kwargs.pop('additional_info', {}) or {})
        info['h1_pipeline'] = self.h1_pipeline
        # 小型结构权重随 Stage4 保存；不包含 H1 的优化器，外层优化器仅属于 fusion。
        keep = ('format', 'head_config', 'operator_config', 'event_config', 'state_dict',
                'confidence_trained', 'image_size', 'confidence_tau', 'phase', 'scene_split')
        info['h1_checkpoint'] = {k: self.h1_metadata[k] for k in keep if k in self.h1_metadata}
        return super().save_model(outf, additional_info=info, **kwargs)
