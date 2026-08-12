import os

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel

from osediff import initialize_unet, initialize_vae
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from safetensors.torch import load_file
import json

from models.event_encoder import EventEncoder
from models.event_film_lora import EventLoRAConfig, normalize_event_lora_variant
from models.rgb_event_encoder import RGBEventEncoder
from models.event_lora_utils import (
    clear_event_feats_for_all_event_lora,
    collect_event_lora_debug,
    copy_compatible_state_dict,
    event_lora_state_keys,
    remap_legacy_event_film_state_dict,
    replace_self_attn_to_v_with_event_lora,
    set_event_feats_for_all_event_lora,
    set_event_lora_trainable,
    set_modulation_enabled_for_all_event_lora,
)


def _event_lora_config_from_args(args):
    return EventLoRAConfig(
        variant=getattr(args, "event_lora_variant", "film_affine"),
        rank=args.lora_rank,
        alpha=float(args.lora_rank),
        mod_channels=getattr(args, "mod_channels", 32),
        modulation_scale=getattr(args, "modulation_scale", 0.1),
        cbam_reduction=getattr(args, "event_lora_cbam_reduction", 8),
        cbam_spatial_kernel=getattr(args, "event_lora_cbam_spatial_kernel", 7),
        output_gate_channels=getattr(args, "event_lora_output_gate_channels", "spatial"),
        mod_num_heads=getattr(args, "mod_num_heads", 4),
        mod_window_size=getattr(args, "mod_window_size", 8),
    )


def _checkpoint_event_lora_config(checkpoint):
    config = checkpoint.get("event_lora_config")
    if config is None:
        # 旧 idea2 checkpoint 只有 Event-FiLM，因此缺少配置时按 film_affine 解释。
        return EventLoRAConfig(variant="film_affine")
    config = dict(config)
    legacy_variant = checkpoint.get("event_lora_variant")
    config.setdefault("variant", normalize_event_lora_variant(legacy_variant or "film_affine"))
    rank = int(checkpoint.get("rank_unet", config.get("rank", 4)))
    config.setdefault("rank", rank)
    config.setdefault("alpha", float(rank))
    # idea2-1 的 residual checkpoint 只记录以下三个 cross-attention 字段。
    config.setdefault("mod_channels", 64 if legacy_variant in {
        "residual_cross_attn", "residual_cross_attn_bax"
    } else 32)
    return EventLoRAConfig(**config)


def _validate_event_lora_config(runtime_config, checkpoint_config, context):
    structural_keys = (
        "variant",
        "rank",
        "alpha",
        "mod_channels",
        "modulation_scale",
        "cbam_reduction",
        "cbam_spatial_kernel",
        "output_gate_channels",
        "mod_num_heads",
        "mod_window_size",
    )
    mismatch = {
        key: (getattr(runtime_config, key), getattr(checkpoint_config, key))
        for key in structural_keys
        if getattr(runtime_config, key) != getattr(checkpoint_config, key)
    }
    if mismatch:
        raise ValueError(
            f"{context} Event-LoRA config mismatch (runtime, checkpoint): {mismatch}"
        )


def _normalize_prompt_batch(prompt_batch):
    """将 DataLoader 对字符串/单元素列表的多种 collate 结果统一为字符串列表。"""
    if prompt_batch is None:
        return []
    if isinstance(prompt_batch, str):
        return [prompt_batch]
    if len(prompt_batch) > 0 and isinstance(prompt_batch[0], (list, tuple)):
        # 默认 collate 会把每个样本的 ``[prompt]`` 转置为 ``[(p1, p2, ...)]``。
        # 这里必须还原整个 batch，不能只取 tuple 的第一个元素，否则文本 batch
        # 与图像 batch 不一致，会在 UNet cross-attention 中触发序列维度错误。
        if len(prompt_batch) == 1 and all(
            isinstance(item, str) for item in prompt_batch[0]
        ):
            return list(prompt_batch[0])
        return [p[0] for p in prompt_batch]
    return list(prompt_batch)


def _load_compatible_named_parameters(module, checkpoint_state):
    loaded, skipped = 0, 0
    named = dict(module.named_parameters())
    for name, tensor in checkpoint_state.items():
        if name in named and named[name].shape == tensor.shape:
            named[name].data.copy_(tensor.to(dtype=named[name].dtype))
            loaded += 1
        else:
            skipped += 1
    return loaded, skipped


def _first_parameter_dtype(module, default=torch.float32):
    for param in module.parameters(recurse=True):
        return param.dtype
    return default


def _unet_cross_attention_dtype(unet, default=torch.float32):
    """
    推理/训练时 cross-attention 的 text embedding dtype 必须匹配 attn2.to_k/to_v。

    PEFT LoRA wrapper 在不同版本中可能让部分 base_layer 保持 float32，即使外层
    UNet 已经 `.to(dtype=torch.float16)`。动态读取实际 to_k 权重 dtype，比硬编码
    fp16/fp32 更稳，能避免 F.linear(input, weight) 的 Half/Float mismatch。
    """
    for name, module in unet.named_modules():
        if name.endswith("attn2.to_k"):
            return _first_parameter_dtype(module, default=default)
    return default


class OSEDiffEventFiLMGen(torch.nn.Module):
    """
    Event-FiLM LoRA Generator。

    作用：
    - 复用原 OSEDiff 的 VAE、Text Encoder、UNet 与 scheduler；
    - 保留完整 Stage1 可兼容权重；
    - 将 self-attention 的 attn1.to_v 替换成事件调制 LoRA。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        self.vae, self.lora_vae_modules_encoder = initialize_vae(args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(args)
        self.event_encoder = EventEncoder(
            in_channels=getattr(args, "event_channels", 5),
            base_channels=getattr(args, "event_base_channels", 32),
        )
        self.event_lora_config = _event_lora_config_from_args(args)
        self.decoder_refinement_enabled = bool(
            getattr(args, "decoder_refinement_enabled", False)
        )
        self.decoder_fusion_enabled = bool(
            getattr(args, "decoder_fusion_enabled", self.decoder_refinement_enabled)
        )
        self.decoder_fusion_weight = float(
            getattr(args, "decoder_fusion_weight", 1.0)
        )
        self.refinement_fusion_mode = getattr(
            args, "refinement_fusion_mode", "both"
        )
        self.last_refinement_stats = {}
        self.rgb_event_encoder = None
        if self.decoder_refinement_enabled:
            self.install_decoder_refinement()
        self.event_film_installed = False
        self.event_film_replaced_names = []
        self.lora_rank_unet = args.lora_rank
        self.lora_rank_vae = args.lora_rank

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.event_encoder.to("cuda")
        if self.rgb_event_encoder is not None:
            self.rgb_event_encoder.to("cuda")
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

    def install_decoder_refinement(self):
        """安装单一 FeatureAwareDecoder，并按需加载冻结 RGB-event encoder。"""
        self.vae.install_feature_aware_decoder(
            condition_channels=(64, 128, 256, 512),
            fusion_weight=self.decoder_fusion_weight,
            fusion_init=getattr(self.args, "decoder_fusion_init", "near_zero"),
        )
        if self.rgb_event_encoder is None:
            self.rgb_event_encoder = RGBEventEncoder(
                event_channels=getattr(self.args, "event_channels", 5)
            )
        pretrained_root = getattr(self.args, "rgb_event_pretrained_root", None)
        encoder_i_path = getattr(self.args, "rgb_event_encoder_i_path", None)
        encoder_e_path = getattr(self.args, "rgb_event_encoder_e_path", None)
        fusion_path = getattr(self.args, "rgb_event_fusion_path", None)
        if pretrained_root or encoder_i_path or encoder_e_path or fusion_path:
            self.rgb_event_encoder.load_pretrained(
                pretrained_root=pretrained_root,
                encoder_i_path=encoder_i_path,
                encoder_e_path=encoder_e_path,
                fusion_path=fusion_path,
                strict=bool(getattr(self.args, "rgb_event_strict_load", True)),
            )
        vae_parameter = next(self.vae.parameters())
        self.rgb_event_encoder.to(
            device=vae_parameter.device, dtype=vae_parameter.dtype
        )
        self.rgb_event_encoder.requires_grad_(False)
        self.decoder_refinement_enabled = True
        return self.vae.decoder

    def set_train_decoder_refinement(self):
        """Phase 1B 显式冻结全模型，只解冻 decoder feature fusion blocks。"""
        if not self.decoder_refinement_enabled or self.rgb_event_encoder is None:
            raise RuntimeError("Decoder refinement has not been installed.")
        self.requires_grad_(False)
        self.vae.decoder.feature_fusion_blocks.requires_grad_(True)
        self.rgb_event_encoder.eval()
        self.event_encoder.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.vae.train()
        trainable = list(self.vae.decoder.feature_fusion_blocks.parameters())
        trainable_ids = {id(param) for param in trainable}
        stray = [
            name for name, param in self.named_parameters()
            if param.requires_grad and id(param) not in trainable_ids
        ]
        if stray:
            raise RuntimeError(f"Unexpected trainable parameters: {stray}")
        return trainable

    def install_event_lora(self):
        if self.event_film_installed:
            return 0, self.event_film_replaced_names
        num_replaced, names = replace_self_attn_to_v_with_event_lora(
            self.unet,
            config=self.event_lora_config,
            event_channels_by_size=self.event_encoder.out_channels_by_size,
        )
        self.event_film_installed = True
        self.event_film_replaced_names = names
        print(
            f"[Event-LoRA:{self.event_lora_config.variant}] "
            f"Replaced {num_replaced} self-attn to_v layer(s)."
        )
        for name in names[:20]:
            print(f"[Event-LoRA]   {name}")
        return num_replaced, names

    # 兼容旧代码入口。
    install_event_film_lora = install_event_lora

    def load_stage1_full(self, checkpoint_path):
        """
        完整加载 Stage1 中可兼容权重，然后迁移 attn1.to_v LoRA 到 Event-FiLM 模块。

        加载顺序很关键：
        1. 先把 Stage1 的 PEFT LoRA/conv_in/VAE LoRA 加到原始结构；
        2. 再替换 attn1.to_v，此时替换工具会复制已经加载好的 LoRA A/B。
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        unet_loaded, unet_skipped = _load_compatible_named_parameters(
            self.unet, checkpoint.get("state_dict_unet", {})
        )
        vae_loaded, vae_skipped = _load_compatible_named_parameters(
            self.vae, checkpoint.get("state_dict_vae", {})
        )
        num_replaced, _ = self.install_event_lora()
        print(
            "[Stage1 Load] "
            f"unet_loaded={unet_loaded}, unet_skipped={unet_skipped}, "
            f"vae_loaded={vae_loaded}, vae_skipped={vae_skipped}, "
            f"event_film_initialized={num_replaced}"
        )
        return checkpoint

    def set_train_stage(self, stage):
        self.unet.train()
        self.vae.train()
        self.event_encoder.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.event_encoder.requires_grad_(stage in {2, 3, "2", "3"})
        if stage in {2, "2"}:
            set_event_lora_trainable(self.unet, train_lora=False, train_modulation=True)
            set_modulation_enabled_for_all_event_lora(self.unet, True)
        elif stage in {3, "3"}:
            set_event_lora_trainable(self.unet, train_lora=True, train_modulation=True)
            set_modulation_enabled_for_all_event_lora(self.unet, True)
        else:
            self.event_encoder.requires_grad_(False)
            set_event_lora_trainable(self.unet, train_lora=True, train_modulation=False)
            set_modulation_enabled_for_all_event_lora(self.unet, False)

    def get_lora_params(self):
        params = []
        for name, param in self.unet.named_parameters():
            if param.requires_grad and ("lora_down" in name or "lora_up" in name):
                params.append(param)
        return params

    def get_event_params(self):
        # residual variant 的共享 head 注册在 UNet 顶层，名称不含 modulator；
        # 因此以 trainable 且非 LoRA A/B 为准，并按参数 id 去重。
        params = [p for p in self.event_encoder.parameters() if p.requires_grad]
        seen = {id(param) for param in params}
        for name, param in self.unet.named_parameters():
            if not param.requires_grad or "lora_down" in name or "lora_up" in name:
                continue
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
        return params

    def encode_prompt(self, prompt_batch):
        prompt_batch = _normalize_prompt_batch(prompt_batch)
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                prompt_embeds = self.text_encoder(text_input_ids.to(self.text_encoder.device))[0]
                prompt_embeds_list.append(prompt_embeds)
        return torch.cat(prompt_embeds_list, dim=0)

    def predict_denoised_latent(self, c_t, events, timesteps, batch):
        """运行一次 VAE encode + event-cLoRA UNet，返回可复用的 z0。"""
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        cross_attn_dtype = _unet_cross_attention_dtype(self.unet, default=encoded_control.dtype)

        # 事件特征只在 UNet forward 期间放入 LoRA 缓存，避免跨 batch 残留错误条件。
        event_feats = self.event_encoder(events)
        set_event_feats_for_all_event_lora(self.unet, event_feats)
        try:
            model_pred = self.unet(
                encoded_control,
                timesteps,
                encoder_hidden_states=prompt_embeds.to(dtype=cross_attn_dtype),
            ).sample
        finally:
            clear_event_feats_for_all_event_lora(self.unet)

        x_denoised = self.noise_scheduler.step(
            model_pred, timesteps, encoded_control, return_dict=True
        ).prev_sample
        return x_denoised, prompt_embeds, neg_prompt_embeds

    def decode_denoised_latent(
        self,
        x_denoised,
        c_t,
        events_refinement=None,
        decoder_fusion_enabled=None,
        refinement_fusion_mode=None,
    ):
        """对固定 z0 只执行一次 decoder，供 off/on 公平对比复用。"""
        vae_dtype = _first_parameter_dtype(self.vae, default=x_denoised.dtype)
        fusion_enabled = (
            self.decoder_fusion_enabled
            if decoder_fusion_enabled is None else bool(decoder_fusion_enabled)
        )
        fused_features = None
        if self.decoder_refinement_enabled:
            if events_refinement is None:
                raise ValueError(
                    "Decoder refinement requires the separately normalized "
                    "events_refinement tensor."
                )
            fusion_mode = refinement_fusion_mode or self.refinement_fusion_mode
            with torch.no_grad():
                encoder_output = self.rgb_event_encoder(
                    c_t,
                    events_refinement,
                    fusion_mode=fusion_mode,
                )
            fused_features = [
                feature.detach() for feature in encoder_output.fused_features
            ]
            self.last_refinement_stats = {}
            for family, features in (
                ("rgb", encoder_output.rgb_features),
                ("event", encoder_output.event_features),
                ("fused", encoder_output.fused_features),
            ):
                for level, feature in enumerate(features):
                    detached = feature.detach().float()
                    prefix = f"{family}_level_{level}"
                    self.last_refinement_stats[f"{prefix}/mean"] = detached.mean().item()
                    self.last_refinement_stats[f"{prefix}/std"] = detached.std(unbiased=False).item()
                    self.last_refinement_stats[f"{prefix}/abs_max"] = detached.abs().max().item()
        output = (
            self.vae.decode(
                x_denoised.to(dtype=vae_dtype) / self.vae.config.scaling_factor,
                fused_features=fused_features,
                fusion_enabled=fusion_enabled,
                fusion_weight=self.decoder_fusion_weight,
            ).sample
        ).clamp(-1, 1)
        detached_output = output.detach().float()
        self.last_refinement_stats.update({
            "output/min": detached_output.min().item(),
            "output/max": detached_output.max().item(),
            "output/mean": detached_output.mean().item(),
            "output/std": detached_output.std(unbiased=False).item(),
        })
        return output

    def forward(
        self,
        c_t,
        events,
        timesteps,
        batch=None,
        events_refinement=None,
        decoder_fusion_enabled=None,
        refinement_fusion_mode=None,
    ):
        x_denoised, prompt_embeds, neg_prompt_embeds = self.predict_denoised_latent(
            c_t, events, timesteps, batch
        )
        output_image = self.decode_denoised_latent(
            x_denoised,
            c_t,
            events_refinement=events_refinement,
            decoder_fusion_enabled=decoder_fusion_enabled,
            refinement_fusion_mode=refinement_fusion_mode,
        )
        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def debug_stats(self):
        stats = collect_event_lora_debug(self.unet)
        if self.decoder_refinement_enabled:
            stats.update({
                f"decoder_refinement/{key}": value
                for key, value in self.last_refinement_stats.items()
            })
            stats.update({
                f"decoder_refinement/{key}": value
                for key, value in self.vae.decoder.debug_stats().items()
            })
        return stats

    def save_model(
        self,
        outf,
        optimizer=None,
        lr_scheduler=None,
        optimizer_reg=None,
        lr_scheduler_reg=None,
        scaler=None,
        global_step=0,
        epoch=0,
        stage=2,
        timestep_config=None,
        additional_info=None,
    ):
        state_unet = {}
        keep_markers = ("lora", "conv_in") + event_lora_state_keys()
        for key, value in self.unet.state_dict().items():
            if any(marker in key for marker in keep_markers):
                state_unet[key] = value
        sd = {
            "vae_lora_encoder_modules": self.lora_vae_modules_encoder,
            "unet_lora_encoder_modules": self.lora_unet_modules_encoder,
            "unet_lora_decoder_modules": self.lora_unet_modules_decoder,
            "unet_lora_others_modules": self.lora_unet_others,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": state_unet,
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k},
            "state_dict_event_encoder": self.event_encoder.state_dict(),
            "event_lora_config": self.event_lora_config.to_dict(),
            "event_lora_variant": self.event_lora_config.variant,
            "event_film_replaced_names": self.event_film_replaced_names,
            "stage": stage,
            "timestep_config": timestep_config or {},
            "global_step": global_step,
            "epoch": epoch,
        }
        if optimizer is not None:
            sd["optimizer_state_dict"] = optimizer.state_dict()
        if lr_scheduler is not None:
            sd["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        if optimizer_reg is not None:
            sd["optimizer_reg_state_dict"] = optimizer_reg.state_dict()
        if lr_scheduler_reg is not None:
            sd["lr_scheduler_reg_state_dict"] = lr_scheduler_reg.state_dict()
        if scaler is not None:
            sd["scaler_state_dict"] = scaler.state_dict()
        if additional_info is not None:
            sd["additional_info"] = additional_info
        if self.decoder_refinement_enabled:
            sd.update({
                "checkpoint_version": "idea3_phase1_v1",
                "state_dict_decoder_refinement": self.vae.decoder.feature_fusion_blocks.state_dict(),
                "decoder_refinement_config": self.vae.decoder.refinement_config(),
                "rgb_event_encoder_config": self.rgb_event_encoder.config.to_dict(),
                "rgb_event_pretrained_manifest": self.rgb_event_encoder.pretrained_manifest,
                # idea3 显式字段；保留旧字段以兼容现有 Stage 2/3 loader。
                "state_dict_unet_event_lora": state_unet,
                "state_dict_existing_event_encoder": self.event_encoder.state_dict(),
            })
            if bool(getattr(self.args, "bundle_rgb_event_encoder", False)):
                sd["state_dict_rgb_event_encoder"] = self.rgb_event_encoder.state_dict()
        torch.save(sd, outf)

    def load_decoder_refinement_checkpoint(self, checkpoint, ignore_mismatch=False):
        if "state_dict_decoder_refinement" not in checkpoint:
            return 0, 0
        if not self.decoder_refinement_enabled:
            self.install_decoder_refinement()
        runtime_config = self.vae.decoder.refinement_config()
        checkpoint_config = checkpoint.get("decoder_refinement_config", {})
        structural_keys = (
            "condition_channels",
            "decoder_channels",
            "fusion_blocks",
            "fusion_init",
        )
        mismatch = {
            key: (runtime_config.get(key), checkpoint_config.get(key))
            for key in structural_keys
            if checkpoint_config.get(key) is not None
            and runtime_config.get(key) != checkpoint_config.get(key)
        }
        if mismatch and not ignore_mismatch:
            raise ValueError(f"Decoder refinement config mismatch: {mismatch}")
        refinement_state = checkpoint["state_dict_decoder_refinement"]
        target_state = self.vae.decoder.feature_fusion_blocks.state_dict()
        missing_keys = sorted(set(target_state) - set(refinement_state))
        loaded, skipped = copy_compatible_state_dict(
            self.vae.decoder.feature_fusion_blocks,
            refinement_state,
        )
        if (missing_keys or skipped) and not ignore_mismatch:
            raise RuntimeError(
                "Decoder refinement state mismatch: "
                f"missing={missing_keys}, skipped={skipped}"
            )
        bundled = checkpoint.get("state_dict_rgb_event_encoder")
        if bundled is not None:
            rgb_loaded, rgb_skipped = copy_compatible_state_dict(
                self.rgb_event_encoder, bundled
            )
            if rgb_skipped and not ignore_mismatch:
                raise RuntimeError(
                    f"Bundled RGBEventEncoder load skipped {rgb_skipped} tensors."
                )
            print(
                f"[Idea3 Load] bundled_rgb_event_encoder loaded={rgb_loaded}, "
                f"skipped={rgb_skipped}"
            )
        elif self.rgb_event_encoder.pretrained_manifest is None:
            raise RuntimeError(
                "Checkpoint does not bundle RGBEventEncoder; provide "
                "--rgb_event_pretrained_root (or three explicit weight paths)."
            )
        expected_manifest = checkpoint.get("rgb_event_pretrained_manifest")
        actual_manifest = self.rgb_event_encoder.pretrained_manifest
        if (
            expected_manifest
            and actual_manifest
            and expected_manifest.get("sha256") != actual_manifest.get("sha256")
            and not ignore_mismatch
        ):
            raise ValueError("RGB-event pretrained manifest hash mismatch.")
        print(f"[Idea3 Load] decoder_refinement loaded={loaded}, skipped={skipped}")
        return loaded, skipped

    def load_event_checkpoint(self, checkpoint, ignore_config_mismatch=False):
        checkpoint_config = _checkpoint_event_lora_config(checkpoint)
        if ignore_config_mismatch:
            print("[Event-LoRA Load] config mismatch checks are explicitly disabled.")
        else:
            _validate_event_lora_config(
                self.event_lora_config, checkpoint_config, context="Generator"
            )
        self.install_event_lora()
        unet_state = remap_legacy_event_film_state_dict(
            checkpoint.get(
                "state_dict_unet",
                checkpoint.get("state_dict_unet_event_lora", {}),
            )
        )
        unet_loaded, unet_skipped = copy_compatible_state_dict(
            self.unet, unet_state
        )
        vae_loaded, vae_skipped = copy_compatible_state_dict(
            self.vae, checkpoint.get("state_dict_vae", {})
        )
        enc_loaded, enc_skipped = copy_compatible_state_dict(
            self.event_encoder,
            checkpoint.get(
                "state_dict_event_encoder",
                checkpoint.get("state_dict_existing_event_encoder", {}),
            ),
        )
        print(
            "[Event-FiLM Load] "
            f"unet_loaded={unet_loaded}, unet_skipped={unet_skipped}, "
            f"vae_loaded={vae_loaded}, vae_skipped={vae_skipped}, "
            f"event_encoder_loaded={enc_loaded}, event_encoder_skipped={enc_skipped}"
        )
        if "state_dict_decoder_refinement" in checkpoint:
            self.load_decoder_refinement_checkpoint(
                checkpoint,
                ignore_mismatch=ignore_config_mismatch,
            )


class OSEDiffEventFiLMReg(torch.nn.Module):
    """
    fine-tuned regularizer。

    作用：
    沿用 OSEDiff 的 VSD/diff loss 路径，但 timestep 由训练 loop 传入，
    保证每个 step 中 Generator 和 regularizer 使用同一个 timestep 值。
    """

    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        vae_dir = os.path.join(args.pretrained_model_name_or_path, "vae")
        with open(os.path.join(vae_dir, "config.json"), "r") as f:
            vae_config = json.load(f)
        self.vae = AutoencoderKL(
            in_channels=vae_config.get("in_channels", 3),
            out_channels=vae_config.get("out_channels", 3),
            down_block_types=tuple(vae_config["down_block_types"]),
            up_block_types=tuple(vae_config["up_block_types"]),
            block_out_channels=tuple(vae_config["block_out_channels"]),
            layers_per_block=vae_config.get("layers_per_block", 2),
            act_fn=vae_config.get("act_fn", "silu"),
            latent_channels=vae_config.get("latent_channels", 4),
            norm_num_groups=vae_config.get("norm_num_groups", 32),
            sample_size=vae_config.get("sample_size", 768),
        )
        self.vae.load_state_dict(load_file(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")))

        unet_dir = os.path.join(args.pretrained_model_name_or_path, "unet")
        with open(os.path.join(unet_dir, "config.json"), "r") as f:
            unet_config = json.load(f)
        self.unet_fix = UNet2DConditionModel(
            in_channels=unet_config.get("in_channels", 4),
            out_channels=unet_config.get("out_channels", 4),
            down_block_types=tuple(unet_config["down_block_types"]),
            up_block_types=tuple(unet_config["up_block_types"]),
            block_out_channels=tuple(unet_config["block_out_channels"]),
            layers_per_block=unet_config.get("layers_per_block", 2),
            cross_attention_dim=unet_config.get("cross_attention_dim", 1024),
            attention_head_dim=unet_config.get("attention_head_dim"),
            sample_size=unet_config.get("sample_size", 64),
            use_linear_projection=unet_config.get("use_linear_projection", False),
        )
        self.unet_fix.load_state_dict(load_file(os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")))
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(args)
        self.regularizer_lora_mode = getattr(
            args, "regularizer_lora_mode", "same_as_generator"
        )
        if self.regularizer_lora_mode == "event_film":
            print(
                "[DEPRECATED] regularizer_lora_mode=event_film is mapped to "
                "same_as_generator."
            )
            self.regularizer_lora_mode = "same_as_generator"
        self._shared_event_encoder_ref = [None]
        self.event_lora_config = None
        self.event_film_installed = False
        self.event_film_replaced_names = []
        self.lora_rank_unet = args.lora_rank

        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_update.to(accelerator.device)
        self.vae.to(accelerator.device)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_shared_event_encoder(self, event_encoder, event_lora_config):
        # 使用普通 list 保存引用，避免把共享 EventEncoder 注册成 regularizer 的子模块。
        self._shared_event_encoder_ref[0] = event_encoder
        self.event_lora_config = event_lora_config

    def get_shared_event_encoder(self):
        event_encoder = self._shared_event_encoder_ref[0]
        if event_encoder is None:
            raise RuntimeError(
                "regularizer_lora_mode=same_as_generator requires "
                "sharing Generator.event_encoder."
            )
        return event_encoder

    def install_event_lora(self):
        if self.regularizer_lora_mode != "same_as_generator":
            return 0, self.event_film_replaced_names
        if self.event_film_installed:
            return 0, self.event_film_replaced_names
        if self.event_lora_config is None:
            raise RuntimeError("Generator EventLoRAConfig has not been shared.")
        event_encoder = self.get_shared_event_encoder()
        num_replaced, names = replace_self_attn_to_v_with_event_lora(
            self.unet_update,
            config=self.event_lora_config,
            event_channels_by_size=event_encoder.out_channels_by_size,
        )
        self.event_film_installed = True
        self.event_film_replaced_names = names
        print(
            f"[Reg Event-LoRA:{self.event_lora_config.variant}] "
            f"Replaced {num_replaced} self-attn to_v layer(s)."
        )
        for name in names[:20]:
            print(f"[Reg Event-LoRA]   {name}")
        return num_replaced, names

    install_event_film_lora = install_event_lora

    def set_train(self):
        self.unet_update.train()
        self.unet_update.requires_grad_(False)
        for name, param in self.unet_update.named_parameters():
            param.requires_grad = "lora" in name
        if self.regularizer_lora_mode == "same_as_generator":
            self.install_event_lora()
            # 只共享 EventEncoder；regularizer 的 LoRA A/B 与 CBAM/FiLM head 均独立训练。
            set_event_lora_trainable(self.unet_update, train_lora=True, train_modulation=True)
            set_modulation_enabled_for_all_event_lora(self.unet_update, True)

    def load_ckpt(self, checkpoint, strict_config=True):
        state = checkpoint.get("state_dict_reg", checkpoint.get("state_dict_unet", {}))
        checkpoint_mode = checkpoint.get("regularizer_lora_mode")
        if checkpoint_mode == "event_film":
            checkpoint_mode = "same_as_generator"
        if (
            strict_config
            and checkpoint_mode is not None
            and checkpoint_mode != self.regularizer_lora_mode
        ):
            raise ValueError(
                "regularizer_lora_mode mismatch: "
                f"runtime={self.regularizer_lora_mode}, checkpoint={checkpoint_mode}"
            )
        if (
            not strict_config
            and checkpoint_mode is not None
            and checkpoint_mode != self.regularizer_lora_mode
        ):
            print(
                "[Reg Init] mode changed across stages: "
                f"{checkpoint_mode} -> {self.regularizer_lora_mode}; "
                "only compatible LoRA weights will be initialized."
            )
        checkpoint_config_dict = checkpoint.get("regularizer_event_lora_config")
        if (
            strict_config
            and checkpoint_config_dict is not None
            and self.event_lora_config is not None
        ):
            _validate_event_lora_config(
                self.event_lora_config,
                EventLoRAConfig(**checkpoint_config_dict),
                context="Regularizer",
            )
        state = remap_legacy_event_film_state_dict(state)
        has_event_film_state = any(
            any(marker in key for marker in event_lora_state_keys())
            for key in state
        )
        if self.regularizer_lora_mode == "same_as_generator" and has_event_film_state:
            self.install_event_lora()
            loaded, skipped = copy_compatible_state_dict(self.unet_update, state)
        elif self.regularizer_lora_mode == "same_as_generator":
            # 普通 LoRA checkpoint：先加载 unet_update，再按 Generator variant 替换。
            loaded, skipped = _load_compatible_named_parameters(self.unet_update, state)
            self.install_event_lora()
        else:
            loaded, skipped = _load_compatible_named_parameters(self.unet_update, state)

        print(
            "[Reg Load] "
            f"mode={self.regularizer_lora_mode}, "
            f"unet_loaded={loaded}, unet_skipped={skipped}, "
            f"shared_event_encoder={self.regularizer_lora_mode == 'same_as_generator'}"
        )

    def state_dict_reg(self):
        keep_markers = ("lora",) + event_lora_state_keys()
        return {
            k: v for k, v in self.unet_update.state_dict().items()
            if any(marker in k for marker in keep_markers)
        }

    def checkpoint_state(self):
        state = {
            "state_dict_reg": self.state_dict_reg(),
            "regularizer_lora_mode": self.regularizer_lora_mode,
            "regularizer_event_lora_config": (
                self.event_lora_config.to_dict()
                if self.regularizer_lora_mode == "same_as_generator"
                else None
            ),
            "reg_event_film_replaced_names": self.event_film_replaced_names,
        }
        return state

    def _predict_with_update_unet(self, noisy_latents, timesteps, prompt_embeds, events):
        if self.regularizer_lora_mode != "same_as_generator":
            return self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample
        # 事件特征只在当前 regularizer UNet forward 期间注入，避免跨 batch 残留。
        with torch.no_grad():
            event_feats = self.get_shared_event_encoder()(events)
        event_feats = {key: value.detach() for key, value in event_feats.items()}
        set_event_feats_for_all_event_lora(self.unet_update, event_feats)
        try:
            return self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample
        finally:
            clear_event_feats_for_all_event_lora(self.unet_update)

    def diff_loss(self, latents, prompt_embeds, timesteps, events):
        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        t_batch = timesteps.expand(bsz).to(latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, t_batch)
        noise_pred = self._predict_with_update_unet(noisy_latents, t_batch, prompt_embeds, events)
        return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, timesteps, cfg_vsd, events):
        bsz = latents.shape[0]
        t_batch = timesteps.expand(bsz).to(latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, t_batch)
        with torch.no_grad():
            noise_pred_update = self._predict_with_update_unet(
                noisy_latents,
                t_batch,
                prompt_embeds.float(),
                events,
            )
            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, t_batch)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([t_batch] * 2)
            prompt_embeds_input = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds_input.to(dtype=self.weight_dtype),
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + cfg_vsd * (noise_pred_text - noise_pred_uncond)
            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, t_batch)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True).clamp_min(1e-6)
        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        return F.mse_loss(latents, (latents - grad).detach())


class OSEDiffEventFiLMTest(torch.nn.Module):
    """
    Event-FiLM 推理模型。

    推理只使用 Generator；regularizer 只在训练阶段提供 VSD 约束。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = OSEDiffEventFiLMGen(args)
        checkpoint = torch.load(args.osediff_path, map_location="cpu")
        self.gen.load_event_checkpoint(
            checkpoint,
            ignore_config_mismatch=bool(
                getattr(args, "resume_ignore_mismatch", False)
            ),
        )
        self.gen.eval()
        self.weight_dtype = torch.float16 if getattr(args, "mixed_precision", "fp16") == "fp16" else torch.float32
        self.gen.unet.to(self.device, dtype=self.weight_dtype)
        self.gen.vae.to(self.device, dtype=self.weight_dtype)
        self.gen.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.gen.event_encoder.to(self.device, dtype=self.weight_dtype)
        if self.gen.rgb_event_encoder is not None:
            self.gen.rgb_event_encoder.to(self.device, dtype=self.weight_dtype)
        self.timesteps = torch.tensor([getattr(args, "generator_timestep", 999)], device=self.device).long()
        print(f"[INFO] Generator inference timestep: {self.timesteps.item()}")

    @torch.no_grad()
    def forward(
        self,
        lq,
        events,
        prompt,
        events_refinement=None,
        decoder_fusion_enabled=None,
        refinement_fusion_mode=None,
    ):
        batch = {
            "prompt": [prompt],
            "neg_prompt": [getattr(self.args, "neg_prompt", "")],
        }
        out, _, _, _ = self.gen(
            lq.to(self.device, dtype=self.weight_dtype),
            events.to(self.device, dtype=self.weight_dtype),
            self.timesteps,
            batch=batch,
            events_refinement=(
                events_refinement.to(self.device, dtype=self.weight_dtype)
                if events_refinement is not None else None
            ),
            decoder_fusion_enabled=decoder_fusion_enabled,
            refinement_fusion_mode=refinement_fusion_mode,
        )
        return out

    def debug_stats(self):
        return self.gen.debug_stats()
