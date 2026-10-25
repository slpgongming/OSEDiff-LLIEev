import os
import time
from dataclasses import dataclass

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
from models.event_film_lora import EventLoRAConfig, SpatialRGBEventLoRALinear, normalize_event_lora_variant, parse_event_lora_targets
from models.rgb_event_encoder import RGBEventEncoder
from models.decoder_refinement import (
    normalize_activity_mode,
    normalize_refinement_levels,
    normalize_refinement_variant,
)
from models.event_lora_utils import (
    clear_event_feats_for_all_event_lora,
    collect_event_lora_debug,
    copy_compatible_state_dict,
    event_lora_state_keys,
    iter_event_lora_modules,
    migrate_event_lora_state,
    remap_legacy_event_film_state_dict,
    replace_self_attn_projections_with_event_lora,
    set_event_feats_for_all_event_lora,
    set_event_lora_trainable,
    set_modulation_enabled_for_all_event_lora,
)


UNET_LORA_PARTS = ("down", "mid", "up")
_UNET_LORA_ADAPTERS = {
    "down": "default_encoder",  # down_blocks and conv_in
    "mid": "default_others",   # mid_block and remaining non-down/up modules
    "up": "default_decoder",   # up_blocks and conv_out
}


def parse_unet_lora_parts(value):
    """Parse a comma-separated selection of ordinary UNet PEFT LoRA groups."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.strip().lower()
        parts = [] if value in {"", "none"} else value.split(",")
    else:
        parts = list(value)
    parts = [part.strip().lower() for part in parts]
    invalid = set(parts) - set(UNET_LORA_PARTS)
    if invalid or len(parts) != len(set(parts)):
        raise ValueError(
            f"Invalid UNet LoRA parts {value!r}; use none or a comma-separated "
            "selection of down,mid,up without duplicates."
        )
    return tuple(part for part in UNET_LORA_PARTS if part in parts)


def validate_unet_lora_training_parts(active, trainable):
    active = parse_unet_lora_parts(active)
    trainable = parse_unet_lora_parts(trainable)
    if not set(trainable).issubset(active):
        raise ValueError(
            "Trainable ordinary UNet LoRA parts must also be active; "
            f"active={active}, trainable={trainable}."
        )
    return active, trainable


def _is_unet_lora_parameter_name(name):
    return any(marker in name for marker in ("lora_A.", "lora_B.", "lora_down", "lora_up"))


def _event_lora_config_from_args(args):
    targets = parse_event_lora_targets(getattr(args, "event_lora_targets", "auto"), allow_auto=True)
    if targets == "auto":
        source = next((getattr(args, key, None) for key in (
            "osediff_path", "event_lora_checkpoint", "resume_from_checkpoint",
            "init_from_event_checkpoint",
        ) if getattr(args, key, None) not in (None, "latest")), None)
        if source:
            try:
                checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(source, map_location="cpu")
            targets = _checkpoint_event_lora_config(checkpoint).targets
        else:
            targets = ("v",)
    return EventLoRAConfig(
        variant=getattr(args, "event_lora_variant", "film_affine"),
        targets=targets,
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
    metadata = checkpoint.get("event_lora_metadata") or {}
    if "targets" in metadata and "targets" in config and tuple(metadata["targets"]) != tuple(config["targets"]):
        raise ValueError("Checkpoint Event-LoRA target metadata contradicts its config.")
    config.setdefault("targets", metadata.get("targets", ("v",)))
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


def _event_checkpoint_state(checkpoint):
    return remap_legacy_event_film_state_dict(checkpoint.get(
        "state_dict_unet", checkpoint.get("state_dict_unet_event_lora", {})
    ))


def _validate_checkpoint_target_evidence(checkpoint, config, state):
    suffixes = {"q": "attn1.to_q", "k": "attn1.to_k",
                "v": "attn1.to_v", "out": "attn1.to_out.0"}
    actual = set()
    for key in state:
        if not key.endswith(".lora_down.weight"):
            continue
        path = key[:-len(".lora_down.weight")]
        actual.update(target for target, suffix in suffixes.items() if path.endswith(suffix))
    declared = set(config.targets)
    if actual != declared:
        raise ValueError(f"Checkpoint Event-LoRA targets contradict weights: declared={sorted(declared)}, present={sorted(actual)}")
    replaced = checkpoint.get("event_film_replaced_names") or []
    for path in replaced:
        target = next((target for target, suffix in suffixes.items() if path.endswith(suffix)), None)
        if target not in declared:
            raise ValueError(f"Checkpoint replaced_names contradict targets: {path}")
    metadata = checkpoint.get("event_lora_metadata") or {}
    expected_layout = ("per_projection_per_scale" if config.variant in {
        "residual_cross_attn", "residual_cross_attn_bax"
    } else "per_module")
    if metadata and metadata.get("shared_heads") != expected_layout:
        raise ValueError(f"Unsupported Event-LoRA shared-head layout: {metadata.get('shared_heads')}")


def _validate_layer_scaling(unet, checkpoint):
    layers = (checkpoint.get("event_lora_metadata") or {}).get("layers") or {}
    modules = dict(iter_event_lora_modules(unet))
    for name, source in layers.items():
        module = modules.get(name)
        if module is None:
            continue
        if int(source["rank"]) != module.rank or abs(float(source["scaling"]) - module.scaling) > 1e-6:
            raise ValueError(f"Event-LoRA rank/scaling mismatch at {name}: checkpoint={source}, runtime={(module.rank, module.scaling)}")


def _assert_event_checkpoint_keys(unet, state, replaced_names, new_targets=()):
    """Event A/B and existing modulation heads must never be silently skipped."""
    expected = unet.state_dict()
    prefixes = tuple(name + "." for name in replaced_names)
    new_names = {name for name, module in iter_event_lora_modules(unet) if module.projection in new_targets}
    critical = [key for key in expected if key.startswith(prefixes) and (
        ".lora_down." in key or ".lora_up." in key or
        (".modulator." in key and not any(key.startswith(name + ".") for name in new_names))
    )]
    critical += [key for key in expected if "_event_lora_shared_heads_" in key and not any(
        key.startswith(f"_event_lora_shared_heads_{variant}_{target}.") or
        (target == "v" and key.startswith(f"_event_lora_shared_heads_{variant}."))
        for target in new_targets for variant in ("residual_cross_attn", "residual_cross_attn_bax")
    )]
    missing = [key for key in critical if key not in state]
    mismatched = [key for key in critical if key in state and expected[key].shape != state[key].shape]
    unexpected = [key for key in state if (
        ".lora_down." in key or ".lora_up." in key or
        ".modulator." in key or "_event_lora_shared_heads_" in key
    ) and key not in expected]
    if missing or mismatched or unexpected:
        raise RuntimeError(
            "Event-LoRA checkpoint is structurally incomplete: "
            f"missing={missing[:12]}, shape_mismatch={mismatched[:12]}, unexpected={unexpected[:12]}"
        )


def resolve_decoder_refinement_variant(value, checkpoint_path=None):
    """解析 CLI variant；``auto`` 从已有 refinement checkpoint 继承配置。

    这样 Stage5/推理在不显式指定 variant 时，能够安全复用 Stage4 权重，
    同时避免在模型构造后才发现结构不匹配。
    """
    variant = normalize_refinement_variant(value, allow_auto=True)
    if variant != "auto":
        return variant
    if not checkpoint_path:
        raise ValueError("decoder_refinement_variant=auto requires a checkpoint path.")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("decoder_refinement_config") or {}
    # 旧 checkpoint 没有 variant 字段时，历史实现就是 legacy_fused。
    return normalize_refinement_variant(config.get("variant", "legacy_fused"))


def resolve_decoder_refinement_args(args, checkpoint_path):
    """``auto`` 时把 checkpoint 中的 CGDRS 结构参数一并带入运行参数。"""
    requested = normalize_refinement_variant(
        getattr(args, "decoder_refinement_variant", "legacy_fused"), allow_auto=True
    )
    if requested != "auto":
        args.decoder_refinement_variant = requested
        return args
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("decoder_refinement_config") or {}
    args.decoder_refinement_variant = normalize_refinement_variant(
        config.get("variant", "legacy_fused")
    )
    if args.decoder_refinement_variant == "cgdrs":
        if "cgdrs_event_levels" in config:
            args.cgdrs_event_levels = ",".join(
                str(level) for level in config["cgdrs_event_levels"]
            )
        if "cgdrs_gate_bias" in config:
            args.cgdrs_gate_bias = float(config["cgdrs_gate_bias"])
        if "cgdrs_activity_mode" in config:
            args.cgdrs_activity_mode = config["cgdrs_activity_mode"]
    return args


def _validate_event_lora_config(runtime_config, checkpoint_config, context):
    structural_keys = (
        "variant",
        "targets",
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


def _keep_event_features_for_checkpoint_backward(unet):
    return torch.is_grad_enabled() and any(
        getattr(module, "gradient_checkpointing", False) for module in unet.modules()
    )


@dataclass
class DenoisingContext:
    """同一图像的 RGB latent、文本条件与原始 event features，供公平 sweep 复用。"""

    encoded_control: torch.Tensor
    prompt_embeds: torch.Tensor
    neg_prompt_embeds: torch.Tensor
    event_feats: dict


def _synchronize_tensor_device(tensor):
    """只在 CUDA 推理计时时同步，CPU 与训练路径不产生额外副作用。"""
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


class OSEDiffEventFiLMGen(torch.nn.Module):
    """
    Event-FiLM LoRA Generator。

    作用：
    - 复用原 OSEDiff 的 VAE、Text Encoder、UNet 与 scheduler；
    - 保留完整 Stage1 可兼容权重；
    - 将选中的 self-attention Q/K/V/out 投影替换成事件调制 LoRA。
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
        self.decoder_refinement_variant = normalize_refinement_variant(
            getattr(args, "decoder_refinement_variant", "legacy_fused")
        )
        self.cgdrs_event_levels = normalize_refinement_levels(
            getattr(args, "cgdrs_event_levels", (1, 2, 3))
        )
        self.cgdrs_gate_bias = float(getattr(args, "cgdrs_gate_bias", -2.0))
        self.cgdrs_activity_mode = normalize_activity_mode(
            getattr(args, "cgdrs_activity_mode", "abs_mean")
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
            refinement_variant=self.decoder_refinement_variant,
            cgdrs_event_levels=self.cgdrs_event_levels,
            cgdrs_gate_bias=self.cgdrs_gate_bias,
            cgdrs_activity_mode=self.cgdrs_activity_mode,
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
                load_fusion=self.decoder_refinement_variant == "legacy_fused",
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

    def activate_lora_adapters(self, parts=UNET_LORA_PARTS):
        """选择普通 UNet LoRA；自定义 Event-LoRA 不受 PEFT 开关影响。"""
        parts = parse_unet_lora_parts(parts)
        if parts:
            self.unet.set_adapter([_UNET_LORA_ADAPTERS[part] for part in parts])
            self.unet.enable_adapters()
        else:
            self.unet.disable_adapters()
        self.vae.set_adapter(["default_encoder"])
        self.active_unet_lora_parts = parts
        print(f"[UNet ordinary LoRA] active={','.join(parts) or 'none'}")

    def set_train_stage5(self):
        """Stage5 的联合微调解冻策略。

        保持 Stable Diffusion UNet 主干、原始 VAE decoder 主体、文本编码器和
        RGB/Event 辅助编码器冻结；只训练已有 LoRA、Event-FiLM 分支、EventEncoder、
        VAE encoder LoRA，以及 refinement decoder 的 feature-fusion blocks。
        """
        if not self.decoder_refinement_enabled or self.rgb_event_encoder is None:
            raise RuntimeError("Stage5 requires an installed decoder refinement module.")

        # initialize_unet 注册了三组 PEFT adapter，但 add_adapter 不保证它们同时
        # 处于 active 状态。未激活的 adapter 不会进入 forward，也不会得到梯度。
        requested_active = getattr(self.args, "unet_lora_active_parts", None)
        requested_trainable = getattr(self.args, "unet_lora_trainable_parts", None)
        if requested_active is None and requested_trainable is None:
            active_parts = trainable_parts = UNET_LORA_PARTS
        else:
            active_parts, trainable_parts = validate_unet_lora_training_parts(
                UNET_LORA_PARTS if requested_active is None else requested_active,
                () if requested_trainable is None else requested_trainable,
            )
        self.activate_lora_adapters(active_parts)
        self.trainable_unet_lora_parts = trainable_parts

        # 先统一冻结，避免 Stage1/Stage3 的 requires_grad 状态泄漏到 Stage5。
        self.requires_grad_(False)
        self.unet.train()
        self.vae.train()
        self.event_encoder.train()
        self.text_encoder.eval()
        self.rgb_event_encoder.eval()

        # PEFT A/B follow the independent ordinary-LoRA selection.
        selected_adapters = {_UNET_LORA_ADAPTERS[part] for part in trainable_parts}
        for name, parameter in self.unet.named_parameters():
            if "lora_A." in name or "lora_B." in name:
                marker = "lora_A." if "lora_A." in name else "lora_B."
                adapter = name.split(marker, 1)[1].split(".", 1)[0]
                parameter.requires_grad_(adapter in selected_adapters)
        set_event_lora_trainable(
            self.unet,
            train_lora=True,
            train_modulation=True,
        )
        set_modulation_enabled_for_all_event_lora(self.unet, True)

        # EventEncoder 是 Stage2/3 已训练的事件条件分支，Stage5 继续联合适配。
        self.event_encoder.requires_grad_(bool(self.event_lora_config.targets))

        # VAE 只开放 encoder LoRA；原始 decoder backbone 保持冻结。
        for name, parameter in self.vae.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
        self.vae.decoder.feature_fusion_blocks.requires_grad_(True)

        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        allowed_ids = {
            id(parameter)
            for parameter in self.event_encoder.parameters()
        }
        allowed_ids.update(
            id(parameter)
            for parameter in self.vae.decoder.feature_fusion_blocks.parameters()
        )
        allowed_ids.update(
            id(parameter)
            for name, parameter in self.vae.named_parameters()
            if "lora_" in name
        )
        # Event-LoRA 的 modulation/shared-head 参数名称不一定包含 lora_。
        allowed_ids.update(
            id(parameter)
            for _, module in iter_event_lora_modules(self.unet)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        allowed_ids.update(
            id(parameter)
            for name, parameter in self.unet.named_parameters()
            if "lora_" in name
        )
        allowed_prefixes = (
            "event_encoder.",
            "vae.decoder.feature_fusion_blocks.",
        )
        stray = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in allowed_ids and (
                name.startswith(allowed_prefixes)
                or name.startswith("vae.") and "lora_" in name
                or name.startswith("unet.")
            ):
                continue
            stray.append(name)
        if stray:
            raise RuntimeError(f"Unexpected Stage5 trainable parameters: {stray}")
        print(f"[Stage5 LoRA] active={active_parts}, trainable={trainable_parts}, "
              f"parameters={self.trainable_parameter_summary()}")
        return trainable

    def install_event_lora(self):
        if self.event_film_installed:
            return 0, self.event_film_replaced_names
        num_replaced, names = replace_self_attn_projections_with_event_lora(
            self.unet,
            config=self.event_lora_config,
            event_channels_by_size=self.event_encoder.out_channels_by_size,
        )
        self.event_film_installed = True
        self.event_film_replaced_names = names
        self.event_film_replaced_by_target = {
            target: [name for name in names if name.endswith({
                "q": "attn1.to_q", "k": "attn1.to_k", "v": "attn1.to_v",
                "out": "attn1.to_out.0",
            }[target])]
            for target in ("q", "k", "v", "out")
        }
        counts = {target: len(paths) for target, paths in self.event_film_replaced_by_target.items()}
        print(
            f"[Event-LoRA:{self.event_lora_config.variant}] "
            f"targets={','.join(self.event_lora_config.targets) or 'none'}; "
            f"replaced={counts}; shared_heads={('per_projection_per_scale' if self.event_lora_config.variant in {'residual_cross_attn', 'residual_cross_attn_bax'} else 'per_module')}."
        )
        for name in names[:20]:
            print(f"[Event-LoRA]   {name}")
        return num_replaced, names

    # 兼容旧代码入口。
    install_event_film_lora = install_event_lora

    def load_stage1_full(self, checkpoint_path):
        """
        完整加载 Stage1 中可兼容权重，然后迁移选中投影的普通 LoRA 到 Event-LoRA。

        加载顺序很关键：
        1. 先把 Stage1 的 PEFT LoRA/conv_in/VAE LoRA 加到原始结构；
        2. 再替换选中投影，此时替换工具会复制已经加载好的 LoRA A/B。
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
        if stage in {2, 3, "2", "3"}:
            active_parts, trainable_parts = validate_unet_lora_training_parts(
                getattr(self.args, "unet_lora_active_parts", ()),
                getattr(self.args, "unet_lora_trainable_parts", ()),
            )
            self.activate_lora_adapters(active_parts)
            self.trainable_unet_lora_parts = trainable_parts
        self.unet.train()
        self.vae.train()
        self.event_encoder.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.event_encoder.requires_grad_(bool(self.event_lora_config.targets) and stage in {2, 3, "2", "3"})
        if stage in {2, 3, "2", "3"}:
            selected_adapters = {
                _UNET_LORA_ADAPTERS[part] for part in self.trainable_unet_lora_parts
            }
            matched_adapters = set()
            for name, param in self.unet.named_parameters():
                if "lora_A." in name or "lora_B." in name:
                    marker = "lora_A." if "lora_A." in name else "lora_B."
                    adapter = name.split(marker, 1)[1].split(".", 1)[0]
                    param.requires_grad_(adapter in selected_adapters)
                    matched_adapters.add(adapter)
            if selected_adapters - matched_adapters:
                raise RuntimeError(
                    "Selected ordinary UNet LoRA adapters have no parameters: "
                    f"{sorted(selected_adapters - matched_adapters)}"
                )
            print(f"[UNet ordinary LoRA] trainable={','.join(trainable_parts) or 'none'}")
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
        if stage in {2, 3, "2", "3"} and not any(p.requires_grad for p in self.unet.parameters()) and not any(p.requires_grad for p in self.event_encoder.parameters()):
            raise RuntimeError("No generator parameters participate in training: Event-LoRA targets and ordinary LoRA trainable parts are both none.")
        if stage in {2, 3, "2", "3"}:
            print(f"[Event-LoRA] A/B trainable={stage in {3, '3'}}; "
                  f"parameter_groups={self.trainable_parameter_summary()}")

    def trainable_parameter_summary(self):
        event_ab = {id(param) for _, module in iter_event_lora_modules(self.unet)
                    for param in (*module.lora_down.parameters(), *module.lora_up.parameters())}
        summary = {"ordinary_ab": 0, "event_ab": 0, "event_heads": 0,
                   "event_encoder": 0}
        for name, param in self.unet.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in event_ab:
                key = "event_ab"
            elif "lora_A." in name or "lora_B." in name:
                key = "ordinary_ab"
            else:
                key = "event_heads"
            summary[key] += param.numel()
        summary["event_encoder"] = sum(param.numel() for param in self.event_encoder.parameters()
                                       if param.requires_grad)
        return summary

    def get_lora_params(self):
        params = []
        for name, param in self.unet.named_parameters():
            if param.requires_grad and _is_unet_lora_parameter_name(name):
                params.append(param)
        return params

    def get_event_params(self):
        # residual variant 的共享 head 注册在 UNet 顶层，名称不含 modulator；
        # 因此以 trainable 且非 LoRA A/B 为准，并按参数 id 去重。
        params = [p for p in self.event_encoder.parameters() if p.requires_grad]
        seen = {id(param) for param in params}
        for name, param in self.unet.named_parameters():
            if not param.requires_grad or _is_unet_lora_parameter_name(name):
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

    def prepare_denoising_context(self, c_t, events, batch):
        """各执行一次 VAE/text/event encode，固定重复推理的全部公共输入。"""
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        event_feats = self.event_encoder(events) if self.event_lora_config.targets else {}
        return DenoisingContext(
            encoded_control=encoded_control,
            prompt_embeds=prompt_embeds,
            neg_prompt_embeds=neg_prompt_embeds,
            event_feats=event_feats,
        )

    def _run_unet_with_event_features(
        self, encoded_control, timesteps, prompt_embeds, event_feats, feature_guidance=None
    ):
        """在一次 UNet forward 的严格作用域内设置并清理 Event-LoRA feature。"""
        cross_attn_dtype = _unet_cross_attention_dtype(
            self.unet, default=encoded_control.dtype
        )
        set_event_feats_for_all_event_lora(self.unet, event_feats)
        kwargs = {} if feature_guidance is None else {"feature_guidance": feature_guidance}
        try:
            result = self.unet(
                encoded_control,
                timesteps,
                encoder_hidden_states=prompt_embeds.to(dtype=cross_attn_dtype),
                **kwargs,
            ).sample
        except BaseException:
            clear_event_feats_for_all_event_lora(self.unet)
            raise
        if not _keep_event_features_for_checkpoint_backward(self.unet):
            clear_event_feats_for_all_event_lora(self.unet)
        return result

    def predict_denoised_latent_from_context(
        self, context, timesteps, measure_time=False, feature_guidance=None,
    ):
        """固定 context 的普通单步预测；可选回调只修改当前样本的主干特征。"""
        if not isinstance(context, DenoisingContext):
            raise TypeError(f"Expected DenoisingContext, got {type(context)}.")
        if measure_time:
            _synchronize_tensor_device(context.encoded_control)
            start_time = time.perf_counter()
        kwargs = {} if feature_guidance is None else {"feature_guidance": feature_guidance}
        model_pred = self._run_unet_with_event_features(
            context.encoded_control, timesteps, context.prompt_embeds,
            context.event_feats, **kwargs,
        )
        stats = {}
        if measure_time:
            _synchronize_tensor_device(context.encoded_control)
            stats["unet_ms"] = (time.perf_counter() - start_time) * 1000.0
        # 候选始终从原 latent 做同一次 scheduler 映射，不递推上一次输出。
        x_denoised = self.noise_scheduler.step(
            model_pred, timesteps, context.encoded_control, return_dict=True,
        ).prev_sample
        return x_denoised, stats

    def predict_denoised_latent(self, c_t, events, timesteps, batch):
        """兼容旧接口：运行一次 VAE encode + event-cLoRA UNet 并返回 z0。"""
        context = self.prepare_denoising_context(c_t, events, batch)
        x_denoised, _ = self.predict_denoised_latent_from_context(
            context, timesteps
        )
        return x_denoised, context.prompt_embeds, context.neg_prompt_embeds

    def decode_denoised_latent(
        self,
        x_denoised,
        c_t,
        events_refinement=None,
        decoder_fusion_enabled=None,
        refinement_fusion_mode=None,
        clamp_output=True,
    ):
        """对固定 z0 只执行一次 decoder，供 off/on 公平对比复用。"""
        vae_dtype = _first_parameter_dtype(self.vae, default=x_denoised.dtype)
        fusion_enabled = (
            self.decoder_fusion_enabled
            if decoder_fusion_enabled is None else bool(decoder_fusion_enabled)
        )
        fusion_mode = refinement_fusion_mode or self.refinement_fusion_mode
        fused_features = None
        rgb_features = None
        event_features = None
        event_activity = None
        if self.decoder_refinement_enabled:
            if events_refinement is None:
                raise ValueError(
                    "Decoder refinement requires the separately normalized "
                    "events_refinement tensor."
                )
            with torch.no_grad():
                encoder_output = self.rgb_event_encoder(
                    c_t,
                    events_refinement,
                    fusion_mode=fusion_mode,
                )
            if self.decoder_refinement_variant == "legacy_fused":
                fused_features = [
                    feature.detach() for feature in encoder_output.fused_features
                ]
            else:
                rgb_features = [feature.detach() for feature in encoder_output.rgb_features]
                event_features = [feature.detach() for feature in encoder_output.event_features]
                # 沿 event-bin 聚合得到稳定的空间活动图，避免 gate 依赖具体 bin 分布。
                activity_reduce = (
                    events_refinement.detach().float().abs().sum
                    if self.cgdrs_activity_mode == "abs_sum"
                    else events_refinement.detach().float().abs().mean
                )
                activity = activity_reduce(dim=1, keepdim=True)
                activity = activity / activity.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                activity = activity.clamp(0.0, 4.0).div(4.0)
                event_activity = [
                    F.interpolate(
                        activity.to(device=feature.device),
                        size=feature.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).to(dtype=feature.dtype).detach()
                    for feature in event_features
                ]
            self.last_refinement_stats = {}
            for family, features in (
                ("rgb", encoder_output.rgb_features),
                ("event", encoder_output.event_features),
                ("fused", encoder_output.fused_features),
            ):
                if features is None:
                    continue
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
                rgb_features=rgb_features,
                event_features=event_features,
                event_activity=event_activity,
                fusion_enabled=fusion_enabled,
                fusion_weight=self.decoder_fusion_weight,
                fusion_mode=fusion_mode,
            ).sample
        )
        if clamp_output:
            output = output.clamp(-1, 1)
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
            "event_lora_metadata": {
                "version": 2,
                "targets": tuple(self.event_lora_config.targets),
                "shared_heads": ("per_projection_per_scale" if self.event_lora_config.variant in {
                    "residual_cross_attn", "residual_cross_attn_bax"
                } else "per_module"),
                "replaced_by_target": getattr(self, "event_film_replaced_by_target", {}),
                "layers": {
                    name: {"projection": module.projection, "rank": module.rank,
                           "alpha": module.lora_alpha, "scaling": module.scaling}
                    for name, module in iter_event_lora_modules(self.unet)
                },
            },
            "event_lora_variant": self.event_lora_config.variant,
            "event_film_replaced_names": self.event_film_replaced_names,
            "train_unet_lora_active_parts": tuple(getattr(self, "active_unet_lora_parts", ())),
            "train_unet_lora_trainable_parts": tuple(getattr(self, "trainable_unet_lora_parts", ())),
            "stage": stage,
            "timestep_config": timestep_config or {},
            "global_step": global_step,
            "epoch": epoch,
        }
        if stage in {2, 3, "2", "3"}:
            sd["event_regularization_mode"] = "legacy_detached_delta_and_head_l2"
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
                "checkpoint_version": "idea3_phase1_v2",
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
        checkpoint_variant = normalize_refinement_variant(
            checkpoint_config.get("variant", "legacy_fused")
        )
        runtime_variant = normalize_refinement_variant(
            runtime_config.get("variant", "legacy_fused")
        )
        if checkpoint_variant != runtime_variant:
            raise ValueError(
                "Decoder refinement variant mismatch: "
                f"runtime={runtime_variant}, checkpoint={checkpoint_variant}. "
                "Start a new Stage4 run instead of cross-variant resume."
            )
        # CGDRS 不需要 Merge 前向；显式关闭可避免无效计算。
        self.rgb_event_encoder.fusion_enabled = runtime_variant == "legacy_fused"
        structural_keys = (
            "variant",
            "config_version",
            "condition_channels",
            "decoder_channels",
            "fusion_blocks",
            "fusion_init",
            "cgdrs_event_levels",
            "cgdrs_gate_bias",
            "cgdrs_activity_mode",
            "cgdrs_activity_clip",
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

    def load_event_checkpoint(self, checkpoint, ignore_config_mismatch=False, migrate_targets=False):
        checkpoint_config = _checkpoint_event_lora_config(checkpoint)
        if checkpoint_config.targets != self.event_lora_config.targets and not migrate_targets:
            raise ValueError(
                f"Event-LoRA targets mismatch: runtime={self.event_lora_config.targets}, "
                f"checkpoint={checkpoint_config.targets}. Use weights-only initialization for migration."
            )
        if ignore_config_mismatch:
            print("[Event-LoRA Load] config mismatch checks are explicitly disabled.")
        else:
            if not migrate_targets:
                _validate_event_lora_config(
                    self.event_lora_config, checkpoint_config, context="Generator"
                )
            else:
                _validate_event_lora_config(
                    EventLoRAConfig(**{**self.event_lora_config.to_dict(), "targets": checkpoint_config.targets}),
                    checkpoint_config, context="Generator migration",
                )
        self.install_event_lora()
        unet_state = _event_checkpoint_state(checkpoint)
        _validate_checkpoint_target_evidence(checkpoint, checkpoint_config, unet_state)
        _validate_layer_scaling(self.unet, checkpoint)
        if migrate_targets and checkpoint_config.targets != self.event_lora_config.targets:
            unet_state, report = migrate_event_lora_state(
                self.unet, unet_state, checkpoint_config.targets,
                self.event_lora_config.targets,
                (checkpoint.get("event_lora_metadata") or {}).get("layers"),
            )
            print(f"[Event-LoRA migration] {report}")
            _assert_event_checkpoint_keys(
                self.unet, unet_state, self.event_film_replaced_names,
                new_targets=set(self.event_lora_config.targets) - set(checkpoint_config.targets),
            )
        else:
            _assert_event_checkpoint_keys(self.unet, unet_state, self.event_film_replaced_names)
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
        # The caller applies its own training/inference ordinary-LoRA policy.


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
        if self.regularizer_lora_mode in {"event_lora", "event_film"}:
            print(
                f"[DEPRECATED] regularizer_lora_mode={self.regularizer_lora_mode} "
                "is mapped to same_as_generator."
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
        num_replaced, names = replace_self_attn_projections_with_event_lora(
            self.unet_update,
            config=self.event_lora_config,
            event_channels_by_size=event_encoder.out_channels_by_size,
        )
        self.event_film_installed = True
        self.event_film_replaced_names = names
        print(
            f"[Reg Event-LoRA:{self.event_lora_config.variant}] "
            f"targets={','.join(self.event_lora_config.targets) or 'none'}; replaced={num_replaced}."
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

    def load_ckpt(self, checkpoint, strict_config=True, migrate_targets=False):
        state = checkpoint.get("state_dict_reg", checkpoint.get("state_dict_unet", {}))
        checkpoint_mode = checkpoint.get("regularizer_lora_mode")
        if checkpoint_mode in {"event_lora", "event_film"}:
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
        if (migrate_targets and checkpoint_config_dict and self.event_lora_config is not None
                and EventLoRAConfig(**checkpoint_config_dict).variant != self.event_lora_config.variant):
            raise ValueError("Regularizer Event-LoRA variant cannot change during target migration.")
        source_targets = EventLoRAConfig(**checkpoint_config_dict).targets if checkpoint_config_dict else ("v",)
        regularizer_metadata = checkpoint.get("regularizer_event_lora_metadata") or {}
        if regularizer_metadata and tuple(regularizer_metadata.get("targets", ())) != tuple(source_targets):
            raise ValueError("Regularizer checkpoint targets contradict its config.")
        if regularizer_metadata and self.event_lora_config is not None:
            layout = ("per_projection_per_scale" if self.event_lora_config.variant in {
                "residual_cross_attn", "residual_cross_attn_bax"
            } else "per_module")
            if regularizer_metadata.get("shared_heads") != layout:
                raise ValueError("Regularizer Event-LoRA shared-head layout mismatch.")
        if (strict_config or not migrate_targets) and self.regularizer_lora_mode == "same_as_generator" and self.event_lora_config is not None and any(".lora_down." in key for key in state) and source_targets != self.event_lora_config.targets:
            raise ValueError(f"Regularizer Event-LoRA targets mismatch: {source_targets} -> {self.event_lora_config.targets}")
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
            if migrate_targets and source_targets != self.event_lora_config.targets:
                state, report = migrate_event_lora_state(
                    self.unet_update, state, source_targets, self.event_lora_config.targets,
                    regularizer_metadata.get("layers"),
                )
                print(f"[Reg Event-LoRA migration] {report}")
                _assert_event_checkpoint_keys(
                    self.unet_update, state, self.event_film_replaced_names,
                    new_targets=set(self.event_lora_config.targets) - set(source_targets),
                )
            else:
                _assert_event_checkpoint_keys(self.unet_update, state, self.event_film_replaced_names)
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
        active = getattr(self.unet_update, "active_adapters", ())
        if callable(active):
            active = active()
        if isinstance(active, str):
            active = (active,)
        active = active or ()
        trainable = sorted({
            name.split(marker, 1)[1].split(".", 1)[0]
            for name, param in self.unet_update.named_parameters()
            for marker in ("lora_A.", "lora_B.")
            if param.requires_grad and marker in name
        })
        state = {
            "state_dict_reg": self.state_dict_reg(),
            "regularizer_lora_mode": self.regularizer_lora_mode,
            "regularizer_event_lora_config": (
                self.event_lora_config.to_dict()
                if self.regularizer_lora_mode == "same_as_generator"
                else None
            ),
            "regularizer_event_lora_metadata": (
                {
                    "version": 2,
                    "targets": tuple(self.event_lora_config.targets),
                    "shared_heads": ("per_projection_per_scale" if self.event_lora_config.variant in {
                        "residual_cross_attn", "residual_cross_attn_bax"
                    } else "per_module"),
                    "replaced_names": tuple(self.event_film_replaced_names),
                    "layers": {
                        name: {"rank": module.rank, "scaling": module.scaling}
                        for name, module in iter_event_lora_modules(self.unet_update)
                    },
                }
                if self.regularizer_lora_mode == "same_as_generator" else None
            ),
            "reg_event_film_replaced_names": self.event_film_replaced_names,
            "regularizer_ordinary_lora_active_adapters": tuple(active),
            "regularizer_ordinary_lora_trainable_adapters": tuple(trainable),
        }
        return state

    def _predict_with_update_unet(self, noisy_latents, timesteps, prompt_embeds, events):
        if self.regularizer_lora_mode != "same_as_generator" or not self.event_lora_config.targets:
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
            result = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample
        except BaseException:
            clear_event_feats_for_all_event_lora(self.unet_update)
            raise
        if not _keep_event_features_for_checkpoint_backward(self.unet_update):
            clear_event_feats_for_all_event_lora(self.unet_update)
        return result

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
        # Stage2/3 测试入口显式提供默认空选择；其他使用此 wrapper 的
        # Stage4/5 入口继续沿用完整普通 LoRA adapter 组合。
        self.gen.activate_lora_adapters(
            getattr(args, "unet_lora_parts", UNET_LORA_PARTS)
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
