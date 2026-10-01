import copy

import torch
import torch.nn as nn

from models.event_film_lora import (
    EventLoRAConfig,
    SharedResidualEventHeads,
    SpatialRGBEventLoRALinear,
)


def get_parent_module(root, module_name):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_event_lora_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, SpatialRGBEventLoRALinear):
            yield name, module


# 兼容旧调用名称。
iter_event_film_lora_modules = iter_event_lora_modules


def _first_lora_adapter(module):
    if not hasattr(module, "lora_A"):
        return None
    keys = list(module.lora_A.keys())
    return keys[0] if keys else None


def _extract_peft_linear(module):
    """提取冻结 base Linear 和已加载的 PEFT LoRA A/B，供 Event-LoRA 无损迁移。"""
    if hasattr(module, "base_layer"):
        base_linear = copy.deepcopy(module.base_layer)
        adapter = _first_lora_adapter(module)
        if adapter is None:
            return base_linear, None, None, None
        lora_a = module.lora_A[adapter].weight.detach().clone()
        lora_b = module.lora_B[adapter].weight.detach().clone()
        alpha = module.lora_alpha[adapter] if hasattr(module, "lora_alpha") else lora_a.shape[0]
        return base_linear, lora_a, lora_b, alpha
    if isinstance(module, nn.Linear):
        return copy.deepcopy(module), None, None, None
    raise TypeError(f"Unsupported to_v module type: {type(module)}")


def _infer_bax_query_channels_by_size(unet, event_channels_by_size):
    """按 UNet 从高到低的 block channels 推导各事件尺度 BAx 通道。"""
    unet_config = getattr(unet, "config", None)
    block_channels = getattr(unet_config, "block_out_channels", None)
    if block_channels is None and isinstance(unet_config, dict):
        block_channels = unet_config.get("block_out_channels")
    if block_channels is None:
        raise ValueError("residual_cross_attn_bax requires unet.config.block_out_channels.")
    sizes = sorted((int(size) for size in event_channels_by_size), reverse=True)
    block_channels = [int(channels) for channels in block_channels]
    if len(sizes) != len(block_channels):
        raise ValueError(
            "Cannot map event sizes to UNet channels: "
            f"sizes={sizes}, block_channels={block_channels}."
        )
    return {str(size): channels for size, channels in zip(sizes, block_channels)}


def replace_self_attn_to_v_with_event_lora(unet, config, event_channels_by_size):
    """只替换图像 self-attention 的 `attn1.to_v`，不触碰文本 cross-attention。"""
    replaced_names = []
    shared_residual_heads = None
    if config.variant in {"residual_cross_attn", "residual_cross_attn_bax"}:
        shared_attr = f"_event_lora_shared_heads_{config.variant}"
        shared_residual_heads = getattr(unet, shared_attr, None)
        if shared_residual_heads is None:
            query_channels_by_size = None
            if config.variant == "residual_cross_attn_bax":
                query_channels_by_size = _infer_bax_query_channels_by_size(
                    unet, event_channels_by_size
                )
            shared_residual_heads = SharedResidualEventHeads(
                config=config,
                event_channels_by_size=event_channels_by_size,
                query_channels_by_size=query_channels_by_size,
            )
            setattr(unet, shared_attr, shared_residual_heads)
    for name, module in list(unet.named_modules()):
        if not name.endswith("attn1.to_v"):
            continue
        base_linear, lora_a, lora_b, alpha = _extract_peft_linear(module)
        effective_rank = lora_a.shape[0] if lora_a is not None else config.rank
        layer_config = EventLoRAConfig(
            **{
                **config.to_dict(),
                "rank": effective_rank,
                "alpha": float(alpha if alpha is not None else effective_rank),
            }
        )
        event_lora = SpatialRGBEventLoRALinear(
            base_linear=base_linear,
            config=layer_config,
            event_channels_by_size=event_channels_by_size,
            shared_residual_heads=shared_residual_heads,
        )
        if lora_a is not None:
            event_lora.lora_down.weight.data.copy_(lora_a)
        if lora_b is not None:
            event_lora.lora_up.weight.data.copy_(lora_b)
        parent, child_name = get_parent_module(unet, name)
        setattr(parent, child_name, event_lora)
        replaced_names.append(name)
    return len(replaced_names), replaced_names


def replace_self_attn_to_v_with_event_film_lora(
    unet,
    rank=4,
    modulation_scale=0.1,
    mod_channels=32,
    event_channels_by_size=None,
):
    """旧接口兼容层：等价于 variant=film_affine。"""
    config = EventLoRAConfig(
        variant="film_affine",
        rank=rank,
        alpha=rank,
        modulation_scale=modulation_scale,
        mod_channels=mod_channels,
    )
    return replace_self_attn_to_v_with_event_lora(
        unet, config, event_channels_by_size or {}
    )


def set_event_feats_for_all_event_lora(unet, event_feats):
    for _, module in iter_event_lora_modules(unet):
        module.set_event_feats(event_feats)


def clear_event_feats_for_all_event_lora(unet):
    for _, module in iter_event_lora_modules(unet):
        module.clear_event_feats()


def set_modulation_enabled_for_all_event_lora(unet, enabled):
    for _, module in iter_event_lora_modules(unet):
        module.set_modulation_enabled(enabled)


def set_event_lora_trainable(unet, train_lora=False, train_modulation=True):
    for _, module in iter_event_lora_modules(unet):
        module.base_linear.requires_grad_(False)
        module.lora_down.requires_grad_(train_lora)
        module.lora_up.requires_grad_(train_lora)
        if module.modulator is not None:
            module.modulator.requires_grad_(train_modulation)
        if hasattr(module, "shared_residual_heads"):
            module.shared_residual_heads.requires_grad_(train_modulation)


def collect_event_lora_debug(unet):
    values_by_key = {}
    for _, module in iter_event_lora_modules(unet):
        for key, value in module.last_debug.items():
            values_by_key.setdefault(key, []).append(value)
    return {
        key: float(sum(values) / len(values))
        for key, values in values_by_key.items()
        if values
    }


def collect_event_lora_visuals(unet):
    """返回最近一次 forward 的首组空间调制热力图。"""
    for _, module in iter_event_lora_modules(unet):
        if module.last_visuals:
            return {key: value.detach() for key, value in module.last_visuals.items()}
    return {}


def _total_variation(x):
    loss = torch.zeros((), device=x.device, dtype=x.dtype)
    if x.shape[-2] > 1:
        loss = loss + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    if x.shape[-1] > 1:
        loss = loss + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return loss


def collect_event_lora_regularization(unet, device):
    """旧版正则：detach 后的调制增量统计，加可反传的事件头 L2。"""
    smooth = torch.zeros((), device=device)
    mod = torch.zeros((), device=device)
    count = 0
    seen = set()
    for _, module in iter_event_lora_modules(unet):
        z_delta = module.last_z_delta
        if z_delta is not None:
            z_delta = z_delta.to(device=device, dtype=torch.float32)
            mod = mod + z_delta.pow(2).mean()
            smooth = smooth + _total_variation(z_delta)
            count += 1
        # residual variant 的共享头只计一次；其他 variant 使用独立 modulator。
        head = (module.shared_residual_heads if hasattr(module, "shared_residual_heads")
                else module.modulator)
        if head is not None:
            for param in head.parameters():
                if id(param) in seen:
                    continue
                seen.add(id(param))
                mod = mod + param.float().pow(2).mean()
                count += 1
    if count:
        smooth = smooth / count
        mod = mod / count
    return smooth, mod


def event_lora_state_keys():
    return (
        "lora_down",
        "lora_up",
        "modulator",
        "_event_lora_shared_heads",
        # 兼容旧 idea2 Event-FiLM checkpoint 的 key。
        "event_proj",
        "z_proj",
        "modulation_head",
        "core",
        "out_proj",
        "z_pos",
        "event_pos",
        "q_proj",
        "k_proj",
        "v_proj",
        "attn_out",
    )


event_film_state_keys = event_lora_state_keys


def remap_legacy_event_film_state_dict(state_dict):
    """把 idea2 旧版 FiLM 模块 key 映射到统一 modulator 层级。"""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if "_event_film_shared_modulation_heads" in new_key:
            new_key = new_key.replace(
                "_event_film_shared_modulation_heads",
                "_event_lora_shared_heads_film_affine",
            )
        elif ".modulator." not in new_key and "_event_lora_shared_heads" not in new_key:
            new_key = new_key.replace(".event_proj.", ".modulator.event_proj.")
            new_key = new_key.replace(".z_proj.", ".modulator.z_proj.")
            new_key = new_key.replace(".modulation_head.", ".modulator.fusion.")
        remapped[new_key] = value
    return remapped


def copy_compatible_state_dict(module, state_dict, prefix_filter=None):
    own_state = module.state_dict()
    loaded, skipped = 0, 0
    for key, value in state_dict.items():
        if prefix_filter is not None and not prefix_filter(key):
            continue
        if key in own_state and own_state[key].shape == value.shape:
            own_state[key].copy_(value)
            loaded += 1
        else:
            skipped += 1
    return loaded, skipped
