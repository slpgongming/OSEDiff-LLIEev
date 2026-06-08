import copy

import torch
import torch.nn as nn

from models.event_film_lora import SpatialRGBEventFiLMLoRALinear


def get_parent_module(root, module_name):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_event_film_lora_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, SpatialRGBEventFiLMLoRALinear):
            yield name, module


def _first_lora_adapter(module):
    if not hasattr(module, "lora_A"):
        return None
    keys = list(module.lora_A.keys())
    return keys[0] if keys else None


def _extract_peft_linear(module):
    """
    从 PEFT LoRA Linear 中取出冻结 base_layer 和已经加载的 LoRA A/B。

    Stage2 要完整继承 Stage1，因此替换前应先把 Stage1 checkpoint 加载到
    PEFT 模块里，再调用本函数迁移 attn1.to_v 的 LoRA 权重。
    """
    if hasattr(module, "base_layer"):
        base_linear = copy.deepcopy(module.base_layer)
        adapter = _first_lora_adapter(module)
        if adapter is None:
            return base_linear, None, None, None, None
        lora_a = module.lora_A[adapter].weight.detach().clone()
        lora_b = module.lora_B[adapter].weight.detach().clone()
        alpha = module.lora_alpha[adapter] if hasattr(module, "lora_alpha") else lora_a.shape[0]
        scaling = module.scaling[adapter] if hasattr(module, "scaling") else float(alpha) / lora_a.shape[0]
        return base_linear, lora_a, lora_b, alpha, scaling
    if isinstance(module, nn.Linear):
        return copy.deepcopy(module), None, None, None, None
    raise TypeError(f"Unsupported to_v module type for Event-FiLM replacement: {type(module)}")


def replace_self_attn_to_v_with_event_film_lora(
    unet,
    rank=4,
    modulation_scale=0.1,
    mod_channels=32,
    event_channels_by_size=None,
):
    """
    只替换 self-attention 的 attn1.to_v，不触碰 attn2.to_v。

    返回：
        (num_replaced, replaced_names)
    """
    replaced_names = []
    for name, module in list(unet.named_modules()):
        if not name.endswith("attn1.to_v"):
            continue
        base_linear, lora_a, lora_b, alpha, _ = _extract_peft_linear(module)
        effective_rank = lora_a.shape[0] if lora_a is not None else rank
        event_lora = SpatialRGBEventFiLMLoRALinear(
            base_linear=base_linear,
            rank=effective_rank,
            lora_alpha=alpha if alpha is not None else effective_rank,
            modulation_scale=modulation_scale,
            mod_channels=mod_channels,
            event_channels_by_size=event_channels_by_size,
        )
        if lora_a is not None:
            event_lora.lora_down.weight.data.copy_(lora_a)
        if lora_b is not None:
            event_lora.lora_up.weight.data.copy_(lora_b)
        parent, child_name = get_parent_module(unet, name)
        setattr(parent, child_name, event_lora)
        replaced_names.append(name)
    return len(replaced_names), replaced_names


def set_event_feats_for_all_event_lora(unet, event_feats):
    for _, module in iter_event_film_lora_modules(unet):
        module.set_event_feats(event_feats)


def clear_event_feats_for_all_event_lora(unet):
    for _, module in iter_event_film_lora_modules(unet):
        module.clear_event_feats()


def set_modulation_enabled_for_all_event_lora(unet, enabled):
    for _, module in iter_event_film_lora_modules(unet):
        module.set_modulation_enabled(enabled)


def set_event_lora_trainable(unet, train_lora=False, train_modulation=True):
    """
    控制 Event-FiLM 模块内部参数的训练状态。

    Stage2: train_lora=False, train_modulation=True
    Stage3: train_lora=True,  train_modulation=True
    """
    for _, module in iter_event_film_lora_modules(unet):
        module.base_linear.requires_grad_(False)
        module.lora_down.requires_grad_(train_lora)
        module.lora_up.requires_grad_(train_lora)
        module.event_proj.requires_grad_(train_modulation)
        module.z_proj.requires_grad_(train_modulation)
        module.modulation_head.requires_grad_(train_modulation)


def collect_event_lora_debug(unet):
    stats = {
        "dgamma_abs_mean": [],
        "dbeta_abs_mean": [],
        "dgamma_abs_max": [],
        "dbeta_abs_max": [],
        "base_norm": [],
        "lora_norm": [],
    }
    for _, module in iter_event_film_lora_modules(unet):
        for key in stats:
            if key in module.last_debug:
                stats[key].append(module.last_debug[key])
    out = {}
    for key, values in stats.items():
        out[key] = float(sum(values) / len(values)) if values else 0.0
    return out


def collect_event_lora_regularization(unet, device):
    smooth = torch.zeros((), device=device)
    mod = torch.zeros((), device=device)
    count = 0
    for _, module in iter_event_film_lora_modules(unet):
        debug_feats = getattr(module, "last_debug", None)
        # 真实 dgamma/dbeta 没有缓存张量，正则项用 modulation head 权重约束，避免持久保存大特征图。
        for p in module.modulation_head.parameters():
            mod = mod + p.float().pow(2).mean()
            count += 1
        for p in module.event_proj.parameters():
            smooth = smooth + p.float().pow(2).mean()
            count += 1
        _ = debug_feats
    if count > 0:
        smooth = smooth / count
        mod = mod / count
    return smooth, mod


def event_film_state_keys():
    return (
        "lora_down",
        "lora_up",
        "event_proj",
        "z_proj",
        "modulation_head",
    )


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
