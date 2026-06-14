import copy

import torch
import torch.nn as nn

from models.event_film_lora import SharedEventLoRAHeads, SpatialRGBEventFiLMLoRALinear


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


def _event_lora_attr_name(variant):
    return f"_event_lora_shared_heads_{variant}"


def replace_self_attn_to_v_with_event_lora(
    unet,
    rank=4,
    modulation_scale=0.1,
    mod_channels=64,
    event_channels_by_size=None,
    mod_num_heads=4,
    mod_window_size=8,
    event_lora_variant="film",
):
    """
    只替换 self-attention 的 attn1.to_v，不触碰 attn2.to_v。

    返回：
        (num_replaced, replaced_names)
    """
    replaced_names = []
    shared_attr = _event_lora_attr_name(event_lora_variant)
    shared_heads = getattr(unet, shared_attr, None)
    shared_rank = None
    for name, module in list(unet.named_modules()):
        if not name.endswith("attn1.to_v"):
            continue
        base_linear, lora_a, lora_b, alpha, _ = _extract_peft_linear(module)
        effective_rank = lora_a.shape[0] if lora_a is not None else rank
        if shared_heads is None:
            shared_heads = SharedEventLoRAHeads(
                variant=event_lora_variant,
                rank=effective_rank,
                event_channels_by_size=event_channels_by_size,
                dim=mod_channels,
                num_heads=mod_num_heads,
                window_size=mod_window_size,
            )
            # 将共享 Event-LoRA heads 注册到当前 UNet 顶层；Generator 与
            # finetuned regularizer 会各自调用本函数，因此权重天然独立。
            setattr(unet, shared_attr, shared_heads)
            shared_rank = effective_rank
        elif shared_rank is None:
            shared_rank = effective_rank
        elif effective_rank != shared_rank:
            raise ValueError(
                f"Event-LoRA shared heads require one LoRA rank, got {shared_rank} and {effective_rank}."
            )
        event_lora = SpatialRGBEventFiLMLoRALinear(
            base_linear=base_linear,
            rank=effective_rank,
            lora_alpha=alpha if alpha is not None else effective_rank,
            modulation_scale=modulation_scale,
            mod_channels=mod_channels,
            event_channels_by_size=event_channels_by_size,
            mod_num_heads=mod_num_heads,
            mod_window_size=mod_window_size,
            shared_modulation_heads=shared_heads,
            event_lora_variant=event_lora_variant,
        )
        if lora_a is not None:
            event_lora.lora_down.weight.data.copy_(lora_a)
        if lora_b is not None:
            event_lora.lora_up.weight.data.copy_(lora_b)
        parent, child_name = get_parent_module(unet, name)
        setattr(parent, child_name, event_lora)
        replaced_names.append(name)
    return len(replaced_names), replaced_names


def replace_self_attn_to_v_with_event_film_lora(*args, **kwargs):
    # 兼容旧调用名；默认仍是 film 变体。
    kwargs.setdefault("event_lora_variant", "film")
    return replace_self_attn_to_v_with_event_lora(*args, **kwargs)


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
        if hasattr(module, "shared_modulation_heads"):
            module.shared_modulation_heads.requires_grad_(train_modulation)
        # 兼容旧版 Conv Event-FiLM checkpoint/代码路径。
        for attr in ("event_proj", "z_proj", "modulation_head"):
            if hasattr(module, attr):
                getattr(module, attr).requires_grad_(train_modulation)


def _to_float_value(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    return float(value)


def collect_event_lora_debug(unet):
    stats = {
        "dgamma_abs_mean": [],
        "dbeta_abs_mean": [],
        "dgamma_abs_max": [],
        "dbeta_abs_max": [],
        "z_event_abs_mean": [],
        "z_event_abs_max": [],
        "z_delta_abs_mean": [],
        "z_delta_abs_max": [],
        "base_norm": [],
        "lora_norm": [],
    }
    for _, module in iter_event_film_lora_modules(unet):
        for key in stats:
            if key in module.last_debug:
                stats[key].append(_to_float_value(module.last_debug[key]))
    out = {}
    for key, values in stats.items():
        out[key] = float(sum(values) / len(values)) if values else 0.0
    return out


def collect_event_lora_regularization(unet, device):
    smooth = torch.zeros((), device=device)
    mod = torch.zeros((), device=device)
    count = 0
    seen = set()
    for _, module in iter_event_film_lora_modules(unet):
        if getattr(module, "last_z_delta", None) is not None:
            z_delta = module.last_z_delta.to(device=device, dtype=torch.float32)
            mod = mod + z_delta.pow(2).mean()
            if z_delta.shape[-1] > 1:
                smooth = smooth + (z_delta[..., :, 1:] - z_delta[..., :, :-1]).abs().mean()
            if z_delta.shape[-2] > 1:
                smooth = smooth + (z_delta[..., 1:, :] - z_delta[..., :-1, :]).abs().mean()
            count += 1
        # 共享 head 会被多个 LoRA 层引用；按参数 id 去重，避免正则项重复累加。
        if hasattr(module, "shared_modulation_heads"):
            params = list(module.shared_modulation_heads.parameters())
        else:
            params = []
            for attr in ("modulation_head", "event_proj", "z_proj"):
                if hasattr(module, attr):
                    params += list(getattr(module, attr).parameters())
        for p in params:
            if id(p) in seen:
                continue
            seen.add(id(p))
            mod = mod + p.float().pow(2).mean()
            count += 1
    if count > 0:
        mod = mod / count
        smooth = smooth / count
    return smooth, mod


def collect_event_lora_visuals(unet):
    """
    收集最近一次 forward 的 Event-FiLM 热力图。

    返回的张量仍在当前设备上，只供训练 loop 转成 W&B Image；
    不在这里保存本地文件。
    """
    for _, module in iter_event_film_lora_modules(unet):
        visuals = getattr(module, "last_visuals", None)
        if visuals:
            return {key: value.detach() for key, value in visuals.items()}
    return {}


def event_film_state_keys():
    return (
        "lora_down",
        "lora_up",
        "shared_modulation_heads",
        "_event_lora_shared_heads",
        "_event_film_shared_modulation_heads",
        "event_proj",
        "z_proj",
        "modulation_head",
        "fuse_head",
        "out_proj",
        "core",
        "q_proj",
        "k_proj",
        "v_proj",
        "attn_out",
    )


def _legacy_event_head_key(key):
    """
    兼容 idea2-1 旧 film checkpoint：
    旧 key: _event_film_shared_modulation_heads.heads.32.q_proj.weight
    新 key: _event_lora_shared_heads_film.heads.32.core.q_proj.weight
    """
    if "_event_film_shared_modulation_heads" not in key:
        return None
    mapped = key.replace("_event_film_shared_modulation_heads", "_event_lora_shared_heads_film")
    for name in ("z_proj", "event_proj", "z_pos", "event_pos", "q_proj", "k_proj", "v_proj", "attn_out"):
        marker = f".heads."
        needle = f".{name}."
        if marker in mapped and needle in mapped and f".core.{name}." not in mapped:
            mapped = mapped.replace(needle, f".core.{name}.")
            break
    return mapped


def copy_compatible_state_dict(module, state_dict, prefix_filter=None):
    own_state = module.state_dict()
    loaded, skipped = 0, 0
    for key, value in state_dict.items():
        if prefix_filter is not None and not prefix_filter(key):
            continue
        load_key = key
        if load_key not in own_state:
            legacy_key = _legacy_event_head_key(key)
            if legacy_key in own_state:
                load_key = legacy_key
        if load_key in own_state and own_state[load_key].shape == value.shape:
            own_state[load_key].copy_(value)
            loaded += 1
        else:
            skipped += 1
    return loaded, skipped
