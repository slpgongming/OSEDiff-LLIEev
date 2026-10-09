import copy

import torch
import torch.nn as nn

from models.event_film_lora import (
    EventLoRAConfig,
    SharedResidualEventHeads,
    SpatialRGBEventLoRALinear,
)

SELF_ATTN_PROJECTIONS = {
    "q": "attn1.to_q", "k": "attn1.to_k",
    "v": "attn1.to_v", "out": "attn1.to_out.0",
}
ADAPTER_BY_BLOCK = {
    "down_blocks": "default_encoder",
    "mid_block": "default_others",
    "up_blocks": "default_decoder",
}


def get_parent_module(root, module_name):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def iter_event_lora_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, SpatialRGBEventLoRALinear):
            yield name, module


# 兼容旧调用名称。
iter_event_film_lora_modules = iter_event_lora_modules


def _extract_peft_linear(module, adapter):
    """提取冻结 base Linear 和已加载的 PEFT LoRA A/B，供 Event-LoRA 无损迁移。"""
    if hasattr(module, "base_layer"):
        base_linear = copy.deepcopy(module.base_layer)
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"Event-LoRA needs a Linear base layer, got {type(base_linear)}")
        if not hasattr(module, "lora_A") or set(module.lora_A) != {adapter} or set(module.lora_B) != {adapter}:
            raise ValueError(f"Expected exactly one complete PEFT adapter {adapter}; got {list(getattr(module, 'lora_A', {}))}")
        if getattr(module, "merged", False) or getattr(module, "use_dora", {}).get(adapter, False):
            raise ValueError("Merged or DoRA PEFT layers cannot be converted losslessly.")
        dropouts = getattr(module, "lora_dropout", {})
        dropout = dropouts[adapter] if adapter in dropouts else None
        if dropout is not None and not isinstance(dropout, nn.Identity) and getattr(dropout, "p", 0) != 0:
            raise ValueError("Event-LoRA conversion requires lora_dropout=0.")
        lora_a = module.lora_A[adapter].weight.detach().clone()
        lora_b = module.lora_B[adapter].weight.detach().clone()
        if lora_a.shape != (lora_a.shape[0], base_linear.in_features) or lora_b.shape != (base_linear.out_features, lora_a.shape[0]):
            raise ValueError(f"Invalid PEFT A/B shapes: {tuple(lora_a.shape)}, {tuple(lora_b.shape)}")
        alpha = module.lora_alpha[adapter] if hasattr(module, "lora_alpha") else lora_a.shape[0]
        scaling = float(module.scaling[adapter])
        if abs(float(alpha) / lora_a.shape[0] - scaling) > 1e-6:
            raise ValueError(f"Unsupported PEFT scaling {scaling}; expected alpha/r.")
        return base_linear, lora_a, lora_b, alpha
    if isinstance(module, nn.Linear):
        return copy.deepcopy(module), None, None, None
    raise TypeError(f"Unsupported self-attention projection type: {type(module)}")


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


def replace_self_attn_projections_with_event_lora(unet, config, event_channels_by_size):
    """Validate all selected self-attention PEFT layers before changing the UNet."""
    targets = tuple(config.targets)
    existing = {module.projection for _, module in iter_event_lora_modules(unet)}
    if existing:
        if existing != set(targets):
            raise ValueError(f"Event-LoRA already installed for {sorted(existing)}; cannot switch to {targets}.")
        return 0, []
    candidates = []
    for name, module in list(unet.named_modules()):
        for target in targets:
            suffix = SELF_ATTN_PROJECTIONS[target]
            if name.endswith(suffix) and (name == suffix or name[-len(suffix)-1] == "."):
                block = name.split(".", 1)[0]
                if block not in ADAPTER_BY_BLOCK and not isinstance(module, nn.Linear):
                    raise ValueError(f"Cannot identify PEFT adapter for {name}.")
                base, lora_a, lora_b, alpha = _extract_peft_linear(module, ADAPTER_BY_BLOCK.get(block))
                candidates.append((name, module, target, base, lora_a, lora_b, alpha))
                break
    if targets and not candidates:
        raise ValueError(f"No self-attention projections found for {targets}.")
    found = {item[2] for item in candidates}
    if set(targets) != found:
        raise ValueError(f"Missing selected Event-LoRA projections: {set(targets) - found}.")

    shared = {}
    pending_heads = {}
    if config.variant in {"residual_cross_attn", "residual_cross_attn_bax"}:
        for target in targets:
            rows = [item for item in candidates if item[2] == target]
            ranks = {item[4].shape[0] if item[4] is not None else config.rank for item in rows}
            out_channels = {item[3].out_features for item in rows}
            if len(ranks) != 1:
                raise ValueError(f"Shared head dimensions differ for {target}: ranks={ranks}, outputs={out_channels}.")
            if target == "v":
                attr = f"_event_lora_shared_heads_{config.variant}"
            else:
                attr = f"_event_lora_shared_heads_{config.variant}_{target}"
            head_config = EventLoRAConfig(**{**config.to_dict(), "rank": ranks.pop()})
            query_channels = None
            if config.variant == "residual_cross_attn_bax":
                query_channels = _infer_bax_query_channels_by_size(unet, event_channels_by_size)
                if not set(out_channels).issubset(set(query_channels.values())):
                    raise ValueError(f"BAx shared head cannot represent {target} outputs {out_channels}.")
            head = SharedResidualEventHeads(head_config, event_channels_by_size, query_channels)
            head.to(device=rows[0][3].weight.device, dtype=rows[0][3].weight.dtype)
            pending_heads[attr] = head
            shared[target] = head

    prepared = []
    for name, original, target, base, lora_a, lora_b, alpha in candidates:
        effective_rank = lora_a.shape[0] if lora_a is not None else config.rank
        layer_config = EventLoRAConfig(**{
            **config.to_dict(), "rank": effective_rank,
            "alpha": float(alpha if alpha is not None else effective_rank),
        })
        event_lora = SpatialRGBEventLoRALinear(
            base, layer_config, event_channels_by_size,
            shared_residual_heads=shared.get(target), projection=target,
        )
        event_lora.to(device=base.weight.device, dtype=base.weight.dtype)
        if lora_a is not None:
            event_lora.lora_down.weight.data.copy_(lora_a)
            event_lora.lora_up.weight.data.copy_(lora_b)
        prepared.append((name, event_lora))
    for attr, head in pending_heads.items():
        setattr(unet, attr, head)
    replaced_names = []
    for name, event_lora in prepared:
        parent, child = get_parent_module(unet, name)
        if child.isdigit():
            parent[int(child)] = event_lora
        else:
            setattr(parent, child, event_lora)
        replaced_names.append(name)
    return len(replaced_names), replaced_names


def replace_self_attn_to_v_with_event_lora(unet, config, event_channels_by_size):
    """Legacy V-only API, independent of the caller's new target selection."""
    v_config = EventLoRAConfig(**{**config.to_dict(), "targets": ("v",)})
    return replace_self_attn_projections_with_event_lora(unet, v_config, event_channels_by_size)


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


def migrate_event_lora_state(unet, source_state, source_targets, destination_targets,
                             source_layer_metadata=None):
    """Map A/B between PEFT and Event projections at the same logical path."""
    migrated = dict(source_state)
    source_layer_metadata = source_layer_metadata or {}
    report = {"ordinary_to_event": [], "event_to_ordinary": [], "event_to_event": []}
    for name, module in unet.named_modules():
        target = getattr(module, "projection", None)
        if not isinstance(module, SpatialRGBEventLoRALinear):
            target = next((key for key, suffix in SELF_ATTN_PROJECTIONS.items()
                           if name.endswith(suffix)), None)
            if target not in source_targets or target in destination_targets or not hasattr(module, "lora_A"):
                continue
            pairs = (("lora_down.weight", "lora_A"), ("lora_up.weight", "lora_B"))
            direction = "event_to_ordinary"
        elif target not in source_targets:
            pairs = (("lora_A", "lora_down.weight"), ("lora_B", "lora_up.weight"))
            direction = "ordinary_to_event"
        else:
            report["event_to_event"].append(name)
            continue
        block = name.split(".", 1)[0]
        if block not in ADAPTER_BY_BLOCK:
            raise ValueError(f"Unknown adapter block for migration: {name}")
        adapter = ADAPTER_BY_BLOCK[block]
        if direction == "event_to_ordinary" and name in source_layer_metadata:
            source = source_layer_metadata[name]
            rank = module.lora_A[adapter].weight.shape[0]
            scaling = float(module.scaling[adapter])
            if int(source["rank"]) != rank or abs(float(source["scaling"]) - scaling) > 1e-6:
                raise ValueError(f"Event-to-ordinary rank/scaling mismatch at {name}: checkpoint={source}, runtime={(rank, scaling)}")
        for source_part, dest_part in pairs:
            source_suffix = f"{source_part}.{adapter}.weight" if source_part in ("lora_A", "lora_B") else source_part
            dest_suffix = f"{dest_part}.{adapter}.weight" if dest_part in ("lora_A", "lora_B") else dest_part
            source_key, dest_key = f"{name}.{source_suffix}", f"{name}.{dest_suffix}"
            if source_key not in source_state:
                raise RuntimeError(f"Cannot migrate LoRA: missing {source_key}")
            migrated[dest_key] = source_state[source_key]
        report[direction].append(name)
    own = unet.state_dict()
    for key in set(migrated) & set(own):
        if own[key].shape != migrated[key].shape:
            raise RuntimeError(f"LoRA migration shape mismatch at {key}: {tuple(migrated[key].shape)} -> {tuple(own[key].shape)}")
    report["discarded_event_keys"] = [key for key in migrated if key not in own and (
        ".modulator." in key or "_event_lora_shared_heads_" in key
    )]
    return {key: value for key, value in migrated.items() if key in own}, report


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
