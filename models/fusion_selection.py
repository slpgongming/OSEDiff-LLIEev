"""训练与推理共用的 decoder 执行层选择；d0 是最深层，d3 是最高分辨率层。"""


def parse_fusion_layers(value):
    """接受 d0,d2 / 0,2 / all / none；拒绝重复和越界，避免静默改变实验。"""
    if isinstance(value, str):
        if value == 'all':
            return (0, 1, 2, 3)
        if value == 'none':
            return ()
        parts = value.split(',')
        try:
            levels = tuple(int(p.strip().removeprefix('d')) for p in parts)
        except ValueError as exc:
            raise ValueError('Use d0,d1,d2,d3, all, or none.') from exc
    else:
        levels = tuple(value)
    if len(set(levels)) != len(levels) or any(type(i) is not int or i not in range(4) for i in levels):
        raise ValueError(f'Invalid decoder fusion layers: {value!r}')
    return tuple(sorted(levels))


def select_fusion_layers(decoder, levels, *, trainable=False):
    """旁路未选层，训练时仅解冻被选层；四层参数均保留以兼容已有权重。"""
    levels = parse_fusion_layers(levels)
    decoder.set_inference_fusion_levels(levels)
    if trainable:
        decoder.feature_fusion_blocks.requires_grad_(False)
        for index in levels:
            decoder.feature_fusion_blocks[index].requires_grad_(True)
    return levels
