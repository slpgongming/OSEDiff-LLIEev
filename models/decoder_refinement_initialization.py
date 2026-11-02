"""为 H1 开/关消融加载相同的 decoder fusion 初始权重与初始化后的随机状态。"""
import random
import numpy as np
import torch
from models.rgb_event_encoder import sha256_file
from models.decoder_refinement import normalize_refinement_variant


def load_refinement_initialization(model, path, args):
    """只加载融合模块，不覆盖 Stage3 或 RGBEventEncoder；不恢复任何优化器状态。"""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') != 'paired-stage4-refinement-init-v1':
        raise ValueError('Unknown paired refinement initialization format.')
    if saved['seed'] != args.seed or saved['stage3_sha256'] != sha256_file(args.event_lora_checkpoint):
        raise ValueError('Paired initialization seed/Stage3 mismatch.')
    # 旧 rgb_event_decoder_fusion 元数据含无效 CGDRS 字段，忽略它们而保留结构检查。
    topology={k:v for k,v in saved['decoder_refinement_config'].items() if not k.startswith('cgdrs_')}
    topology['variant'] = normalize_refinement_variant(topology.get('variant', 'rgb_event_decoder_fusion'))
    if topology != model.vae.decoder.refinement_config():
        raise ValueError('Paired initialization decoder topology mismatch.')
    if saved['rgb_event_manifest']['sha256'] != model.rgb_event_encoder.pretrained_manifest['sha256']:
        raise ValueError('Paired initialization RGBEventEncoder weights mismatch.')
    model.vae.decoder.feature_fusion_blocks.load_state_dict(saved['state_dict'], strict=True)
    # 加载H1与否会消耗不同数量的初始化随机数；恢复到H1融合模块初始化完成后的状态。
    torch.set_rng_state(saved['torch_rng_cpu'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(saved['torch_rng_cuda'], device=next(model.parameters()).device)
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    return dict(path=path, sha256=sha256_file(path), state_sha256=saved['state_sha256'],
                source='reconstructed_h1_initialization', rng_restored=True)
