"""从旧指导测试迁移的公共保证：普通预测、原 latent 和事件缓存生命周期。"""

import types
import pytest
import torch
from torch import nn

from models.event_film_lora import EVENT_LORA_VARIANTS, EventLoRAConfig
from models.event_lora_utils import replace_self_attn_to_v_with_event_lora, clear_event_feats_for_all_event_lora
from osediff_event_film import DenoisingContext, OSEDiffEventFiLMGen


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = nn.Module()
        self.attn1.to_v = nn.Linear(4, 4, bias=False)
        self.config = types.SimpleNamespace(block_out_channels=(4, 4, 4, 4))
        self.fail = False
        self.calls = 0

    def forward(self, sample, timesteps, encoder_hidden_states):
        self.calls += 1
        if self.fail:
            raise RuntimeError("intentional failure")
        tokens = self.attn1.to_v(sample.flatten(2).transpose(1, 2))
        return types.SimpleNamespace(sample=tokens.transpose(1, 2).reshape_as(sample))


def make_gen(variant):
    gen = object.__new__(OSEDiffEventFiLMGen)
    nn.Module.__init__(gen)
    gen.unet = TinyUNet()
    replace_self_attn_to_v_with_event_lora(gen.unet, EventLoRAConfig(variant=variant, rank=4,
        alpha=4, mod_channels=8, mod_num_heads=2), {"8": 2, "4": 2, "2": 2, "1": 2})
    gen.scheduler_inputs = []
    def step(pred, t, sample, return_dict=True):
        gen.scheduler_inputs.append(sample)
        return types.SimpleNamespace(prev_sample=sample-pred)
    gen.noise_scheduler = types.SimpleNamespace(step=step)
    return gen


def make_context():
    return DenoisingContext(torch.randn(1, 4, 2, 2), torch.randn(1, 3, 4), torch.randn(1, 3, 4),
                           {str(size): torch.randn(1, 2, size, size) for size in (8, 4, 2, 1)})


@pytest.mark.parametrize("variant", EVENT_LORA_VARIANTS)
def test_plain_prediction_one_unet_original_latent_and_cleanup(variant):
    gen, context = make_gen(variant), make_context()
    output, stats = gen.predict_denoised_latent_from_context(context, torch.tensor([500]))
    assert gen.unet.calls == 1 and gen.scheduler_inputs[0] is context.encoded_control
    assert torch.isfinite(output).all() and stats == {}
    assert gen.unet.attn1.to_v.event_feats is None


def test_exception_cleanup_and_training_checkpoint_retention():
    gen, context = make_gen("residual_cross_attn"), make_context()
    gen.unet.fail = True
    with pytest.raises(RuntimeError, match="intentional"):
        gen.predict_denoised_latent_from_context(context, torch.tensor([500]))
    assert gen.unet.attn1.to_v.event_feats is None
    gen.unet.fail = False
    gen.unet.gradient_checkpointing = True
    gen.predict_denoised_latent_from_context(context, torch.tensor([500]))
    assert gen.unet.attn1.to_v.event_feats is context.event_feats
    clear_event_feats_for_all_event_lora(gen.unet)
    with torch.no_grad():
        gen.predict_denoised_latent_from_context(context, torch.tensor([500]))
    assert gen.unet.attn1.to_v.event_feats is None
