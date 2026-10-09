"""CPU checks for exact multi-projection replacement and legacy V behavior."""

import itertools
import unittest

import torch
from torch import nn

from models.event_film_lora import EventLoRAConfig, parse_event_lora_targets
from models.event_lora_utils import (
    replace_self_attn_projections_with_event_lora,
    replace_self_attn_to_v_with_event_lora,
    iter_event_lora_modules,
    set_event_feats_for_all_event_lora,
    clear_event_feats_for_all_event_lora,
    set_event_lora_trainable,
)


class FakePEFTLinear(nn.Module):
    def __init__(self, adapter, features=4, rank=2):
        super().__init__()
        self.base_layer = nn.Linear(features, features)
        self.lora_A = nn.ModuleDict({adapter: nn.Linear(features, rank, bias=False)})
        self.lora_B = nn.ModuleDict({adapter: nn.Linear(rank, features, bias=False)})
        self.lora_dropout = nn.ModuleDict({adapter: nn.Identity()})
        self.lora_alpha = {adapter: 4}
        self.scaling = {adapter: 2.0}
        self.adapter = adapter
        nn.init.normal_(self.lora_A[adapter].weight, std=0.1)
        nn.init.normal_(self.lora_B[adapter].weight, std=0.1)

    def forward(self, value):
        return self.base_layer(value) + self.lora_B[self.adapter](
            self.lora_A[self.adapter](value)
        ) * self.scaling[self.adapter]


class FakeAttention(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        for target in ("q", "k", "v"):
            setattr(self, f"to_{target}", FakePEFTLinear(adapter))
        self.to_out = nn.Sequential(FakePEFTLinear(adapter), nn.Dropout(0.2))

    def forward(self, value):
        return self.to_out(self.to_q(value) + self.to_k(value) + self.to_v(value))


class FakeBlock(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.attn1 = FakeAttention(adapter)
        self.attn2 = FakeAttention(adapter)


class FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([FakeBlock("default_encoder")])
        self.mid_block = FakeBlock("default_others")
        self.up_blocks = nn.ModuleList([FakeBlock("default_decoder")])
        self.config = type("Config", (), {"block_out_channels": (4,)})()

    def forward(self, value):
        for block in (*self.down_blocks, self.mid_block, *self.up_blocks):
            value = block.attn1(value)
        return value


class EventLoRATargetTests(unittest.TestCase):
    def test_parse_all_subsets_and_invalid_values(self):
        targets = ("q", "k", "v", "out")
        for n in range(5):
            for subset in itertools.combinations(targets, n):
                self.assertEqual(parse_event_lora_targets(",".join(subset) if subset else "none"), subset)
        self.assertEqual(parse_event_lora_targets(" V, q "), ("q", "v"))
        self.assertEqual(parse_event_lora_targets("auto", allow_auto=True), "auto")
        for value in ("q,q", "all,q", "auto,v", "attn2", ""):
            with self.assertRaises(ValueError):
                parse_event_lora_targets(value, allow_auto=True)

    def test_all_target_subsets_replace_only_attn1(self):
        for n in range(5):
            for targets in itertools.combinations(("q", "k", "v", "out"), n):
                with self.subTest(targets=targets):
                    unet = FakeUNet()
                    dropout = unet.down_blocks[0].attn1.to_out[1]
                    config = EventLoRAConfig(targets=targets, variant="film_affine", rank=2, alpha=4)
                    count, names = replace_self_attn_projections_with_event_lora(unet, config, {"2": 3})
                    self.assertEqual(count, 3 * len(targets))
                    self.assertEqual(len(names), count)
                    self.assertTrue(all("attn1." in name for name in names))
                    self.assertIs(unet.down_blocks[0].attn1.to_out[1], dropout)
                    self.assertIsInstance(unet.down_blocks[0].attn2.to_v, FakePEFTLinear)
                    self.assertEqual(len(list(iter_event_lora_modules(unet))), count)

    def test_legacy_v_and_shared_target_heads(self):
        unet = FakeUNet()
        config = EventLoRAConfig(targets=("q", "k"), variant="residual_cross_attn", rank=2,
                                 alpha=4, mod_channels=4, mod_num_heads=2)
        count, names = replace_self_attn_to_v_with_event_lora(unet, config, {"2": 3})
        self.assertEqual(count, 3)
        self.assertTrue(all(name.endswith("attn1.to_v") for name in names))
        with self.assertRaises(ValueError):
            replace_self_attn_projections_with_event_lora(unet, config, {"2": 3})

        unet = FakeUNet()
        config = EventLoRAConfig(targets=("q", "v"), variant="residual_cross_attn", rank=2,
                                 alpha=4, mod_channels=4, mod_num_heads=2)
        replace_self_attn_projections_with_event_lora(unet, config, {"2": 3})
        q = unet.down_blocks[0].attn1.to_q
        v = unet.down_blocks[0].attn1.to_v
        self.assertIs(q.shared_residual_heads, unet.mid_block.attn1.to_q.shared_residual_heads)
        self.assertIsNot(q.shared_residual_heads, v.shared_residual_heads)
        self.assertAlmostEqual(q.scaling, 2.0)
        event = {"2": torch.randn(1, 3, 2, 2)}
        set_event_feats_for_all_event_lora(unet, event)
        try:
            result = unet(torch.randn(1, 4, 4, requires_grad=True)).square().mean()
            result.backward()
            self.assertIsNotNone(q.lora_down.weight.grad)
            self.assertIsNotNone(q.shared_residual_heads.heads["2"].out_proj[-1].weight.grad)
        finally:
            clear_event_feats_for_all_event_lora(unet)
        self.assertIsNone(q.event_feats)

    def test_five_variants_keep_zero_initialized_forward_and_backward(self):
        for variant in (
            "film_affine", "cbam_lowrank_residual", "cbam_output_gate",
            "residual_cross_attn", "residual_cross_attn_bax",
        ):
            for targets in (("q", "v"), ("q", "k", "v", "out")):
                with self.subTest(variant=variant, targets=targets):
                    unet = FakeUNet().eval()
                    value = torch.randn(1, 4, 4)
                    baseline = unet(value).detach()
                    config = EventLoRAConfig(
                        targets=targets, variant=variant, rank=2, alpha=4,
                        mod_channels=4, mod_num_heads=2,
                    )
                    replace_self_attn_projections_with_event_lora(unet, config, {"2": 3})
                    set_event_feats_for_all_event_lora(unet, {"2": torch.randn(1, 3, 2, 2)})
                    try:
                        result = unet(value)
                        torch.testing.assert_close(result, baseline, rtol=1e-5, atol=1e-5)
                        result.square().mean().backward()
                        first = next(iter_event_lora_modules(unet))[1]
                        self.assertIsNotNone(first.lora_down.weight.grad)
                    finally:
                        clear_event_feats_for_all_event_lora(unet)

    def test_event_ab_freeze_is_independent_of_ordinary_adapter(self):
        unet = FakeUNet()
        config = EventLoRAConfig(targets=("q", "v"), variant="film_affine", rank=2)
        replace_self_attn_projections_with_event_lora(unet, config, {"2": 3})
        ordinary = unet.down_blocks[0].attn1.to_k
        ordinary.lora_A[ordinary.adapter].requires_grad_(False)
        set_event_lora_trainable(unet, train_lora=False, train_modulation=True)
        for _, module in iter_event_lora_modules(unet):
            self.assertFalse(module.lora_down.weight.requires_grad)
            self.assertTrue(any(param.requires_grad for param in module.modulator.parameters()))
        self.assertFalse(ordinary.lora_A[ordinary.adapter].weight.requires_grad)
        set_event_lora_trainable(unet, train_lora=True, train_modulation=True)
        self.assertTrue(all(module.lora_down.weight.requires_grad
                            for _, module in iter_event_lora_modules(unet)))
        self.assertFalse(ordinary.lora_A[ordinary.adapter].weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
