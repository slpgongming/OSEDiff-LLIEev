import types
import unittest
import os

import torch
import torch.nn as nn

from models.event_film_lora import EVENT_LORA_VARIANTS, EventLoRAConfig
from models.event_lora_utils import (
    clear_event_feats_for_all_event_lora,
    collect_event_lora_debug,
    copy_compatible_state_dict,
    remap_legacy_event_film_state_dict,
    replace_self_attn_to_v_with_event_lora,
    set_event_feats_for_all_event_lora,
    set_modulation_enabled_for_all_event_lora,
)
from models.event_film_lora import SpatialRGBEventLoRALinear


class _AttentionOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_v = nn.Linear(4, 4, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _AttentionOne()


class _TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()
        self.config = types.SimpleNamespace(block_out_channels=(4, 4, 4, 4))


class Idea3CheckpointTest(unittest.TestCase):

    def test_all_event_lora_variants_forward_backward_and_reload(self):
        events = {
            str(size): torch.randn(2, 2, size, size)
            for size in (32, 16, 8, 4)
        }
        for variant in EVENT_LORA_VARIANTS:
            with self.subTest(variant=variant):
                config = EventLoRAConfig(
                    variant=variant,
                    rank=4,
                    alpha=4,
                    mod_channels=8,
                    mod_num_heads=2,
                )
                source = _TinyUNet()
                count, _ = replace_self_attn_to_v_with_event_lora(
                    source, config, {key: 2 for key in events}
                )
                self.assertEqual(count, 1)
                nn.init.normal_(source.block.attn1.to_v.lora_up.weight, std=0.1)
                set_event_feats_for_all_event_lora(source, events)
                hidden = torch.randn(2, 16, 4, requires_grad=True)
                output = source.block.attn1.to_v(hidden)
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsNotNone(hidden.grad)
                self.assertTrue(
                    all(
                        torch.isfinite(torch.tensor(value))
                        for value in collect_event_lora_debug(source).values()
                    )
                )
                for ablation_events in (
                    {key: torch.zeros_like(value) for key, value in events.items()},
                    {key: value.flip(0) for key, value in events.items()},
                ):
                    set_event_feats_for_all_event_lora(source, ablation_events)
                    ablation_output = source.block.attn1.to_v(hidden.detach())
                    self.assertTrue(torch.isfinite(ablation_output).all())
                set_modulation_enabled_for_all_event_lora(source, False)
                disabled_output = source.block.attn1.to_v(hidden.detach())
                self.assertTrue(torch.isfinite(disabled_output).all())
                set_modulation_enabled_for_all_event_lora(source, True)
                clear_event_feats_for_all_event_lora(source)

                target = _TinyUNet()
                replace_self_attn_to_v_with_event_lora(
                    target, config, {key: 2 for key in events}
                )
                loaded, skipped = copy_compatible_state_dict(
                    target, source.state_dict()
                )
                self.assertEqual(skipped, 0)
                self.assertEqual(loaded, len(source.state_dict()))


if __name__ == "__main__":
    unittest.main()
