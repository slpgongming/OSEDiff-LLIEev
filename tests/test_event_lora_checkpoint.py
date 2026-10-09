"""CPU tests for projection-aware A/B transfer and structural reload."""

import unittest

import torch

from models.event_film_lora import EventLoRAConfig
from models.event_lora_utils import (
    copy_compatible_state_dict,
    migrate_event_lora_state,
    replace_self_attn_projections_with_event_lora,
)
from test_event_lora_targets import FakeUNet


def build_unet(targets):
    model = FakeUNet().eval()
    config = EventLoRAConfig(
        targets=targets, variant="residual_cross_attn", rank=2,
        alpha=4, mod_channels=4, mod_num_heads=2,
    )
    replace_self_attn_projections_with_event_lora(model, config, {"2": 3})
    return model


class EventLoRACheckpointTests(unittest.TestCase):
    def test_same_structure_state_round_trip(self):
        torch.manual_seed(10)
        original = build_unet(("q", "v"))
        torch.manual_seed(11)
        restored = build_unet(("q", "v"))
        state = original.state_dict()
        loaded, skipped = copy_compatible_state_dict(restored, state)
        self.assertEqual(skipped, 0)
        self.assertEqual(loaded, len(state))
        for key, value in state.items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_v_to_qv_and_qv_to_v_transfer(self):
        for source_targets, dest_targets in ((("v",), ("q", "v")),
                                             (("q", "v"), ("v",))):
            with self.subTest(source=source_targets, dest=dest_targets):
                source = build_unet(source_targets)
                destination = build_unet(dest_targets)
                source_state = source.state_dict()
                migrated, report = migrate_event_lora_state(
                    destination, source_state, source_targets, dest_targets
                )
                for name in ("down_blocks.0.attn1.to_q", "mid_block.attn1.to_q",
                             "up_blocks.0.attn1.to_q"):
                    if "q" in dest_targets:
                        adapter = ("default_encoder" if name.startswith("down") else
                                   "default_decoder" if name.startswith("up") else "default_others")
                        torch.testing.assert_close(
                            migrated[f"{name}.lora_down.weight"],
                            source_state[f"{name}.lora_A.{adapter}.weight"],
                        )
                    else:
                        adapter = ("default_encoder" if name.startswith("down") else
                                   "default_decoder" if name.startswith("up") else "default_others")
                        torch.testing.assert_close(
                            migrated[f"{name}.lora_A.{adapter}.weight"],
                            source_state[f"{name}.lora_down.weight"],
                        )
                loaded, skipped = copy_compatible_state_dict(destination, migrated)
                self.assertGreater(loaded, 0)
                self.assertEqual(skipped, 0)
                self.assertEqual(len(report["ordinary_to_event"] if "q" in dest_targets
                                     else report["event_to_ordinary"]), 3)

    def test_missing_ab_fails_migration(self):
        source = build_unet(("v",))
        destination = build_unet(("q", "v"))
        state = source.state_dict()
        state.pop("down_blocks.0.attn1.to_q.lora_A.default_encoder.weight")
        with self.assertRaisesRegex(RuntimeError, "missing"):
            migrate_event_lora_state(destination, state, ("v",), ("q", "v"))

    def test_event_to_ordinary_rejects_scaling_mismatch(self):
        source = build_unet(("q", "v"))
        destination = build_unet(("v",))
        metadata = {"down_blocks.0.attn1.to_q": {"rank": 2, "scaling": 3.0}}
        with self.assertRaisesRegex(ValueError, "rank/scaling"):
            migrate_event_lora_state(destination, source.state_dict(),
                                     ("q", "v"), ("v",), metadata)


if __name__ == "__main__":
    unittest.main()
