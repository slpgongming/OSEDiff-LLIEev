import types
import unittest

import torch
import torch.nn as nn

from models.event_film_lora import (
    EVENT_LORA_VARIANTS,
    EventLoRAConfig,
)
from models.event_lora_utils import replace_self_attn_to_v_with_event_lora
from models.event_self_swap_guidance import (
    EventSelfSwapConfig,
    swap_event_feature_tokens,
)
from osediff_event_film import DenoisingContext, OSEDiffEventFiLMGen


def _row_multiset(tensor):
    return sorted(tuple(float(value) for value in row) for row in tensor.tolist())


class EventSelfSwapTokenTest(unittest.TestCase):
    def setUp(self):
        # 四个 spatial token 的相似性关系清晰，便于验证 pair policy。
        self.feature = torch.tensor(
            [[[[1.0, 0.9], [-1.0, 0.0]], [[0.0, 0.1], [0.0, -1.0]]]]
        )
        self.features = {"2": self.feature, "1": torch.randn(1, 2, 1, 1)}

    def _config(self, **overrides):
        values = {
            "guidance_scale": 0.25,
            "swap_ratio": 0.5,
            "swap_mode": "spatial",
            "pair_policy": "dissimilar",
            "feature_scales": ("2",),
            "seed": 7,
        }
        values.update(overrides)
        return EventSelfSwapConfig(**values)

    def test_spatial_swap_is_deterministic_non_mutating_and_preserves_tokens(self):
        original = {key: value.clone() for key, value in self.features.items()}
        swapped_a, stats_a = swap_event_feature_tokens(
            self.features, self._config()
        )
        swapped_b, stats_b = swap_event_feature_tokens(
            self.features, self._config()
        )
        self.assertTrue(torch.equal(self.features["2"], original["2"]))
        self.assertIs(swapped_a["1"], self.features["1"])
        self.assertTrue(torch.equal(swapped_a["2"], swapped_b["2"]))
        self.assertEqual(stats_a, stats_b)
        self.assertEqual(swapped_a["2"].shape, self.feature.shape)
        self.assertEqual(swapped_a["2"].dtype, self.feature.dtype)
        before = self.feature.flatten(2).transpose(1, 2)[0]
        after = swapped_a["2"].flatten(2).transpose(1, 2)[0]
        self.assertEqual(_row_multiset(before), _row_multiset(after))

    def test_channel_swap_preserves_channel_maps(self):
        feature = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3)
        swapped, stats = swap_event_feature_tokens(
            {"2": feature},
            self._config(swap_mode="channel", swap_ratio=1.0),
        )
        self.assertEqual(
            _row_multiset(feature.flatten(2)[0]),
            _row_multiset(swapped["2"].flatten(2)[0]),
        )
        self.assertEqual(stats["scale_2/channel_swap_fraction"], 1.0)

    def test_dissimilar_pairs_have_lower_similarity_than_similar_pairs(self):
        _, dissimilar = swap_event_feature_tokens(
            self.features, self._config(pair_policy="dissimilar")
        )
        _, similar = swap_event_feature_tokens(
            self.features, self._config(pair_policy="similar")
        )
        self.assertLess(
            dissimilar["scale_2/spatial_pair_similarity"],
            similar["scale_2/spatial_pair_similarity"],
        )

    def test_zero_ratio_and_too_few_tokens_are_identity(self):
        zero, zero_stats = swap_event_feature_tokens(
            self.features, self._config(swap_ratio=0.0)
        )
        tiny, tiny_stats = swap_event_feature_tokens(
            {"1": self.features["1"]},
            self._config(feature_scales=("1",), swap_ratio=1.0),
        )
        self.assertTrue(torch.equal(zero["2"], self.features["2"]))
        self.assertEqual(zero_stats["scale_2/spatial_swap_fraction"], 0.0)
        self.assertTrue(torch.equal(tiny["1"], self.features["1"]))
        self.assertEqual(tiny_stats["scale_1/spatial_swap_fraction"], 0.0)

    def test_missing_scale_and_invalid_config_fail_early(self):
        with self.assertRaises(KeyError):
            swap_event_feature_tokens(
                self.features, self._config(feature_scales=("8",))
            )
        with self.assertRaises(ValueError):
            self._config(swap_ratio=1.1)


class _Scheduler:
    def __init__(self):
        self.calls = []

    def step(self, model_pred, timesteps, sample, return_dict=True):
        self.calls.append((model_pred.clone(), timesteps.clone(), sample.clone()))
        return types.SimpleNamespace(prev_sample=sample - model_pred)


def _mock_generator():
    generator = object.__new__(OSEDiffEventFiLMGen)
    nn.Module.__init__(generator)
    generator.noise_scheduler = _Scheduler()
    generator.calls = []

    def run_unet(self, latent, timesteps, prompt, event_features):
        self.calls.append(
            {
                "latent": latent.clone(),
                "timesteps": timesteps.clone(),
                "prompt": prompt.clone(),
                "events": {key: value.clone() for key, value in event_features.items()},
            }
        )
        event_signal = event_features["2"].mean(dim=(1, 2, 3), keepdim=True)
        return latent + event_signal

    generator._run_unet_with_event_features = types.MethodType(run_unet, generator)
    return generator


class ESGModelPathTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.context = DenoisingContext(
            encoded_control=torch.randn(2, 4, 2, 2),
            prompt_embeds=torch.randn(2, 3, 4),
            neg_prompt_embeds=torch.randn(2, 3, 4),
            event_feats={"2": torch.randn(2, 4, 2, 2)},
        )
        self.timesteps = torch.tensor([500])

    def _config(self, execution_mode):
        return EventSelfSwapConfig(
            guidance_scale=0.25,
            swap_ratio=0.5,
            swap_mode="spatial",
            pair_policy="dissimilar",
            feature_scales=("2",),
            seed=9,
            execution_mode=execution_mode,
        )

    def test_scale_zero_uses_one_unet_call_and_original_scheduler_sample(self):
        generator = _mock_generator()
        output, stats = generator.predict_denoised_latent_from_context(
            self.context,
            self.timesteps,
            EventSelfSwapConfig(
                guidance_scale=0.0, feature_scales=("2",)
            ),
        )
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(generator.noise_scheduler.calls), 1)
        self.assertTrue(
            torch.equal(
                generator.noise_scheduler.calls[0][2],
                self.context.encoded_control,
            )
        )
        self.assertEqual(stats["mean_abs_delta"], 0.0)
        self.assertTrue(torch.isfinite(output).all())

    def test_batched_and_sequential_match_and_share_positive_inputs(self):
        batched = _mock_generator()
        sequential = _mock_generator()
        batched_output, _ = batched.predict_denoised_latent_from_context(
            self.context, self.timesteps, self._config("batched")
        )
        sequential_output, _ = sequential.predict_denoised_latent_from_context(
            self.context, self.timesteps, self._config("sequential")
        )
        self.assertTrue(torch.allclose(batched_output, sequential_output))
        paired_call = batched.calls[0]
        positive_latent = paired_call["latent"][:2]
        self.assertTrue(
            torch.equal(positive_latent, paired_call["latent"][2:])
        )
        self.assertTrue(torch.equal(positive_latent, self.context.encoded_control))
        self.assertTrue(
            torch.equal(paired_call["prompt"][:2], paired_call["prompt"][2:])
        )
        self.assertEqual(len(batched.noise_scheduler.calls), 1)
        self.assertEqual(len(sequential.noise_scheduler.calls), 1)


class _AttentionOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_v = nn.Linear(4, 4, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _AttentionOne()


class _TinyEventUNet(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.block = _TinyBlock()
        self.config = types.SimpleNamespace(block_out_channels=(4, 4, 4, 4))
        self.fail = fail

    def forward(self, sample, timesteps, encoder_hidden_states):
        if self.fail:
            raise RuntimeError("intentional failure")
        hidden = sample.flatten(2).transpose(1, 2)
        output = self.block.attn1.to_v(hidden)
        return types.SimpleNamespace(
            sample=output.transpose(1, 2).reshape_as(sample)
        )


class ESGEventLoRACacheTest(unittest.TestCase):
    def _build_generator(self, variant, fail=False):
        generator = object.__new__(OSEDiffEventFiLMGen)
        nn.Module.__init__(generator)
        generator.unet = _TinyEventUNet(fail=fail)
        config = EventLoRAConfig(
            variant=variant,
            rank=4,
            alpha=4,
            mod_channels=8,
            mod_num_heads=2,
        )
        event_channels = {"8": 2, "4": 2, "2": 2, "1": 2}
        replace_self_attn_to_v_with_event_lora(generator.unet, config, event_channels)
        nn.init.normal_(generator.unet.block.attn1.to_v.lora_up.weight, std=0.1)
        generator.noise_scheduler = _Scheduler()
        return generator

    def test_all_variants_smoke_and_clear_cache(self):
        latent = torch.randn(2, 4, 2, 2)
        prompts = torch.randn(2, 3, 4)
        events = {
            "8": torch.randn(2, 2, 8, 8),
            "4": torch.randn(2, 2, 4, 4),
            "2": torch.randn(2, 2, 2, 2),
            "1": torch.randn(2, 2, 1, 1),
        }
        for variant in EVENT_LORA_VARIANTS:
            with self.subTest(variant=variant):
                generator = self._build_generator(variant)
                context = DenoisingContext(
                    encoded_control=latent,
                    prompt_embeds=prompts,
                    neg_prompt_embeds=torch.zeros_like(prompts),
                    event_feats=events,
                )
                output, stats = generator.predict_denoised_latent_from_context(
                    context,
                    torch.tensor([500]),
                    EventSelfSwapConfig(
                        guidance_scale=0.25,
                        swap_ratio=0.5,
                        feature_scales=("2",),
                        execution_mode="batched",
                    ),
                )
                self.assertTrue(torch.isfinite(output).all())
                self.assertIn("mean_abs_delta", stats)
                self.assertEqual(len(generator.noise_scheduler.calls), 1)
                self.assertIsNone(generator.unet.block.attn1.to_v.event_feats)

    def test_cache_is_cleared_when_unet_raises(self):
        generator = self._build_generator("film_affine", fail=True)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            generator._run_unet_with_event_features(
                torch.randn(1, 4, 2, 2),
                torch.tensor([500]),
                torch.randn(1, 3, 4),
                {
                    "8": torch.randn(1, 2, 8, 8),
                    "4": torch.randn(1, 2, 4, 4),
                    "2": torch.randn(1, 2, 2, 2),
                    "1": torch.randn(1, 2, 1, 1),
                },
            )
        self.assertIsNone(generator.unet.block.attn1.to_v.event_feats)


if __name__ == "__main__":
    unittest.main()
