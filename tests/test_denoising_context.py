"""固定编码上下文、scheduler 输入以及 Event-LoRA 缓存生命周期回归。"""

import types
import unittest

import torch
import torch.nn as nn

from models.event_film_lora import EVENT_LORA_VARIANTS, EventLoRAConfig
from models.event_lora_utils import replace_self_attn_to_v_with_event_lora
from osediff_event_film import DenoisingContext, OSEDiffEventFiLMGen


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

    def run_unet(self, latent, timesteps, prompt, event_features, feature_guidance=None):
        self.calls.append({
            "latent": latent.clone(), "timesteps": timesteps.clone(),
            "prompt": prompt.clone(),
            "events": {key: value.clone() for key, value in event_features.items()},
        })
        event_signal = event_features["2"].mean(dim=(1, 2, 3), keepdim=True)
        return latent + event_signal

    generator._run_unet_with_event_features = types.MethodType(run_unet, generator)
    return generator


class DenoisingContextTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.context = DenoisingContext(
            encoded_control=torch.randn(2, 4, 2, 2),
            prompt_embeds=torch.randn(2, 3, 4),
            neg_prompt_embeds=torch.randn(2, 3, 4),
            event_feats={"2": torch.randn(2, 4, 2, 2)},
        )
        self.timesteps = torch.tensor([500])

    def test_baseline_calls_one_unet_and_original_scheduler_sample(self):
        generator = _mock_generator()
        output, stats = generator.predict_denoised_latent_from_context(
            self.context, self.timesteps, measure_time=True,
        )
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(generator.noise_scheduler.calls), 1)
        prediction, timestep, scheduler_sample = generator.noise_scheduler.calls[0]
        self.assertTrue(torch.equal(scheduler_sample, self.context.encoded_control))
        self.assertTrue(torch.equal(timestep, self.timesteps))
        self.assertTrue(torch.equal(output, self.context.encoded_control - prediction))
        self.assertGreaterEqual(stats["unet_ms"], 0.0)
        self.assertTrue(torch.isfinite(output).all())

    def test_context_reuse_does_not_mutate_inputs(self):
        generator = _mock_generator()
        original = {
            "latent": self.context.encoded_control.clone(),
            "prompt": self.context.prompt_embeds.clone(),
            "negative": self.context.neg_prompt_embeds.clone(),
            "events": self.context.event_feats["2"].clone(),
        }
        outputs = [generator.predict_denoised_latent_from_context(
            self.context, self.timesteps,
        )[0] for _ in range(2)]
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertEqual(len(generator.calls), 2)
        for call in generator.calls:
            self.assertTrue(torch.equal(call["latent"], original["latent"]))
            self.assertTrue(torch.equal(call["prompt"], original["prompt"]))
            self.assertTrue(torch.equal(call["events"]["2"], original["events"]))
        self.assertTrue(torch.equal(self.context.encoded_control, original["latent"]))
        self.assertTrue(torch.equal(self.context.prompt_embeds, original["prompt"]))
        self.assertTrue(torch.equal(self.context.neg_prompt_embeds, original["negative"]))
        self.assertTrue(torch.equal(self.context.event_feats["2"], original["events"]))

    def test_generic_prediction_retains_training_gradients(self):
        # 通用预测入口仍被训练调用，不能在迁移时添加全局 no_grad 或 detach。
        generator = _mock_generator()
        self.context.event_feats["2"].requires_grad_(True)
        output, _ = generator.predict_denoised_latent_from_context(self.context, self.timesteps)
        gradient = torch.autograd.grad(output.sum(), self.context.event_feats["2"])[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


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

    def forward(self, sample, timesteps, encoder_hidden_states, feature_guidance=None):
        if self.fail:
            raise RuntimeError("intentional failure")
        hidden = sample.flatten(2).transpose(1, 2)
        output = self.block.attn1.to_v(hidden)
        return types.SimpleNamespace(sample=output.transpose(1, 2).reshape_as(sample))


class EventLoRACacheTest(unittest.TestCase):
    def _build_generator(self, variant, fail=False):
        generator = object.__new__(OSEDiffEventFiLMGen)
        nn.Module.__init__(generator)
        generator.unet = _TinyEventUNet(fail=fail)
        config = EventLoRAConfig(variant=variant, rank=4, alpha=4, mod_channels=8, mod_num_heads=2)
        event_channels = {"8": 2, "4": 2, "2": 2, "1": 2}
        replace_self_attn_to_v_with_event_lora(generator.unet, config, event_channels)
        nn.init.normal_(generator.unet.block.attn1.to_v.lora_up.weight, std=0.1)
        generator.noise_scheduler = _Scheduler()
        return generator

    @staticmethod
    def _events(batch_size):
        return {scale: torch.randn(batch_size, 2, int(scale), int(scale))
                for scale in ("8", "4", "2", "1")}

    def test_all_variants_smoke_and_clear_cache(self):
        latent = torch.randn(2, 4, 2, 2)
        prompts = torch.randn(2, 3, 4)
        events = self._events(2)
        for variant in EVENT_LORA_VARIANTS:
            with self.subTest(variant=variant):
                generator = self._build_generator(variant)
                context = DenoisingContext(
                    encoded_control=latent, prompt_embeds=prompts,
                    neg_prompt_embeds=torch.zeros_like(prompts), event_feats=events,
                )
                output, _ = generator.predict_denoised_latent_from_context(context, torch.tensor([500]))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(len(generator.noise_scheduler.calls), 1)
                self.assertIsNone(generator.unet.block.attn1.to_v.event_feats)

    def test_cache_is_cleared_when_unet_raises(self):
        generator = self._build_generator("film_affine", fail=True)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            generator._run_unet_with_event_features(
                torch.randn(1, 4, 2, 2), torch.tensor([500]),
                torch.randn(1, 3, 4), self._events(1),
            )
        self.assertIsNone(generator.unet.block.attn1.to_v.event_feats)


if __name__ == "__main__":
    unittest.main()
