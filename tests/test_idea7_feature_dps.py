"""全通道特征反馈的数值、生命周期和真实小模型梯度检查；不下载 checkpoint。"""

import unittest

import torch
from torch import nn

from models.feature_dps_guidance import (
    FeatureCorrectionState,
    FeatureDPSConfig,
    FeatureDPSRunner,
    SkipFeatureDPS,
    freeze_for_feature_dps,
)


class ToyGenerator(nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.output = nn.Conv2d(6, 3, 1, bias=False)
        nn.init.constant_(self.output.weight, 1 / 6)
        self.stochastic = stochastic
        self.calls = 0
        self.states = []
        self.random_values = []
        self.gradient_checkpointing = True

    def forward(self, state):
        self.calls += 1
        self.states.append(state)
        feature = torch.full((1, 6, 4, 4), 0.4)
        if state is not None:
            state.bind_num_up_blocks(4)
            for index in range(4):
                feature = state.apply(index, feature)
        image = self.output(feature)
        if self.stochastic:
            noise = torch.rand(()) * 0.001
            self.random_values.append(noise.item())
            image = image + noise
        return image.clamp(0, 1)


class FeatureStateTest(unittest.TestCase):
    def test_full_channel_identity_shape_and_unique_injection(self):
        state = FeatureCorrectionState(FeatureDPSConfig())
        state.bind_num_up_blocks(4)
        state.begin_forward()
        feature = torch.randn(1, 7, 4, 4)
        self.assertIs(state.apply(0, feature), feature)
        result = state.apply(2, feature)
        torch.testing.assert_close(result, feature, rtol=0, atol=0)
        self.assertEqual(state.u.shape, feature.shape)
        self.assertTrue(state.u.is_leaf)
        self.assertEqual(state.u.dtype, torch.float32)
        state.finish_forward()
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            state.apply(2, feature)
        state.clear()
        self.assertIsNone(state.u)
        self.assertIsNone(state.scale)

    def test_invalid_target_and_missing_injection_are_rejected(self):
        with self.assertRaises(ValueError):
            FeatureCorrectionState(FeatureDPSConfig(up_block_index=3)).bind_num_up_blocks(4)
        state = FeatureCorrectionState(FeatureDPSConfig())
        state.bind_num_up_blocks(4)
        with self.assertRaises(RuntimeError):
            state.finish_forward()

    def test_shape_dtype_and_batch_changes_are_rejected(self):
        state = FeatureCorrectionState(FeatureDPSConfig())
        state.bind_num_up_blocks(4)
        state.apply(2, torch.ones(1, 6, 4, 4))
        for feature in (torch.ones(1, 7, 4, 4), torch.ones(1, 6, 4, 4).double(), torch.ones(2, 6, 4, 4)):
            state.begin_forward()
            with self.assertRaises(ValueError):
                state.apply(2, feature)

    def test_feature_gradient_matches_finite_difference(self):
        state = FeatureCorrectionState(FeatureDPSConfig())
        state.bind_num_up_blocks(4)
        h = torch.full((1, 3, 2, 2), 0.4)
        result = state.apply(2, h)
        gradient = torch.autograd.grad(result.square().mean(), state.u)[0]
        direction = torch.randn_like(state.u)
        expected = (gradient * direction).sum()
        epsilon = 1e-3
        losses = []
        for sign in (-1, 1):
            state.u = (sign * epsilon * direction).detach().requires_grad_(True)
            state.begin_forward()
            losses.append(state.apply(2, h).square().mean())
        observed = (losses[1] - losses[0]) / (2 * epsilon)
        torch.testing.assert_close(expected, observed, rtol=2e-3, atol=1e-5)


class FeatureRunnerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = freeze_for_feature_dps(ToyGenerator())

    def test_zero_steps_and_radius_need_no_observation(self):
        for config in (FeatureDPSConfig(steps=0), FeatureDPSConfig(radius_rel=0)):
            result = FeatureDPSRunner(config).run(self.model)
            self.assertEqual(result.stats["forward_calls"], 1)
            self.assertEqual(result.stats["backward_calls"], 0)
            torch.testing.assert_close(result.image, result.baseline, rtol=0, atol=0)
            self.assertIsNone(self.model.states[-1])

    def test_skip_reason_needs_no_loss(self):
        result = FeatureDPSRunner().run(self.model, skip_reason="scene_start")
        self.assertEqual(result.stats["skip_reason"], "scene_start")
        self.assertEqual(self.model.calls, 1)

    def test_one_and_two_updates_count_and_freeze(self):
        for steps in (1, 2):
            model = freeze_for_feature_dps(ToyGenerator())
            before = {key: value.clone() for key, value in model.state_dict().items()}
            factories = []

            def factory(baseline):
                factories.append(baseline.clone())
                self.assertFalse(baseline.requires_grad)
                return lambda image: (image - 0.8).square().mean()

            result = FeatureDPSRunner(FeatureDPSConfig(steps=steps)).run(model, factory)
            self.assertEqual(model.calls, steps + 1)
            self.assertEqual(result.stats["forward_calls"], steps + 1)
            self.assertEqual(result.stats["backward_calls"], steps)
            self.assertEqual(len(factories), 1)
            self.assertLess(result.stats["energy_after"], result.stats["energy_before"])
            self.assertEqual(result.stats["feature_shape"], [1, 6, 4, 4])
            self.assertFalse(result.image.requires_grad)
            for parameter in model.parameters():
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertFalse(model.gradient_checkpointing)
            self.assertTrue(all(state.u is None for state in model.states))

    def test_projection_limits_total_relative_rms(self):
        result = FeatureDPSRunner(FeatureDPSConfig(steps=2, step_rel=0.5, radius_rel=0.01)).run(
            self.model, lambda baseline: lambda image: (image - 0.9).square().mean()
        )
        for candidate in result.history:
            self.assertLessEqual(candidate["relative_rms"], 0.010001)

    def test_best_candidate_can_be_middle_iteration(self):
        # 第一步到达0.6，第二步回到0.4；目标0.59应保留中间候选而非最后一轮。
        result = FeatureDPSRunner(FeatureDPSConfig(steps=2, step_rel=0.5, radius_rel=1)).run(
            self.model, lambda baseline: lambda image: (image - 0.59).square().mean()
        )
        self.assertEqual(result.stats["selected_iteration"], 1)
        self.assertEqual(result.stats["forward_calls"], 3)
        self.assertEqual(result.stats["backward_calls"], 2)
        self.assertAlmostEqual(result.image.mean().item(), 0.6, places=5)

    def test_no_improvement_keeps_baseline(self):
        result = FeatureDPSRunner(FeatureDPSConfig(step_rel=0.5, radius_rel=1)).run(
            self.model, lambda baseline: lambda image: (image - 0.41).square().mean()
        )
        self.assertEqual(result.stats["selected_iteration"], 0)
        self.assertEqual(result.stats["skip_reason"], "no_energy_improvement")
        torch.testing.assert_close(result.image, result.baseline, rtol=0, atol=0)

    def test_zero_and_disconnected_gradients(self):
        for energy, reason, backward_calls in (
            (lambda image: (image * 0).sum(), "zero_gradient", 1),
            (lambda image: image.detach().sum(), "disconnected_energy", 0),
        ):
            result = FeatureDPSRunner().run(self.model, lambda baseline: energy)
            self.assertEqual(result.stats["skip_reason"], reason)
            self.assertEqual(result.stats["forward_calls"], 1)
            self.assertEqual(result.stats["backward_calls"], backward_calls)

    def test_nonfinite_candidate_falls_back_and_cleans_state(self):
        def generate(state):
            image = self.model(state)
            return image if self.model.calls == 1 else image * float("nan")

        result = FeatureDPSRunner().run(generate, lambda baseline: lambda image: image.square().mean())
        self.assertEqual(result.stats["skip_reason"], "non_finite_candidate")
        self.assertEqual(result.stats["forward_calls"], 2)
        torch.testing.assert_close(result.image, result.baseline, rtol=0, atol=0)
        self.assertIsNone(self.model.states[-1].u)

    def test_factory_skip_and_exception_cleanup(self):
        cleaned = []

        def unavailable(baseline):
            raise SkipFeatureDPS("insufficient_valid_mask")

        result = FeatureDPSRunner().run(self.model, unavailable, cleanup=lambda: cleaned.append(True))
        self.assertEqual(result.stats["skip_reason"], "insufficient_valid_mask")
        self.assertEqual(result.stats["forward_calls"], 1)
        self.assertEqual(result.stats["backward_calls"], 0)
        self.assertTrue(cleaned)
        cleaned.clear()

        def broken(baseline):
            raise ValueError("test failure")

        with self.assertRaisesRegex(ValueError, "test failure"):
            FeatureDPSRunner().run(self.model, broken, cleanup=lambda: cleaned.append(True))
        self.assertTrue(cleaned)
        self.assertTrue(all(state.u is None for state in self.model.states))

    def test_rng_shared_between_candidates_and_restored_after_run(self):
        model = freeze_for_feature_dps(ToyGenerator(stochastic=True))
        before = torch.get_rng_state().clone()
        result = FeatureDPSRunner(FeatureDPSConfig(steps=2)).run(
            model, lambda baseline: lambda image: (image - 0.8).square().mean()
        )
        self.assertEqual(result.stats["forward_calls"], 3)
        self.assertEqual(len(set(model.random_values)), 1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_inference_mode_is_rejected(self):
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, "inference_mode"):
            FeatureDPSRunner().run(self.model, lambda baseline: lambda image: image.sum())


class ActualSmallModelGradientTest(unittest.TestCase):
    def test_real_unet_event_cross_attention_scheduler_and_vae(self):
        # 使用真实仓库模块的小尺寸随机权重，验证代码链而非声称真实 checkpoint 效果。
        from diffusers import DDPMScheduler
        from models.autoencoder_kl import AutoencoderKL
        from models.event_film_lora import EventLoRAConfig
        from models.event_lora_utils import replace_self_attn_to_v_with_event_lora, iter_event_lora_modules
        from models.unet_2d_condition import UNet2DConditionModel
        from osediff_event_film import OSEDiffEventFiLMGen, DenoisingContext

        torch.manual_seed(8)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            gen = object.__new__(OSEDiffEventFiLMGen)
            nn.Module.__init__(gen)
            gen.unet = UNet2DConditionModel(
                sample_size=8, in_channels=4, out_channels=4,
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                block_out_channels=(32, 32), layers_per_block=1,
                cross_attention_dim=8, attention_head_dim=4, norm_num_groups=8,
            )
            event_config = EventLoRAConfig(
                variant="residual_cross_attn", rank=2, mod_channels=8,
                mod_num_heads=2, mod_window_size=4, zero_init_last=False,
            )
            replaced, _ = replace_self_attn_to_v_with_event_lora(gen.unet, event_config, {"4": 4, "8": 4})
            self.assertGreater(replaced, 0)
            with torch.no_grad():
                for _, module in iter_event_lora_modules(gen.unet):
                    module.lora_up.weight.fill_(0.01)
            gen.vae = AutoencoderKL(block_out_channels=(32,), norm_num_groups=8, sample_size=8)
            gen.noise_scheduler = DDPMScheduler(num_train_timesteps=20, clip_sample=False)
            gen.noise_scheduler.set_timesteps(1)
            gen.decoder_fusion_enabled = False
            gen.decoder_refinement_enabled = False
            gen.refinement_fusion_mode = "both"
            gen.decoder_fusion_weight = 0.1
            gen.last_refinement_stats = {}
            freeze_for_feature_dps(gen)
            context = DenoisingContext(
                encoded_control=torch.randn(1, 4, 8, 8) * 0.05,
                prompt_embeds=torch.randn(1, 3, 8),
                neg_prompt_embeds=torch.randn(1, 3, 8),
                event_feats={"4": torch.randn(1, 4, 4, 4), "8": torch.randn(1, 4, 8, 8)},
            )
            timesteps = torch.tensor([19])
            original_weights = {key: value.clone() for key, value in gen.state_dict().items()}

            def generate(state):
                latent, _ = gen.predict_denoised_latent_from_context(context, timesteps, feature_guidance=state)
                return gen.decode_denoised_latent(latent, torch.zeros(1, 3, 8, 8)).add(1).div(2)

            plain = FeatureDPSRunner(FeatureDPSConfig(steps=0)).run(
                generate, cleanup=gen.clear_feature_dps_caches,
            )
            result = FeatureDPSRunner().run(
                generate,
                lambda baseline: lambda image: (image - (baseline + 0.03).clamp(0, 1)).square().mean(),
                cleanup=gen.clear_feature_dps_caches,
            )
            self.assertEqual(result.stats["forward_calls"], 2)
            self.assertEqual(result.stats["backward_calls"], 1)
            torch.testing.assert_close(result.baseline, plain.image, rtol=0, atol=0)
            self.assertGreater(result.history[0]["gradient_rms"], 0)
            self.assertEqual(result.stats["up_block_index"], 0)
            self.assertEqual(result.stats["feature_shape"][1], 32)
            self.assertTrue(torch.isfinite(result.image).all())
            for key, value in gen.state_dict().items():
                torch.testing.assert_close(value, original_weights[key], rtol=0, atol=0)
            for _, module in iter_event_lora_modules(gen.unet):
                self.assertIsNone(module.event_feats)
                self.assertFalse(module.last_aux)
            self.assertTrue(all(parameter.grad is None for parameter in gen.parameters()))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
