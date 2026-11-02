"""验证最终融合层的前向旁路、梯度白名单和旧权重布局。"""
import itertools
import unittest
import torch
from models.fusion_selection import parse_fusion_layers, select_fusion_layers
from models.event_film_lora import EventLoRAConfig
from test_decoder_refinement import build_small_vae, build_features


class FinalConfigurationTest(unittest.TestCase):
    def test_selection_validation(self):
        self.assertEqual(parse_fusion_layers('d3,d0'), (0, 3))
        self.assertEqual(parse_fusion_layers('none'), ())
        self.assertEqual(parse_fusion_layers('all'), (0, 1, 2, 3))
        for invalid in ('d0,d0', 'd4', '', 'd1,', 'd-1'):
            with self.assertRaises(ValueError):
                parse_fusion_layers(invalid)
        self.assertEqual(EventLoRAConfig().mod_channels, 64)
        for variant in ('film_affine', 'cbam_lowrank_residual', 'cbam_output_gate', 'residual_cross_attn_bax'):
            with self.assertRaises(ValueError):
                EventLoRAConfig(variant=variant)

    def test_every_subset_bypasses_and_freezes_unselected_layers(self):
        torch.manual_seed(9)
        vae = build_small_vae().eval()
        vae.install_feature_aware_decoder()
        checkpoint = {k:v.clone() for k,v in vae.state_dict().items()}
        features = build_features()
        latent = torch.randn(1, 4, 4, 4)
        with torch.no_grad():
            baseline = vae.decode(latent, fusion_enabled=False).sample
        for bits in itertools.product((False, True), repeat=4):
            levels = tuple(i for i, active in enumerate(bits) if active)
            vae.requires_grad_(False)
            vae.zero_grad(set_to_none=True)
            select_fusion_layers(vae.decoder, levels, trainable=True)
            before = [block.forward_count for block in vae.decoder.feature_fusion_blocks]
            output = vae.decode(latent, fused_features=features, fusion_enabled=True).sample
            if levels:
                output.square().mean().backward()
            else:
                torch.testing.assert_close(output, baseline, rtol=0, atol=0)
            for i, block in enumerate(vae.decoder.feature_fusion_blocks):
                self.assertEqual(block.forward_count - before[i], int(i in levels))
                self.assertEqual(any(p.grad is not None for p in block.parameters()), i in levels)
            self.assertEqual(set(vae.state_dict()), set(checkpoint))
        vae.load_state_dict(checkpoint, strict=True)

    def test_joint_guidance_gradient_and_parity(self):
        # 无真实数据和预训练模型：验证高阶图、冻结教师和原一步引导数值一致。
        from test_joint_h1_refinement import (
            test_joint_gradient_and_inference_parity,
            test_joint_rejection_falls_back,
            test_fixed_teacher_masks_despite_trainable_shared_features,
        )
        test_joint_gradient_and_inference_parity()
        test_joint_rejection_falls_back()
        test_fixed_teacher_masks_despite_trainable_shared_features()
