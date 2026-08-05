import unittest

import torch

from models.autoencoder_kl import AutoencoderKL


def build_small_vae():
    return AutoencoderKL(
        block_out_channels=(32, 64, 64, 64),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        layers_per_block=1,
        sample_size=32,
        latent_channels=4,
        norm_num_groups=32,
    )


def build_features(batch=1):
    return [
        torch.randn(batch, 64, 32, 32),
        torch.randn(batch, 128, 16, 16),
        torch.randn(batch, 256, 8, 8),
        torch.randn(batch, 512, 4, 4),
    ]


class DecoderRefinementTest(unittest.TestCase):
    def test_disabled_parity_near_zero_and_four_level_consumption(self):
        torch.manual_seed(3)
        vae = build_small_vae().eval()
        latent = torch.randn(1, 4, 4, 4)
        with torch.no_grad():
            reference = vae.decode(latent).sample
        vae.install_feature_aware_decoder()
        with torch.no_grad():
            disabled = vae.decode(latent, fusion_enabled=False).sample
            enabled = vae.decode(
                latent,
                fused_features=build_features(),
                fusion_enabled=True,
            ).sample
        torch.testing.assert_close(reference, disabled, rtol=0, atol=0)
        self.assertLess(float((enabled - disabled).abs().max()), 1e-3)
        self.assertEqual(
            [block.forward_count for block in vae.decoder.feature_fusion_blocks],
            [1, 1, 1, 1],
        )

    def test_only_fusion_parameters_receive_gradients(self):
        torch.manual_seed(5)
        vae = build_small_vae()
        vae.install_feature_aware_decoder()
        vae.requires_grad_(False)
        vae.decoder.feature_fusion_blocks.requires_grad_(True)
        latent = torch.randn(1, 4, 4, 4)
        output = vae.decode(
            latent,
            fused_features=build_features(),
            fusion_enabled=True,
        ).sample
        output.square().mean().backward()
        names_with_grad = [
            name for name, parameter in vae.named_parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(names_with_grad)
        self.assertTrue(
            all(
                name.startswith("decoder.feature_fusion_blocks.")
                for name in names_with_grad
            )
        )
        gradient_norm = sum(
            parameter.grad.float().norm().item()
            for parameter in vae.decoder.feature_fusion_blocks.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_norm, 0.0)

    def test_missing_level_and_channel_mismatch_fail_fast(self):
        vae = build_small_vae()
        vae.install_feature_aware_decoder()
        latent = torch.randn(1, 4, 4, 4)
        with self.assertRaisesRegex(ValueError, "fused feature levels"):
            vae.decode(
                latent,
                fused_features=build_features()[:-1],
                fusion_enabled=True,
            )
        bad_features = build_features()
        bad_features[-1] = torch.randn(1, 511, 4, 4)
        with self.assertRaisesRegex(ValueError, "condition channel mismatch"):
            vae.decode(
                latent,
                fused_features=bad_features,
                fusion_enabled=True,
            )

    def test_gradient_checkpointing_keeps_same_fusion_order(self):
        vae = build_small_vae()
        vae.install_feature_aware_decoder()
        vae.requires_grad_(False)
        vae.decoder.feature_fusion_blocks.requires_grad_(True)
        vae.decoder.gradient_checkpointing = True
        vae.train()
        output = vae.decode(
            torch.randn(1, 4, 4, 4),
            fused_features=build_features(),
            fusion_enabled=True,
        ).sample
        output.mean().backward()
        self.assertEqual(
            [block.forward_count for block in vae.decoder.feature_fusion_blocks],
            [1, 1, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
