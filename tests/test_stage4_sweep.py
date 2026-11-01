"""测试 decoder 执行序号开关，避免把 condition 层号误当成 decoder 层号。"""
import pytest
import torch
from test_decoder_refinement import build_small_vae, build_features


def test_decoder_sweep_bypass_parity_and_execution_order():
    torch.manual_seed(17)
    vae = build_small_vae().eval()
    vae.install_feature_aware_decoder()
    latent = torch.randn(1,4,4,4)
    features = build_features()
    def run(enabled=True):
        with torch.no_grad():
            return vae.decode(latent, fused_features=features, fusion_enabled=enabled).sample
    reference = run()
    vae.decoder.set_inference_fusion_levels((0,1,2,3))
    torch.testing.assert_close(run(), reference, rtol=0, atol=0)
    vae.decoder.set_inference_fusion_levels(())
    torch.testing.assert_close(run(), run(False), rtol=0, atol=0)
    for levels in ((2,3),(1,2,3)):
        counts=[block.forward_count for block in vae.decoder.feature_fusion_blocks]
        vae.decoder.set_inference_fusion_levels(levels)
        run()
        assert [block.forward_count-counts[i] for i,block in enumerate(vae.decoder.feature_fusion_blocks)] == [int(i in levels) for i in range(4)]
    with pytest.raises(ValueError):vae.decoder.set_inference_fusion_levels((4,))


def test_global_weight_scales_whole_legacy_residual():
    from models.decoder_refinement import DecoderFeatureFusionBlock
    torch.manual_seed(9)
    block=DecoderFeatureFusionBlock(64,32).eval()
    for module in block.modules():
        if isinstance(module,torch.nn.Conv2d):torch.nn.init.normal_(module.weight,std=.02)
    condition=torch.randn(1,64,8,8);feature=torch.randn(1,32,8,8)
    with torch.no_grad():
        full=block(condition,feature,fusion_weight=1.)
        assert (full-feature).abs().max() > 1e-4
        for weight in (.25,.5,.75,1.):
            actual=block(condition,feature,fusion_weight=weight)
            torch.testing.assert_close(actual-feature,(full-feature)*weight,rtol=1e-4,atol=3e-7)
