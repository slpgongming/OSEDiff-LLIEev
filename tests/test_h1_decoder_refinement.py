"""H1→refinement 的接受状态、批处理、GT隔离和外层梯度检查。"""
from types import SimpleNamespace
import torch
from torch import nn
from models.event_structure_guidance import GuidanceConfig, StructureObservation, optimize_features
from models.image_structure import image_structure
from models.h1_decoder_refinement import guided_latents, refinement_guided_sample, H1DecoderRefinementGen


def test_latent_callback_only_records_baseline_and_accepted_candidates():
    torch.manual_seed(8)
    image = torch.rand(1, 3, 16, 16) - .5
    observation = StructureObservation(image_structure((image.roll(1,-1)+1)/2), torch.ones(1,2,16,16))
    current, accepted = {}, []
    def render(correction):
        value = image if correction is None else correction.apply(image)
        current['value'] = value.detach().clone()
        return value
    callback = lambda: accepted.append(current['value'].clone())
    output, stats = optimize_features(render, observation, GuidanceConfig(steps=1,
        max_low_change=1e-12, max_outside_change=1e-12), on_accept=callback)
    assert stats['accepted_steps'] == 0 and len(accepted) == 1
    torch.testing.assert_close(accepted[-1], image, rtol=0, atol=0)
    assert not torch.equal(current['value'], accepted[-1])
    accepted.clear()
    output, stats = optimize_features(render, observation, GuidanceConfig(steps=1), on_accept=callback)
    assert len(accepted) == 1 + stats['accepted_steps']
    torch.testing.assert_close(output, accepted[-1].clamp(-1,1), rtol=0, atol=0)


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(scaling_factor=1.)
        self.decoder = nn.Module(); self.decoder.feature_fusion_blocks = nn.Conv2d(3,3,1)
    def decode(self, latent, fusion_enabled=False):
        assert not fusion_enabled
        return SimpleNamespace(sample=latent)


class FakeGen(nn.Module):
    def __init__(self):
        super().__init__();self.vae=FakeVAE();self.unet=nn.Module();self.args=SimpleNamespace(generator_timestep=500)
        self.seen=[]
    def prepare_denoising_context(self, rgb, events, batch):
        assert set(batch)=={'prompt','neg_prompt'}
        self.seen.append((rgb.clone(),events.clone()))
        return rgb
    def predict_denoised_latent_from_context(self, context, timesteps, feature_guidance=None):
        return (context if feature_guidance is None else feature_guidance.apply(context)), {}


class FakeHead(nn.Module):
    def __init__(self, confidence):super().__init__();self.confidence=confidence
    def forward(self, events):
        edges=events.new_full((1,2,*events.shape[-2:]),.2)
        return edges, torch.full_like(edges,self.confidence)


def test_batch_inner_loop_detaches_and_only_refinement_receives_outer_gradient():
    torch.manual_seed(2);gen=FakeGen()
    rgb=torch.rand(2,3,256,256)*.5;events=torch.rand(2,5,256,256);structure=torch.rand(2,10,256,256)
    before={k:v.clone() for k,v in gen.state_dict().items()}
    latent,records=guided_latents(gen,rgb,events,structure,
        {'prompt':['a','b'],'neg_prompt':['','']},FakeHead(1.),None,GuidanceConfig(steps=1))
    assert not latent.requires_grad and len(records)==2 and len(gen.seen)==2
    assert torch.equal(gen.seen[1][1],events[1:2])
    optimizer=torch.optim.AdamW(gen.vae.decoder.feature_fusion_blocks.parameters(),lr=1e-3)
    gen.vae.decoder.feature_fusion_blocks(latent).square().mean().backward();optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gen.parameters())
    assert any(not torch.equal(v,gen.state_dict()[k]) for k,v in before.items())


def test_no_supported_events_captures_unmodified_baseline_latent():
    gen=FakeGen();rgb=torch.rand(1,3,256,256)
    latent,rows=guided_latents(gen,rgb,torch.zeros(1,5,256,256),torch.zeros(1,10,256,256),
        {'prompt':['a'],'neg_prompt':['']},FakeHead(0.),None,GuidanceConfig(steps=1))
    torch.testing.assert_close(latent,rgb,rtol=0,atol=0)
    assert rows[0]['reason']=='no_event_support' and not latent.requires_grad


def test_h1_collation_drops_nullable_diagnostic_metadata():
    from train_idea3_decoder_refinement import collate_h1_refinement
    sample = dict(conditioning_pixel_values=torch.zeros(3,4,4), output_pixel_values=torch.ones(3,4,4),
                  events=torch.zeros(5,4,4), events_refinement=torch.zeros(5,4,4),
                  events_structure=torch.zeros(10,4,4), event_voxel_raw=torch.zeros(5,4,4),
                  prompt="a", neg_prompt="", structure_meta={"t_ref":None,"window_before":None})
    batch=collate_h1_refinement([sample,sample])
    assert batch['events_structure'].shape==(2,10,4,4) and batch['prompt']==['a','a']
    assert 'structure_meta' not in batch


def test_full_refinement_guidance_uses_frozen_decoder_and_returns_accepted_image():
    """真实可导条件残差影响目标、拒绝回滚；输入/权重及既有状态不受内层优化修改。"""
    class Gen(FakeGen):
        def __init__(self):
            super().__init__()
            self.decoder_refinement_enabled=True; self.decoder_fusion_weight=1.0
            self.vae.decoder.inference_fusion_levels=(1,2,3)
            self.calls=[]; self.condition_calls=0
        def prepare_decoder_refinement_condition(self, rgb, events, mode):
            self.condition_calls+=1
            return events[:, :1].detach()
        def decode_denoised_latent(self, latent, rgb, *, decoder_fusion_enabled,
                                   refinement_fusion_mode, clamp_output, refinement_condition):
            assert decoder_fusion_enabled and not clamp_output
            assert not any(p.requires_grad for p in self.parameters())
            output=latent * 0.7 + self.decoder_fusion_weight * self.vae.decoder.feature_fusion_blocks(latent) * 0.1 + refinement_condition * .01
            self.calls.append(output.detach().clone())
            return output
    torch.manual_seed(44)
    gen=Gen();head=FakeHead(1.);gen.train();head.train()
    rgb=(torch.rand(1,3,256,256)*.4).requires_grad_()
    events=torch.ones(1,5,256,256,requires_grad=True)
    structure=torch.ones(1,10,256,256,requires_grad=True)
    evref=torch.ones(1,5,256,256,requires_grad=True)
    before={k:v.clone() for k,v in gen.state_dict().items()}
    flags=[p.requires_grad for p in gen.parameters()]
    state=torch.get_rng_state().clone()
    image,z,diag=refinement_guided_sample(gen,rgb,events,structure,
        {'prompt':['x'],'neg_prompt':['']},head,None,
        GuidanceConfig(steps=1,max_low_change=1e-12,max_outside_change=1e-12),events_refinement=evref)
    assert len(gen.calls)>1 and gen.condition_calls==1 and diag['accepted_steps']==0
    torch.testing.assert_close(image,gen.calls[0].clamp(-1,1),rtol=0,atol=0)
    torch.testing.assert_close(z,rgb.detach(),rtol=0,atol=0)
    assert not image.requires_grad and not z.requires_grad and rgb.grad is None and events.grad is None
    assert all(p.grad is None for p in gen.parameters())
    assert all(torch.equal(v,gen.state_dict()[k]) for k,v in before.items())
    assert flags==[p.requires_grad for p in gen.parameters()] and gen.training and head.training
    assert torch.equal(state,torch.get_rng_state())
    assert diag['decoder_path']=='full_refinement' and diag['uses_gt_for_guidance'] is False


def test_full_refinement_rejects_gt_in_prompt_batch():
    import pytest
    gen=FakeGen(); gen.decoder_refinement_enabled=True
    rgb=torch.zeros(1,3,256,256)
    with pytest.raises(ValueError,match='GT'):
        refinement_guided_sample(gen,rgb,torch.zeros(1,5,256,256),torch.zeros(1,10,256,256),
            {'prompt':['x'],'neg_prompt':[''],'output_pixel_values':rgb},FakeHead(1.),None,
            GuidanceConfig(steps=1),events_refinement=torch.zeros(1,5,256,256))


def test_full_refinement_training_restores_outer_fusion_gradients_and_context_rng():
    """内层仅 delta 求导；恢复参数开关后外层 fusion 可更新，encode 随机状态按样本推进。"""
    class TrainingGen(FakeGen):
        forward=H1DecoderRefinementGen.forward
        def __init__(self):
            super().__init__()
            self.args.mixed_precision='no';self.h1_decoder_path='full_refinement'
            self.decoder_refinement_enabled=True;self.decoder_fusion_weight=1.;self.refinement_fusion_mode='both'
            self.h1_predictor=FakeHead(1.);self.h1_operator=None;self.h1_config=GuidanceConfig(steps=1)
            self.inner_flags=[]
        def prepare_denoising_context(self,rgb,events,prompts):
            # 只把一次 VAE encode 的随机消费推进到调用方，不包含候选 replay。
            return rgb+torch.rand_like(rgb)*.01
        def prepare_decoder_refinement_condition(self,rgb,events,mode):
            return events[:, :1].detach()
        def decode_denoised_latent(self,z,rgb,events_refinement=None,decoder_fusion_enabled=None,
                refinement_fusion_mode=None,clamp_output=True,refinement_condition=None):
            self.inner_flags.append(any(p.requires_grad for p in self.parameters()))
            output=z+self.vae.decoder.feature_fusion_blocks(z)*.05
            return output.clamp(-1,1) if clamp_output else output
    gen=TrainingGen().train();rgb=torch.rand(2,3,256,256)*.4
    batch=dict(prompt=['a','b'],neg_prompt=['',''],events_structure=torch.ones(2,10,256,256))
    before={k:v.clone() for k,v in gen.state_dict().items()}
    torch.manual_seed(99);torch.rand_like(rgb[:1]);torch.rand_like(rgb[:1]);expected_rng=torch.get_rng_state()
    torch.manual_seed(99)
    prediction,z,_,_=gen(rgb,torch.ones(2,5,256,256),torch.tensor([500]),batch=batch,
        events_refinement=torch.ones(2,5,256,256),decoder_fusion_enabled=True)
    assert torch.equal(expected_rng,torch.get_rng_state())
    assert not z.requires_grad and prediction.requires_grad
    assert not any(gen.inner_flags[:-1]) and gen.inner_flags[-1]
    optimizer=torch.optim.AdamW(gen.vae.decoder.feature_fusion_blocks.parameters(),lr=1e-3)
    prediction.square().mean().backward();optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gen.vae.decoder.feature_fusion_blocks.parameters())
    assert any(not torch.equal(v,gen.state_dict()[k]) for k,v in before.items())
