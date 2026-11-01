"""无H1对照的纯latent路径与共同初始化校验。"""
from types import SimpleNamespace
import random
import numpy as np
import pytest
import torch
from torch import nn
from models.decoder_refinement_control import frozen_stage3_latents
from models.decoder_refinement_initialization import load_refinement_initialization
from models.rgb_event_encoder import sha256_file


def test_no_h1_latents_are_fp32_detached_and_scheduler_rng_is_restored():
    class Fake(nn.Module):
        def __init__(self):super().__init__();self.unet=nn.Module();self.calls=[]
        def prepare_denoising_context(self,rgb,events,batch):
            assert set(batch)=={'prompt','neg_prompt'} and rgb.dtype==events.dtype==torch.float32
            assert not torch.is_grad_enabled();self.calls.append(batch);return rgb
        def predict_denoised_latent_from_context(self,context,timesteps):
            return context+torch.rand_like(context),{}
    gen=Fake();torch.manual_seed(42);state=torch.get_rng_state()
    rgb=torch.zeros(2,3,8,8,requires_grad=True)
    latent=frozen_stage3_latents(gen,rgb,torch.zeros(2,5,8,8),torch.tensor([500]),dict(prompt=['a','b'],neg_prompt=['','']))
    assert not latent.requires_grad and latent.shape==rgb.shape and len(gen.calls)==2
    assert torch.equal(state,torch.get_rng_state())
    torch.testing.assert_close(latent[0],latent[1],rtol=0,atol=0)


def test_shared_fusion_initialization_validates_weights_and_restores_rng(tmp_path):
    class Fake(nn.Module):
        def __init__(self):
            super().__init__();self.vae=nn.Module();self.vae.decoder=nn.Module()
            self.vae.decoder.feature_fusion_blocks=nn.Linear(3,3)
            self.vae.decoder.refinement_config=lambda:dict(variant='legacy_fused')
            self.rgb_event_encoder=SimpleNamespace(pretrained_manifest=dict(sha256={'a':'b'}))
    model=Fake();source=tmp_path/'stage3.pt';source.write_bytes(b'stage3')
    expected={k:v.clone()+1 for k,v in model.vae.decoder.feature_fusion_blocks.state_dict().items()}
    torch.manual_seed(42);state=torch.get_rng_state()
    saved=dict(format='paired-stage4-refinement-init-v1',seed=42,stage3_sha256=sha256_file(source),
        decoder_refinement_config=model.vae.decoder.refinement_config(),rgb_event_manifest=model.rgb_event_encoder.pretrained_manifest,
        state_dict=expected,state_sha256='fixture',torch_rng_cpu=state,torch_rng_cuda=None,python_rng=random.getstate(),numpy_rng=np.random.get_state())
    path=tmp_path/'init.pt';torch.save(saved,path);torch.rand(50)
    report=load_refinement_initialization(model,str(path),SimpleNamespace(seed=42,event_lora_checkpoint=str(source)))
    assert report['rng_restored'] and torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(v,model.vae.decoder.feature_fusion_blocks.state_dict()[k]) for k,v in expected.items())
    with pytest.raises(ValueError,match='seed/Stage3'):
        load_refinement_initialization(model,str(path),SimpleNamespace(seed=1,event_lora_checkpoint=str(source)))


def test_control_collation_preserves_nested_dataset_prompt_sample_axis():
    from train_idea3_decoder_refinement import collate_h1_refinement
    from osediff_event_film import _normalize_prompt_batch
    # 数据集每个 prompt 是单元素列表；四个样本不能被转置成一个四提示词样本。
    samples = [dict(conditioning_pixel_values=torch.zeros(3,4,4), events=torch.zeros(5,4,4),
                    prompt=[f'caption {i}'], neg_prompt='') for i in range(4)]
    batch = collate_h1_refinement(samples)
    assert batch['conditioning_pixel_values'].shape[0] == len(batch['prompt']) == 4
    for index in range(4):
        assert _normalize_prompt_batch([batch['prompt'][index]]) == [f'caption {index}']
