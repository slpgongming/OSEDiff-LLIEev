"""无H1对照：冻结Stage3保持FP32，refinement沿用外层fp16。"""
import torch
from osediff_event_film import OSEDiffEventFiLMGen
from models.event_lora_utils import clear_event_feats_for_all_event_lora,iter_event_lora_modules


def frozen_stage3_latents(gen, rgb, events, timesteps, batch):
    """与H1路径保持逐样本encode及scheduler RNG作用域，仅省去结构观测/修正。"""
    values=[];devices=[rgb.device.index] if rgb.is_cuda else []
    with torch.no_grad(),torch.autocast(device_type=rgb.device.type,enabled=False):
        for index in range(rgb.shape[0]):
            context=gen.prepare_denoising_context(rgb[index:index+1].float(),events[index:index+1].float(),
                dict(prompt=[batch['prompt'][index]],neg_prompt=[batch['neg_prompt'][index]]))
            with torch.random.fork_rng(devices=devices):
                latent,_=gen.predict_denoised_latent_from_context(context,timesteps)
            values.append(latent.detach())
            clear_event_feats_for_all_event_lora(gen.unet)
            for _,layer in iter_event_lora_modules(gen.unet):
                layer.last_aux={};layer.last_visuals={};layer.last_z_delta=None
    return torch.cat(values).detach()


class NoH1DecoderRefinementGen(OSEDiffEventFiLMGen):
    """没有结构头或delta优化；只训练同一个decoder feature-fusion模块。"""
    def forward(self,c_t,events,timesteps,batch=None,events_refinement=None,
                decoder_fusion_enabled=None,refinement_fusion_mode=None):
        latent=frozen_stage3_latents(self,c_t,events,timesteps,batch)
        output=self.decode_denoised_latent(latent,c_t,events_refinement=events_refinement,
            decoder_fusion_enabled=decoder_fusion_enabled,refinement_fusion_mode=refinement_fusion_mode)
        return output,latent,None,None
