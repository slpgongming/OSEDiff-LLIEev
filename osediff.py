import os
import sys
sys.path.append(os.getcwd())
import json
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig

from my_utils.vaehook import VAEHook, perfcount

def initialize_vae(args, vae_in_channels=None, add_lora=True):
    vae_dir = os.path.join(args.pretrained_model_name_or_path, "vae")
    with open(os.path.join(vae_dir, "config.json"), "r") as f:
        vae_config = json.load(f)
    in_channels = vae_in_channels if vae_in_channels is not None else vae_config.get("in_channels", 3)
    vae = AutoencoderKL(
        in_channels=in_channels,
        out_channels=vae_config.get("out_channels", 3),
        down_block_types=tuple(vae_config["down_block_types"]),
        up_block_types=tuple(vae_config["up_block_types"]),
        block_out_channels=tuple(vae_config["block_out_channels"]),
        layers_per_block=vae_config.get("layers_per_block", 2),
        act_fn=vae_config.get("act_fn", "silu"),
        latent_channels=vae_config.get("latent_channels", 4),
        norm_num_groups=vae_config.get("norm_num_groups", 32),
        sample_size=vae_config.get("sample_size", 768),
    )
    vae_state = load_file(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"))

    # Expand encoder conv_in BEFORE loading: pretrained weights are for 3 channels,
    # but we need to accept `in_channels` (e.g. 8 = 3 RGB + 5 voxel).
    pretrained_in_ch = vae_config.get("in_channels", 3)
    if in_channels != pretrained_in_ch:
        conv_in_key = "encoder.conv_in.weight"
        if conv_in_key in vae_state:
            old_weight = vae_state[conv_in_key]  # [out_ch, pretrained_in_ch, k, k]
            out_ch, _, k_h, k_w = old_weight.shape
            mean_per_out = old_weight.mean(dim=1, keepdim=True)  # [out_ch, 1, k, k]
            repeat_times = in_channels - pretrained_in_ch
            new_channels_weight = mean_per_out.repeat(1, repeat_times, 1, 1)
            expanded_weight = torch.cat([old_weight, new_channels_weight], dim=1)
            vae_state[conv_in_key] = expanded_weight
            print(f"[initialize_vae] Expanded conv_in: {pretrained_in_ch} -> {in_channels} channels "
                  f"(new channels initialized from mean of original {pretrained_in_ch} channels)")

    vae.load_state_dict(vae_state, strict=False)

    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    
    if add_lora:
        lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
        vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet_dir = os.path.join(args.pretrained_model_name_or_path, "unet")
    with open(os.path.join(unet_dir, "config.json"), "r") as f:
        unet_config = json.load(f)
    unet_in_channels = getattr(args, 'unet_in_channels', None)
    in_channels = unet_in_channels if unet_in_channels is not None else unet_config.get("in_channels", 4)
    unet = UNet2DConditionModel(
        in_channels=in_channels,
        out_channels=unet_config.get("out_channels", 4),
        down_block_types=tuple(unet_config["down_block_types"]),
        up_block_types=tuple(unet_config["up_block_types"]),
        block_out_channels=tuple(unet_config["block_out_channels"]),
        layers_per_block=unet_config.get("layers_per_block", 2),
        cross_attention_dim=unet_config.get("cross_attention_dim", 1024),
        attention_head_dim=unet_config.get("attention_head_dim"),
        sample_size=unet_config.get("sample_size", 64),
        use_linear_projection=unet_config.get("use_linear_projection", False),
    )
    unet_state = load_file(os.path.join(unet_dir, "diffusion_pytorch_model.safetensors"))

    # Expand conv_in BEFORE loading: pretrained conv_in is 4 channels (latent only),
    # but we need `in_channels` (e.g. 9 = 4 latent + 5 voxel).
    pretrained_unet_in_ch = unet_config.get("in_channels", 4)
    if in_channels != pretrained_unet_in_ch:
        conv_in_key = "conv_in.weight"
        if conv_in_key in unet_state:
            old_weight = unet_state[conv_in_key]  # [out_ch, 4, k, k]
            out_ch, _, k_h, k_w = old_weight.shape
            mean_per_out = old_weight.mean(dim=1, keepdim=True)  # [out_ch, 1, k, k]
            repeat_times = in_channels - pretrained_unet_in_ch
            new_channels_weight = mean_per_out.repeat(1, repeat_times, 1, 1)
            expanded_weight = torch.cat([old_weight, new_channels_weight], dim=1)
            unet_state[conv_in_key] = expanded_weight
            print(f"[initialize_unet] Expanded conv_in: {pretrained_unet_in_ch} -> {in_channels} channels "
                  f"(new channels initialized from mean of original {pretrained_unet_in_ch} channels)")

    unet.load_state_dict(unet_state, strict=False)
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


class OSEDiff_gen(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        vae_in_channels = getattr(args, 'vae_in_channels', None)
        self.vae, self.lora_vae_modules_encoder = initialize_vae(self.args, vae_in_channels=vae_in_channels)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args)
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def forward(self, c_t, batch=None, args=None):

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])

        # Random small timestep sampling for stable training
        # Support three modes for args.train_timestep_range:
        #   [(low, high, weight), ...]  — pick a bucket by weight, then sample uniformly inside
        # Use scalar (1,) to avoid DDPMScheduler.step() batch-size mismatch error
        ranges = getattr(args, 'train_timestep_range', [(0, 200, 1.0)])
        if len(ranges) == 1:
            low, high, _ = ranges[0]
            timesteps = torch.randint(low, high + 1, (1,), device=c_t.device).long()
        else:
            lows  = torch.tensor([r[0] for r in ranges], dtype=torch.float32, device=c_t.device)
            highs = torch.tensor([r[1] for r in ranges], dtype=torch.float32, device=c_t.device)
            weights = torch.tensor([r[2] for r in ranges], dtype=torch.float32, device=c_t.device)
            weights = weights / weights.sum()
            bucket_idx = torch.multinomial(weights, 1, replacement=True).item()
            low, high = lows[bucket_idx].item(), highs[bucket_idx].item()
            timesteps = torch.randint(int(low), int(high) + 1, (1,), device=c_t.device).long()

        model_pred = self.unet(encoded_control, timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample
        x_denoised = self.noise_scheduler.step(model_pred, timesteps, encoded_control, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def load_ckpt(self, checkpoint):
        # Load UNet LoRA weights
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                if n in checkpoint["state_dict_unet"]:
                    p.data.copy_(checkpoint["state_dict_unet"][n])

        # Load VAE LoRA weights
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                if n in checkpoint["state_dict_vae"]:
                    p.data.copy_(checkpoint["state_dict_vae"][n])

    def save_model(self, outf, optimizer=None, lr_scheduler=None, global_step=0,
                   epoch=0, optimizer_reg=None, lr_scheduler_reg=None, additional_info=None):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["global_step"] = global_step
        sd["epoch"] = epoch
        if optimizer is not None:
            sd["optimizer_state_dict"] = optimizer.state_dict()
        if lr_scheduler is not None:
            sd["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        if optimizer_reg is not None:
            sd["optimizer_reg_state_dict"] = optimizer_reg.state_dict()
        if lr_scheduler_reg is not None:
            sd["lr_scheduler_reg_state_dict"] = lr_scheduler_reg.state_dict()
        if additional_info is not None:
            sd["additional_info"] = additional_info
        torch.save(sd, outf)

class OSEDiff_reg(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        vae_in_channels = getattr(args, 'vae_in_channels', None)
        self.vae, _ = initialize_vae(args, vae_in_channels=vae_in_channels)
        unet_dir = os.path.join(args.pretrained_model_name_or_path, "unet")
        with open(os.path.join(unet_dir, "config.json"), "r") as f:
            unet_config = json.load(f)
        self.unet_fix = UNet2DConditionModel(
            in_channels=unet_config.get("in_channels", 4),
            out_channels=unet_config.get("out_channels", 4),
            down_block_types=tuple(unet_config["down_block_types"]),
            up_block_types=tuple(unet_config["up_block_types"]),
            block_out_channels=tuple(unet_config["block_out_channels"]),
            layers_per_block=unet_config.get("layers_per_block", 2),
            cross_attention_dim=unet_config.get("cross_attention_dim", 1024),
            attention_head_dim=unet_config.get("attention_head_dim"),
            sample_size=unet_config.get("sample_size", 64),
            use_linear_projection=unet_config.get("use_linear_projection", False),
        )
        self.unet_fix.load_state_dict(load_file(os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")))
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others =\
                initialize_unet(args)
        self.lora_rank_unet = args.lora_rank

        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_update.to(accelerator.device)
        self.vae.to(accelerator.device)
        
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def load_ckpt(self, checkpoint):
        # Load UNet LoRA weights (adapters already exist from __init__)
        for n, p in self.unet_update.named_parameters():
            if "lora" in n:
                if n in checkpoint["state_dict_reg"]:
                    p.data.copy_(checkpoint["state_dict_reg"][n])

    def diff_loss(self, latents, prompt_embeds, args):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet_update(
        noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args):
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():

            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + args.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss = F.mse_loss(latents, (latents - grad).detach())

        return loss

class OSEDiff_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        vae_in_channels = getattr(args, 'vae_in_channels', None)
        self.vae, _ = initialize_vae(args, vae_in_channels=vae_in_channels, add_lora=False)
        unet_dir = os.path.join(self.args.pretrained_model_name_or_path, "unet")
        with open(os.path.join(unet_dir, "config.json"), "r") as f:
            unet_config = json.load(f)
        self.unet = UNet2DConditionModel(
            in_channels=unet_config.get("in_channels", 4),
            out_channels=unet_config.get("out_channels", 4),
            down_block_types=tuple(unet_config["down_block_types"]),
            up_block_types=tuple(unet_config["up_block_types"]),
            block_out_channels=tuple(unet_config["block_out_channels"]),
            layers_per_block=unet_config.get("layers_per_block", 2),
            cross_attention_dim=unet_config.get("cross_attention_dim", 1024),
            attention_head_dim=unet_config.get("attention_head_dim"),
            sample_size=unet_config.get("sample_size", 64),
            use_linear_projection=unet_config.get("use_linear_projection", False),
        )
        self.unet.load_state_dict(load_file(os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")))

        # vae tile
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        inference_timestep = getattr(args, 'inference_timestep', 999)
        self.timesteps = torch.tensor([inference_timestep], device="cuda").long()
        print(f"[INFO] Inference timestep set to: {inference_timestep}")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

        lora_sum_unet = sum(p.abs().sum().item() for n, p in self.unet.named_parameters() if "lora" in n)
        lora_sum_vae = sum(p.abs().sum().item() for n, p in self.vae.named_parameters() if "lora" in n)
        print(f"[DEBUG] UNet LoRA weight abs-sum: {lora_sum_unet:.6f}")
        print(f"[DEBUG] VAE LoRA weight abs-sum: {lora_sum_vae:.6f}")

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    # @perfcount
    @torch.no_grad()
    def forward(self, lq, prompt):
        prompt_embeds = self.encode_prompt([prompt])
        print(f"[DEBUG] prompt_embeds dtype={prompt_embeds.dtype}, device={prompt_embeds.device}, shape={prompt_embeds.shape}")
        lq_latent = self.vae.encode(lq.to(self.weight_dtype).to(self.device)).latent_dist.sample() * self.vae.config.scaling_factor
        print(f"[DEBUG] lq_latent dtype={lq_latent.dtype}, mean={lq_latent.mean().item():.6f}, std={lq_latent.std().item():.6f}")
        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
            print(f"[DEBUG] model_pred dtype={model_pred.dtype}, mean={model_pred.mean().item():.6f}, std={model_pred.std().item():.6f}")
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        model_out = self.unet(input_list_t, self.timesteps, encoder_hidden_states=prompt_embeds.to(self.weight_dtype),).sample
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred
            print(f"[DEBUG] (tiled) model_pred dtype={model_pred.dtype}, mean={model_pred.mean().item():.6f}, std={model_pred.std().item():.6f}")

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        print(f"[DEBUG] x_denoised mean={x_denoised.mean().item():.6f}, std={x_denoised.std().item():.6f}, diff_from_input={torch.abs(x_denoised - lq_latent).mean().item():.6f}")
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype).to(self.device) / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        print(f"[DEBUG] output_image mean={output_image.mean().item():.6f}, std={output_image.std().item():.6f}")

        return output_image

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))


class OSEDiff_inference_time(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        vae_in_channels = getattr(args, 'vae_in_channels', None)
        self.vae, _ = initialize_vae(args, vae_in_channels=vae_in_channels, add_lora=False)
        unet_dir = os.path.join(self.args.pretrained_model_name_or_path, "unet")
        with open(os.path.join(unet_dir, "config.json"), "r") as f:
            unet_config = json.load(f)
        self.unet = UNet2DConditionModel(
            in_channels=unet_config.get("in_channels", 4),
            out_channels=unet_config.get("out_channels", 4),
            down_block_types=tuple(unet_config["down_block_types"]),
            up_block_types=tuple(unet_config["up_block_types"]),
            block_out_channels=tuple(unet_config["block_out_channels"]),
            layers_per_block=unet_config.get("layers_per_block", 2),
            cross_attention_dim=unet_config.get("cross_attention_dim", 1024),
            attention_head_dim=unet_config.get("attention_head_dim"),
            sample_size=unet_config.get("sample_size", 64),
            use_linear_projection=unet_config.get("use_linear_projection", False),
        )
        self.unet.load_state_dict(load_file(os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")))

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        inference_timestep = getattr(args, 'inference_timestep', 200)
        self.timesteps = torch.tensor([inference_timestep], device="cuda").long()
        print(f"[INFO] Inference timestep set to: {inference_timestep}")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

        lora_sum_unet = sum(p.abs().sum().item() for n, p in self.unet.named_parameters() if "lora" in n)
        lora_sum_vae = sum(p.abs().sum().item() for n, p in self.vae.named_parameters() if "lora" in n)
        print(f"[DEBUG] UNet LoRA weight abs-sum: {lora_sum_unet:.6f}")
        print(f"[DEBUG] VAE LoRA weight abs-sum: {lora_sum_vae:.6f}")

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    @torch.no_grad()
    def forward(self, lq, prompt):

        prompt_embeds = self.encode_prompt([prompt])
        lq_latent = self.vae.encode(lq.to(self.weight_dtype).to(self.device)).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype).to(self.device) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image
