import os

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel

from osediff import initialize_unet, initialize_vae
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from safetensors.torch import load_file
import json

from models.event_encoder import EventEncoder
from models.event_lora_utils import (
    clear_event_feats_for_all_event_lora,
    collect_event_lora_debug,
    copy_compatible_state_dict,
    event_film_state_keys,
    replace_self_attn_to_v_with_event_film_lora,
    set_event_feats_for_all_event_lora,
    set_event_lora_trainable,
    set_modulation_enabled_for_all_event_lora,
)


def _normalize_prompt_batch(prompt_batch):
    if prompt_batch is None:
        return []
    if isinstance(prompt_batch, str):
        return [prompt_batch]
    if len(prompt_batch) > 0 and isinstance(prompt_batch[0], (list, tuple)):
        return [p[0] for p in prompt_batch]
    return list(prompt_batch)


def _first_parameter_dtype(module, default=torch.float32):
    """返回模块第一个参数的 dtype，用于对齐混合精度边界。"""
    for param in module.parameters(recurse=True):
        return param.dtype
    return default


def _unet_cross_attention_dtype(unet, default=torch.float32):
    """
    推理/训练时 cross-attention 的 text embedding dtype 必须匹配 attn2.to_k/to_v。

    PEFT LoRA wrapper 中 base_layer 可能仍保持 float32，也可能被外层 .to(fp16)
    转成 float16；这里读取实际权重 dtype，避免把 prompt_embeds 固定成某一种精度。
    """
    for name, module in unet.named_modules():
        if name.endswith("attn2.to_k"):
            return _first_parameter_dtype(module, default=default)
    return default


def _load_compatible_named_parameters(module, checkpoint_state):
    loaded, skipped = 0, 0
    named = dict(module.named_parameters())
    for name, tensor in checkpoint_state.items():
        if name in named and named[name].shape == tensor.shape:
            named[name].data.copy_(tensor.to(dtype=named[name].dtype))
            loaded += 1
        else:
            skipped += 1
    return loaded, skipped


class OSEDiffEventFiLMGen(torch.nn.Module):
    """
    Event-FiLM LoRA Generator。

    作用：
    - 复用原 OSEDiff 的 VAE、Text Encoder、UNet 与 scheduler；
    - 保留完整 Stage1 可兼容权重；
    - 将 self-attention 的 attn1.to_v 替换成事件调制 LoRA。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        self.vae, self.lora_vae_modules_encoder = initialize_vae(args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(args)
        self.event_encoder = EventEncoder(
            in_channels=getattr(args, "event_channels", 5),
            base_channels=getattr(args, "event_base_channels", 32),
        )
        self.event_film_installed = False
        self.event_film_replaced_names = []
        self.lora_rank_unet = args.lora_rank
        self.lora_rank_vae = args.lora_rank

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.event_encoder.to("cuda")
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

    def install_event_film_lora(self):
        if self.event_film_installed:
            return 0, self.event_film_replaced_names
        num_replaced, names = replace_self_attn_to_v_with_event_film_lora(
            self.unet,
            rank=self.args.lora_rank,
            modulation_scale=getattr(self.args, "modulation_scale", 0.1),
            mod_channels=getattr(self.args, "mod_channels", 64),
            event_channels_by_size=self.event_encoder.out_channels_by_size,
            mod_num_heads=getattr(self.args, "mod_num_heads", 4),
            mod_window_size=getattr(self.args, "mod_window_size", 8),
        )
        self.event_film_installed = True
        self.event_film_replaced_names = names
        print(f"[Event-FiLM] Replaced {num_replaced} self-attn to_v layer(s).")
        for name in names[:20]:
            print(f"[Event-FiLM]   {name}")
        return num_replaced, names

    def load_stage1_full(self, checkpoint_path):
        """
        完整加载 Stage1 中可兼容权重，然后迁移 attn1.to_v LoRA 到 Event-FiLM 模块。

        加载顺序很关键：
        1. 先把 Stage1 的 PEFT LoRA/conv_in/VAE LoRA 加到原始结构；
        2. 再替换 attn1.to_v，此时替换工具会复制已经加载好的 LoRA A/B。
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        unet_loaded, unet_skipped = _load_compatible_named_parameters(
            self.unet, checkpoint.get("state_dict_unet", {})
        )
        vae_loaded, vae_skipped = _load_compatible_named_parameters(
            self.vae, checkpoint.get("state_dict_vae", {})
        )
        num_replaced, _ = self.install_event_film_lora()
        print(
            "[Stage1 Load] "
            f"unet_loaded={unet_loaded}, unet_skipped={unet_skipped}, "
            f"vae_loaded={vae_loaded}, vae_skipped={vae_skipped}, "
            f"event_film_initialized={num_replaced}"
        )
        return checkpoint

    def set_train_stage(self, stage):
        self.unet.train()
        self.vae.train()
        self.event_encoder.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.event_encoder.requires_grad_(stage in {2, 3, "2", "3"})
        if stage in {2, "2"}:
            set_event_lora_trainable(self.unet, train_lora=False, train_modulation=True)
            set_modulation_enabled_for_all_event_lora(self.unet, True)
        elif stage in {3, "3"}:
            set_event_lora_trainable(self.unet, train_lora=True, train_modulation=True)
            set_modulation_enabled_for_all_event_lora(self.unet, True)
        else:
            self.event_encoder.requires_grad_(False)
            set_event_lora_trainable(self.unet, train_lora=True, train_modulation=False)
            set_modulation_enabled_for_all_event_lora(self.unet, False)

    def get_lora_params(self):
        params = []
        for name, param in self.unet.named_parameters():
            if param.requires_grad and ("lora_down" in name or "lora_up" in name):
                params.append(param)
        return params

    def get_event_params(self):
        params = []
        params += [p for p in self.event_encoder.parameters() if p.requires_grad]
        for name, param in self.unet.named_parameters():
            if param.requires_grad and not ("lora_down" in name or "lora_up" in name):
                params.append(param)
        return params

    def encode_prompt(self, prompt_batch):
        prompt_batch = _normalize_prompt_batch(prompt_batch)
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                prompt_embeds = self.text_encoder(text_input_ids.to(self.text_encoder.device))[0]
                prompt_embeds_list.append(prompt_embeds)
        return torch.cat(prompt_embeds_list, dim=0)

    def forward(self, c_t, events, timesteps, batch=None):
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        cross_attn_dtype = _unet_cross_attention_dtype(self.unet, default=encoded_control.dtype)

        # 事件特征只在 UNet forward 期间放入 LoRA 缓存，避免跨 batch 残留错误条件。
        event_feats = self.event_encoder(events)
        set_event_feats_for_all_event_lora(self.unet, event_feats)
        try:
            model_pred = self.unet(
                encoded_control,
                timesteps,
                # prompt_embeds 是 cross-attention 的输入，必须跟 attn2.to_k/to_v
                # 的实际权重 dtype 一致，否则 fp16 推理会在 PEFT LoRA Linear 报错。
                encoder_hidden_states=prompt_embeds.to(dtype=cross_attn_dtype),
            ).sample
        finally:
            clear_event_feats_for_all_event_lora(self.unet)

        x_denoised = self.noise_scheduler.step(
            model_pred, timesteps, encoded_control, return_dict=True
        ).prev_sample
        vae_dtype = _first_parameter_dtype(self.vae, default=x_denoised.dtype)
        output_image = (
            # VAE decode 也处在混合精度边界，latent dtype 跟随 VAE 权重。
            self.vae.decode(x_denoised.to(dtype=vae_dtype) / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)
        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def debug_stats(self):
        return collect_event_lora_debug(self.unet)

    def save_model(
        self,
        outf,
        optimizer=None,
        lr_scheduler=None,
        optimizer_reg=None,
        lr_scheduler_reg=None,
        global_step=0,
        epoch=0,
        stage=2,
        timestep_config=None,
        additional_info=None,
    ):
        state_unet = {}
        keep_markers = ("lora", "conv_in") + event_film_state_keys()
        for key, value in self.unet.state_dict().items():
            if any(marker in key for marker in keep_markers):
                state_unet[key] = value
        sd = {
            "vae_lora_encoder_modules": self.lora_vae_modules_encoder,
            "unet_lora_encoder_modules": self.lora_unet_modules_encoder,
            "unet_lora_decoder_modules": self.lora_unet_modules_decoder,
            "unet_lora_others_modules": self.lora_unet_others,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": state_unet,
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k},
            "state_dict_event_encoder": self.event_encoder.state_dict(),
            "event_film_replaced_names": self.event_film_replaced_names,
            "stage": stage,
            "timestep_config": timestep_config or {},
            "global_step": global_step,
            "epoch": epoch,
        }
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

    def load_event_checkpoint(self, checkpoint):
        self.install_event_film_lora()
        unet_loaded, unet_skipped = copy_compatible_state_dict(
            self.unet, checkpoint.get("state_dict_unet", {})
        )
        vae_loaded, vae_skipped = copy_compatible_state_dict(
            self.vae, checkpoint.get("state_dict_vae", {})
        )
        enc_loaded, enc_skipped = copy_compatible_state_dict(
            self.event_encoder, checkpoint.get("state_dict_event_encoder", {})
        )
        print(
            "[Event-FiLM Load] "
            f"unet_loaded={unet_loaded}, unet_skipped={unet_skipped}, "
            f"vae_loaded={vae_loaded}, vae_skipped={vae_skipped}, "
            f"event_encoder_loaded={enc_loaded}, event_encoder_skipped={enc_skipped}"
        )


class OSEDiffEventFiLMReg(torch.nn.Module):
    """
    fine-tuned regularizer。

    作用：
    沿用 OSEDiff 的 VSD/diff loss 路径，但 timestep 由训练 loop 传入，
    保证每个 step 中 Generator 和 regularizer 使用同一个 timestep 值。
    """

    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        vae_dir = os.path.join(args.pretrained_model_name_or_path, "vae")
        with open(os.path.join(vae_dir, "config.json"), "r") as f:
            vae_config = json.load(f)
        self.vae = AutoencoderKL(
            in_channels=vae_config.get("in_channels", 3),
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
        self.vae.load_state_dict(load_file(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")))

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
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(args)
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
        for name, param in self.unet_update.named_parameters():
            param.requires_grad = "lora" in name

    def load_ckpt(self, checkpoint):
        state = checkpoint.get("state_dict_reg", checkpoint.get("state_dict_unet", {}))
        loaded, skipped = _load_compatible_named_parameters(self.unet_update, state)
        print(f"[Reg Load] loaded={loaded}, skipped={skipped}")

    def state_dict_reg(self):
        return {
            k: v for k, v in self.unet_update.state_dict().items()
            if "lora" in k
        }

    def diff_loss(self, latents, prompt_embeds, timesteps):
        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        t_batch = timesteps.expand(bsz).to(latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, t_batch)
        noise_pred = self.unet_update(
            noisy_latents,
            timestep=t_batch,
            encoder_hidden_states=prompt_embeds,
        ).sample
        return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, timesteps, cfg_vsd):
        bsz = latents.shape[0]
        t_batch = timesteps.expand(bsz).to(latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, t_batch)
        with torch.no_grad():
            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=t_batch,
                encoder_hidden_states=prompt_embeds.float(),
            ).sample
            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, t_batch)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([t_batch] * 2)
            prompt_embeds_input = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds_input.to(dtype=self.weight_dtype),
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + cfg_vsd * (noise_pred_text - noise_pred_uncond)
            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, t_batch)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True).clamp_min(1e-6)
        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        return F.mse_loss(latents, (latents - grad).detach())


class OSEDiffEventFiLMTest(torch.nn.Module):
    """
    Event-FiLM 推理模型。

    推理只使用 Generator；regularizer 只在训练阶段提供 VSD 约束。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = OSEDiffEventFiLMGen(args)
        checkpoint = torch.load(args.osediff_path, map_location="cpu")
        self.gen.load_event_checkpoint(checkpoint)
        self.gen.eval()
        self.weight_dtype = torch.float16 if getattr(args, "mixed_precision", "fp16") == "fp16" else torch.float32
        self.gen.unet.to(self.device, dtype=self.weight_dtype)
        self.gen.vae.to(self.device, dtype=self.weight_dtype)
        self.gen.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.gen.event_encoder.to(self.device, dtype=self.weight_dtype)
        self.timesteps = torch.tensor([getattr(args, "generator_timestep", 999)], device=self.device).long()
        print(f"[INFO] Generator inference timestep: {self.timesteps.item()}")

    @torch.no_grad()
    def forward(self, lq, events, prompt):
        batch = {
            "prompt": [prompt],
            "neg_prompt": [getattr(self.args, "neg_prompt", "")],
        }
        out, _, _, _ = self.gen(
            lq.to(self.device, dtype=self.weight_dtype),
            events.to(self.device, dtype=self.weight_dtype),
            self.timesteps,
            batch=batch,
        )
        return out

    def debug_stats(self):
        return self.gen.debug_stats()
