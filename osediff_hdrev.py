import json
import os

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import AutoTokenizer, CLIPTextModel

from models.autoencoder_kl import AutoencoderKL
from models.event_latent_projector import EventPyramidLatentProjector
from models.hdrev_encoder import HDRevEncoder
from osediff import initialize_unet


def initialize_frozen_vae(args):
    """Load the SD VAE without adding VAE LoRA; Stage 3 only uses it for GT latent targets and decoding."""
    vae_dir = os.path.join(args.pretrained_model_name_or_path, "vae")
    with open(os.path.join(vae_dir, "config.json"), "r") as f:
        vae_config = json.load(f)

    vae = AutoencoderKL(
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
    vae.load_state_dict(load_file(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")))
    vae.requires_grad_(False)
    vae.eval()
    return vae


def tensor_stats(tensor):
    """Compact numeric stats used to monitor latent distribution drift."""
    tensor = tensor.detach().float()
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


class OSEDiffHDRevGen(torch.nn.Module):
    """
    Stage 3 generator.

    HDRevEncoder + EventPyramidLatentProjector replaces VAE encoding of the
    low-light image. The SD VAE decoder remains the image-space decoder.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device=self.device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        self.vae = initialize_frozen_vae(args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(args)
        use_encoder_projector_pretrain = bool(getattr(args, "encoder_projector_pretrain_path", None))
        self.hdrev_encoder = HDRevEncoder(
            num_bins=args.num_bins,
            pretrained_root=args.hdrev_pretrained_root,
            pretrained_i_path=getattr(args, "hdrev_encoder_i_path", None),
            pretrained_e_path=getattr(args, "hdrev_encoder_e_path", None),
            pretrained_merge_path=getattr(args, "hdrev_merge_path", None),
            pretrained_d_path=getattr(args, "hdrev_decoder_path", None),
            # When clean-latent pretraining is provided, Stage 3 should be
            # initialized directly from that checkpoint. Skipping the original
            # HDRev load means Linux training only needs the pretrain checkpoint,
            # not the separate checkpoints/HDRev files.
            load_pretrained=not use_encoder_projector_pretrain,
        )
        self.event_latent_projector = EventPyramidLatentProjector(latent_channels=4)

        self.lora_rank_unet = args.lora_rank
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)
        self.hdrev_encoder.to(self.device)
        self.event_latent_projector.to(self.device)

    def set_train(self):
        self.unet.train()
        self.hdrev_encoder.train()
        self.event_latent_projector.train()
        self.vae.eval()
        self.text_encoder.eval()

        for param in self.unet.parameters():
            param.requires_grad = False
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        # Keep OSEDiff's original choice of updating conv_in with the LoRA path.
        self.unet.conv_in.requires_grad_(True)
        self.hdrev_encoder.requires_grad_(True)
        self.event_latent_projector.requires_grad_(True)

    def encode_prompt(self, prompt_batch):
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
        return torch.concat(prompt_embeds_list, dim=0)

    def _sample_training_timestep(self, args, device):
        ranges = getattr(args, "train_timestep_range", [(999, 999, 1.0)])
        if len(ranges) == 1:
            low, high, _ = ranges[0]
            return torch.randint(low, high + 1, (1,), device=device).long()

        weights = torch.tensor([r[2] for r in ranges], dtype=torch.float32, device=device)
        weights = weights / weights.sum()
        bucket_idx = torch.multinomial(weights, 1, replacement=True).item()
        low, high, _ = ranges[bucket_idx]
        return torch.randint(int(low), int(high) + 1, (1,), device=device).long()

    def encode_start_latent(self, hdrev_rgb, event_voxel):
        """
        Convert low-light RGB [0,1] and event voxel [B,num_bins,H,W] into a
        Stable Diffusion latent-shaped tensor [B,4,H/8,W/8].
        """
        hdrev_dtype = next(self.hdrev_encoder.parameters()).dtype
        projector_dtype = next(self.event_latent_projector.parameters()).dtype
        _, hdrev_features = self.hdrev_encoder(
            hdrev_rgb.to(device=self.device, dtype=hdrev_dtype),
            event_voxel.to(device=self.device, dtype=hdrev_dtype),
        )
        hdrev_features = [feat.to(dtype=projector_dtype) for feat in hdrev_features]
        return self.event_latent_projector(hdrev_features)

    @torch.no_grad()
    def encode_target_latent(self, image_minus1_1):
        """Encode GT image to the SD latent manifold for optional latent alignment."""
        vae_dtype = next(self.vae.parameters()).dtype
        posterior = self.vae.encode(image_minus1_1.to(device=self.device, dtype=vae_dtype)).latent_dist
        return posterior.sample() * self.vae.config.scaling_factor

    def forward(self, batch, args=None, return_target_latent=False):
        args = args or self.args
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])

        start_latent = self.encode_start_latent(batch["hdrev_pixel_values"], batch["event_voxel"])
        timesteps = self._sample_training_timestep(args, start_latent.device)

        model_pred = self.unet(
            start_latent,
            timesteps,
            encoder_hidden_states=prompt_embeds.to(dtype=torch.float32),
        ).sample
        x_denoised = self.noise_scheduler.step(model_pred, timesteps, start_latent, return_dict=True).prev_sample

        vae_dtype = next(self.vae.parameters()).dtype
        output_image = self.vae.decode(
            x_denoised.to(device=self.device, dtype=vae_dtype) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        target_latent = None
        if return_target_latent:
            target_latent = self.encode_target_latent(batch["output_pixel_values"])

        # Return model_pred and timesteps so the training loop can tell whether
        # NaNs first appear in the UNet prediction or inside scheduler.step().
        return (
            output_image,
            x_denoised,
            prompt_embeds,
            neg_prompt_embeds,
            start_latent,
            target_latent,
            model_pred,
            timesteps,
        )

    def load_ckpt(self, checkpoint, strict=True):
        for name, param in self.unet.named_parameters():
            if ("lora" in name or "conv_in" in name) and name in checkpoint["state_dict_unet"]:
                param.data.copy_(checkpoint["state_dict_unet"][name])

        self.hdrev_encoder.load_state_dict(checkpoint["state_dict_hdrev_encoder"], strict=strict)
        self.event_latent_projector.load_state_dict(checkpoint["state_dict_event_latent_projector"], strict=strict)

    def load_encoder_projector_pretrain(self, checkpoint, strict=True):
        """
        Initialize Stage 3 HDRev/projector from clean-latent pretraining.

        The pretrain checkpoint only contains the event-aware encoder and
        pyramid latent projector. UNet LoRA, VSD regularizer, scheduler state,
        and optimizer state remain freshly initialized for the Stage 3 run.
        """
        if "state_dict_hdrev_encoder" not in checkpoint:
            raise KeyError("Pretrain checkpoint is missing state_dict_hdrev_encoder.")
        if "state_dict_event_latent_projector" not in checkpoint:
            raise KeyError("Pretrain checkpoint is missing state_dict_event_latent_projector.")

        self.hdrev_encoder.load_state_dict(checkpoint["state_dict_hdrev_encoder"], strict=strict)
        self.event_latent_projector.load_state_dict(
            checkpoint["state_dict_event_latent_projector"],
            strict=strict,
        )

    def save_model(
        self,
        outf,
        optimizer=None,
        lr_scheduler=None,
        global_step=0,
        epoch=0,
        optimizer_reg=None,
        lr_scheduler_reg=None,
        additional_info=None,
    ):
        sd = {
            "stage": "osediff_lle_hdrev_stage3",
            "unet_lora_encoder_modules": self.lora_unet_modules_encoder,
            "unet_lora_decoder_modules": self.lora_unet_modules_decoder,
            "unet_lora_others_modules": self.lora_unet_others,
            "rank_unet": self.lora_rank_unet,
            "state_dict_unet": {
                k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k
            },
            "state_dict_hdrev_encoder": self.hdrev_encoder.state_dict(),
            "state_dict_event_latent_projector": self.event_latent_projector.state_dict(),
            "stage3_config": {
                "num_bins": self.args.num_bins,
                "event_norm": bool(getattr(self.args, "event_norm", True)),
                "image_size": self.args.image_size,
                "hdrev_pretrained_root": self.args.hdrev_pretrained_root,
                "lambda_latent_align": getattr(self.args, "lambda_latent_align", 0.05),
            },
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


class OSEDiffHDRevTest(torch.nn.Module):
    """Inference wrapper for Stage 3 checkpoints."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device=self.device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        self.vae = initialize_frozen_vae(args)
        self.unet, _, _, _ = initialize_unet(args)
        self.hdrev_encoder = HDRevEncoder(
            num_bins=args.num_bins,
            pretrained_root=args.hdrev_pretrained_root,
            # Inference checkpoints already contain the trained HDRevEncoder
            # weights. Avoid loading the original HDRev root here so testing
            # only depends on --osediff_path plus the SD base model.
            load_pretrained=False,
        )
        self.event_latent_projector = EventPyramidLatentProjector(latent_channels=4)

        checkpoint = torch.load(args.osediff_path, map_location="cpu")
        self.load_ckpt(checkpoint)

        self.text_encoder.to(self.device, dtype=self.weight_dtype).eval()
        self.vae.to(self.device, dtype=self.weight_dtype).eval()
        self.unet.to(self.device, dtype=self.weight_dtype).eval()
        self.hdrev_encoder.to(self.device, dtype=self.weight_dtype).eval()
        self.event_latent_projector.to(self.device, dtype=self.weight_dtype).eval()

        inference_timestep = getattr(args, "inference_timestep", 999)
        self.timesteps = torch.tensor([inference_timestep], device=self.device).long()

    def load_ckpt(self, checkpoint):
        if checkpoint.get("stage") != "osediff_lle_hdrev_stage3":
            raise ValueError("This is not a Stage 3 HDRev OSEDiff checkpoint.")

        lora_conf_encoder = LoraConfig(
            r=checkpoint["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=checkpoint["unet_lora_encoder_modules"],
        )
        lora_conf_decoder = LoraConfig(
            r=checkpoint["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=checkpoint["unet_lora_decoder_modules"],
        )
        lora_conf_others = LoraConfig(
            r=checkpoint["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=checkpoint["unet_lora_others_modules"],
        )

        # initialize_unet already adds adapters in training, but inference needs
        # to ensure adapter names exist before copying Stage 3 checkpoint weights.
        existing = getattr(self.unet, "peft_config", {})
        if "default_encoder" not in existing:
            self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        if "default_decoder" not in existing:
            self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        if "default_others" not in existing:
            self.unet.add_adapter(lora_conf_others, adapter_name="default_others")

        for name, param in self.unet.named_parameters():
            if ("lora" in name or "conv_in" in name) and name in checkpoint["state_dict_unet"]:
                param.data.copy_(checkpoint["state_dict_unet"][name])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        self.hdrev_encoder.load_state_dict(checkpoint["state_dict_hdrev_encoder"])
        self.event_latent_projector.load_state_dict(checkpoint["state_dict_event_latent_projector"])

    def encode_prompt(self, prompt_batch):
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
        return torch.concat(prompt_embeds_list, dim=0)

    @torch.no_grad()
    def forward(self, hdrev_rgb, event_voxel, prompt):
        prompt_embeds = self.encode_prompt([prompt])
        _, hdrev_features = self.hdrev_encoder(
            hdrev_rgb.to(self.device, dtype=self.weight_dtype),
            event_voxel.to(self.device, dtype=self.weight_dtype),
        )
        start_latent = self.event_latent_projector(hdrev_features)
        # Keep text conditioning in the same precision as the inference UNet.
        # This mirrors the original OSEDiff fp16 inference path and avoids
        # mixed dtype attention inputs when --mixed_precision fp16 is used.
        model_pred = self.unet(
            start_latent,
            self.timesteps,
            encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
        ).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, start_latent, return_dict=True).prev_sample
        vae_dtype = next(self.vae.parameters()).dtype
        # DDPMScheduler arithmetic can promote the denoised latent back to
        # float32 even when model_pred/start_latent are fp16. Cast immediately
        # before VAE decode so z matches the fp16 post_quant_conv weights.
        return self.vae.decode(
            x_denoised.to(device=self.device, dtype=vae_dtype) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)
