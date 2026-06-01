import copy
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel

from models.lle_encoder import LLEEncoder
from osediff import initialize_unet, initialize_vae


def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


def sample_train_timesteps(train_timestep_arg: str, batch_size: int, device: torch.device) -> torch.Tensor:
    # 支持 "500" 固定 timestep，也支持 "400,600" 区间随机采样。
    if "," in str(train_timestep_arg):
        t_min, t_max = [int(value.strip()) for value in str(train_timestep_arg).split(",", 1)]
        return torch.randint(t_min, t_max + 1, (batch_size,), device=device, dtype=torch.long)
    timestep = int(train_timestep_arg)
    return torch.full((batch_size,), timestep, device=device, dtype=torch.long)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _copy_matching_named_parameters(module: nn.Module, state_dict: Dict[str, torch.Tensor], keywords: Sequence[str]) -> Dict[str, int]:
    loaded = 0
    skipped = 0
    for name, param in module.named_parameters():
        if not any(keyword in name for keyword in keywords):
            continue
        if name in state_dict and tuple(state_dict[name].shape) == tuple(param.shape):
            param.data.copy_(state_dict[name])
            loaded += 1
        else:
            skipped += 1
    return {"loaded": loaded, "skipped": skipped}


class PyramidFusion(nn.Module):
    """
    多尺度 HDR feature list -> latent-resolution condition embedding.

    输入来自 LLEEncoder: [H1,H2,H3,H4]，每级 concat 前都会 resize 到当前 embedding 尺寸。
    conv_out 使用 zero init，使刚初始化时事件条件对主干影响接近 0。
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: Tuple[int, ...] = (64, 128, 256, 512),
        block_out_channels: Tuple[int, ...] = (128, 128, 256, 512),
    ):
        super().__init__()
        if len(conditioning_channels) != 4:
            raise ValueError("PyramidFusion expects four LLE feature levels.")
        self.conditioning_channels = conditioning_channels
        self.conv_in = nn.Conv2d(conditioning_channels[0], block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i] + conditioning_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2),
                nn.SiLU(),
            ))
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1] + conditioning_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, hdr_features: Sequence[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        if not isinstance(hdr_features, (list, tuple)) or len(hdr_features) != 4:
            raise ValueError("PyramidFusion input must be a four-level list/tuple from LLEEncoder.")

        embedding = F.silu(self.conv_in(hdr_features[0]))
        for idx, block in enumerate(self.blocks):
            condition = hdr_features[idx]
            if condition.shape[2:] != embedding.shape[2:]:
                condition = F.interpolate(condition, size=embedding.shape[2:], mode="bilinear", align_corners=False)
            embedding = block(torch.cat([embedding, condition], dim=1))

        last_condition = hdr_features[-1]
        if last_condition.shape[2:] != embedding.shape[2:]:
            last_condition = F.interpolate(last_condition, size=embedding.shape[2:], mode="bilinear", align_corners=False)

        condition_embedding = self.conv_out(torch.cat([embedding, last_condition], dim=1))
        if condition_embedding.shape[2:] != target_size:
            condition_embedding = F.interpolate(condition_embedding, size=target_size, mode="bilinear", align_corners=False)
        return condition_embedding


def _resnet_out_channels(resnet: nn.Module) -> int:
    if hasattr(resnet, "out_channels"):
        return int(resnet.out_channels)
    if hasattr(resnet, "conv2") and hasattr(resnet.conv2, "out_channels"):
        return int(resnet.conv2.out_channels)
    raise AttributeError(f"Cannot infer output channels from resnet: {resnet.__class__.__name__}")


def _conv_out_channels(conv: nn.Module) -> int:
    if hasattr(conv, "out_channels"):
        return int(conv.out_channels)
    if hasattr(conv, "base_layer") and hasattr(conv.base_layer, "out_channels"):
        return int(conv.base_layer.out_channels)
    if hasattr(conv, "out_features"):
        return int(conv.out_features)
    raise AttributeError(f"Cannot infer conv output channels from: {conv.__class__.__name__}")


class ControlModule(nn.Module):
    """
    ControlNet-like down/mid branch copied from the frozen UNet down path.

    It receives the low-light latent and PyramidFusion condition embedding, then returns residual tensors whose
    shapes match UNet down_block_res_samples and mid block sample.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.conv_in = copy.deepcopy(unet.conv_in)
        self.time_proj = copy.deepcopy(unet.time_proj)
        self.time_embedding = copy.deepcopy(unet.time_embedding)
        self.class_embedding = copy.deepcopy(getattr(unet, "class_embedding", None))
        self.add_embedding = copy.deepcopy(getattr(unet, "add_embedding", None))
        self.down_blocks = copy.deepcopy(unet.down_blocks)
        self.mid_block = copy.deepcopy(unet.mid_block)

        residual_channels = [_conv_out_channels(unet.conv_in)]
        for down_block in unet.down_blocks:
            for resnet in down_block.resnets:
                residual_channels.append(_resnet_out_channels(resnet))
            if getattr(down_block, "downsamplers", None) is not None:
                residual_channels.append(residual_channels[-1])

        self.controlnet_down_blocks = nn.ModuleList([
            zero_module(nn.Conv2d(channels, channels, kernel_size=1))
            for channels in residual_channels
        ])
        self.controlnet_mid_block = zero_module(nn.Conv2d(residual_channels[-1], residual_channels[-1], kernel_size=1))

    def forward(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_condition: torch.Tensor,
        conditioning_scale: float = 1.0,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.shape[0] != sample.shape[0]:
            timesteps = timesteps[:1].expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        hidden = self.conv_in(sample)
        if control_condition.shape[2:] != hidden.shape[2:]:
            control_condition = F.interpolate(control_condition, size=hidden.shape[2:], mode="bilinear", align_corners=False)
        hidden = hidden + control_condition.to(dtype=hidden.dtype)

        down_block_res_samples = (hidden,)
        for down_block in self.down_blocks:
            if getattr(down_block, "has_cross_attention", False):
                hidden, res_samples = down_block(
                    hidden_states=hidden,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                )
            else:
                hidden, res_samples = down_block(hidden_states=hidden, temb=emb)
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            if getattr(self.mid_block, "has_cross_attention", False):
                hidden = self.mid_block(
                    hidden,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                )
            else:
                hidden = self.mid_block(hidden, emb)

        down_residuals = tuple(
            control_block(res_sample) * conditioning_scale
            for control_block, res_sample in zip(self.controlnet_down_blocks, down_block_res_samples)
        )
        mid_residual = self.controlnet_mid_block(hidden) * conditioning_scale
        return down_residuals, mid_residual


class EventGuidedOSEDiff(nn.Module):
    def __init__(self, args, stage: str = "stage1", device: Optional[torch.device] = None):
        super().__init__()
        self.args = args
        self.stage = stage
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.float32
        if getattr(args, "mixed_precision", "no") == "fp16" and self.device.type == "cuda":
            self.weight_dtype = torch.float16
        elif getattr(args, "mixed_precision", "no") == "bf16" and self.device.type == "cuda":
            self.weight_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device=self.device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        self.vae, self.lora_vae_modules_encoder = initialize_vae(args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(args)
        self.lora_rank_unet = args.lora_rank
        self.lora_rank_vae = args.lora_rank

        self.lle_encoder = LLEEncoder(num_bins=args.num_bins, use_decoder=False)
        self.pyramid_fusion = PyramidFusion(conditioning_embedding_channels=_conv_out_channels(self.unet.conv_in))
        self.control_module = ControlModule(self.unet)

        self.control_alpha_init = float(getattr(args, "control_alpha_init", 0.0))
        self.control_alpha_final = float(getattr(args, "control_alpha_final", 1.0))
        self.control_alpha_warmup_steps = max(1, int(getattr(args, "control_alpha_warmup_steps", 5000)))
        if getattr(args, "use_learnable_control_alpha", False):
            self.learnable_control_alpha = nn.Parameter(torch.tensor(self.control_alpha_init, dtype=torch.float32))
        else:
            self.learnable_control_alpha = None

        self.to(self.device)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        if getattr(args, "pretrained_osediff_lora_path", None):
            self.load_pretrained_osediff_lora(args.pretrained_osediff_lora_path)
        if getattr(args, "pretrained_lle_encoder_path", None):
            self.lle_encoder.load_pretrained(args.pretrained_lle_encoder_path)
        self.configure_trainable(stage)

    def configure_trainable(self, stage: str) -> None:
        # base weights 始终冻结；Stage 1 冻结 LoRA，Stage 2 解冻 LoRA。
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.control_module.train()
        self.pyramid_fusion.train()
        self.lle_encoder.train()
        for param in self.lle_encoder.parameters():
            param.requires_grad = True
        for param in self.pyramid_fusion.parameters():
            param.requires_grad = True
        for param in self.control_module.parameters():
            param.requires_grad = True

        train_lora = stage == "stage2"
        for name, param in self.vae.named_parameters():
            param.requires_grad = train_lora and "lora" in name
        for name, param in self.unet.named_parameters():
            param.requires_grad = train_lora and "lora" in name
        if self.learnable_control_alpha is not None:
            self.learnable_control_alpha.requires_grad = True

    def get_control_alpha(self, global_step: int = 0, disable_control: bool = False) -> float:
        if disable_control:
            return 0.0
        if self.learnable_control_alpha is not None:
            return float(torch.clamp(self.learnable_control_alpha.detach(), 0.0, 10.0).item())
        ratio = min(1.0, float(global_step) / float(self.control_alpha_warmup_steps))
        return self.control_alpha_init + (self.control_alpha_final - self.control_alpha_init) * ratio

    def encode_prompt(self, prompts: Sequence[str]) -> torch.Tensor:
        embeddings = []
        with torch.no_grad():
            for prompt in prompts:
                text_input_ids = self.tokenizer(
                    prompt,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                prompt_embeds = self.text_encoder(text_input_ids.to(self.device))[0]
                embeddings.append(prompt_embeds)
        return torch.cat(embeddings, dim=0)

    def load_pretrained_osediff_lora(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            print(f"[EventGuidedOSEDiff] Skip missing OSEDiff LoRA checkpoint: {checkpoint_path}")
            return
        checkpoint = _torch_load(checkpoint_path)
        if "state_dict_unet" in checkpoint:
            report = _copy_matching_named_parameters(self.unet, checkpoint["state_dict_unet"], keywords=("lora", "conv_in"))
            print(f"[EventGuidedOSEDiff] UNet LoRA/conv_in loaded={report['loaded']} skipped={report['skipped']}")
        if "state_dict_vae" in checkpoint:
            report = _copy_matching_named_parameters(self.vae, checkpoint["state_dict_vae"], keywords=("lora",))
            print(f"[EventGuidedOSEDiff] VAE LoRA loaded={report['loaded']} skipped={report['skipped']}")

    def load_event_guided_checkpoint(self, checkpoint_path: str, load_lora: bool = True) -> Dict:
        checkpoint = _torch_load(checkpoint_path)
        self.lle_encoder.load_state_dict(checkpoint["state_dict_lle_encoder"], strict=False)
        self.pyramid_fusion.load_state_dict(checkpoint["state_dict_pyramid_fusion"], strict=False)
        self.control_module.load_state_dict(checkpoint["state_dict_control_module"], strict=False)
        if load_lora:
            if "state_dict_unet" in checkpoint:
                _copy_matching_named_parameters(self.unet, checkpoint["state_dict_unet"], keywords=("lora", "conv_in"))
            if "state_dict_vae" in checkpoint:
                _copy_matching_named_parameters(self.vae, checkpoint["state_dict_vae"], keywords=("lora",))
        if self.learnable_control_alpha is not None and "learnable_control_alpha" in checkpoint:
            self.learnable_control_alpha.data.copy_(checkpoint["learnable_control_alpha"])
        return checkpoint

    def checkpoint_dict(self, args=None, global_step: int = 0, epoch: int = 0) -> Dict:
        return {
            "state_dict_lle_encoder": self.lle_encoder.state_dict(),
            "state_dict_pyramid_fusion": self.pyramid_fusion.state_dict(),
            "state_dict_control_module": self.control_module.state_dict(),
            "state_dict_unet": {key: value for key, value in self.unet.state_dict().items() if "lora" in key or "conv_in" in key},
            "state_dict_vae": {key: value for key, value in self.vae.state_dict().items() if "lora" in key},
            "learnable_control_alpha": None if self.learnable_control_alpha is None else self.learnable_control_alpha.detach().cpu(),
            "vae_lora_encoder_modules": self.lora_vae_modules_encoder,
            "unet_lora_encoder_modules": self.lora_unet_modules_encoder,
            "unet_lora_decoder_modules": self.lora_unet_modules_decoder,
            "unet_lora_others_modules": self.lora_unet_others,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "global_step": global_step,
            "epoch": epoch,
            "config": vars(args) if args is not None else {},
        }

    def build_optimizer_param_groups(self, args, stage: Optional[str] = None) -> List[Dict]:
        stage = stage or self.stage
        groups = [
            {"params": [p for p in self.lle_encoder.parameters() if p.requires_grad], "lr": args.lr_lle_encoder, "name": "lle_encoder"},
            {"params": [p for p in self.pyramid_fusion.parameters() if p.requires_grad], "lr": args.lr_pyramid_fusion, "name": "pyramid_fusion"},
            {"params": [p for p in self.control_module.parameters() if p.requires_grad], "lr": args.lr_control_adapter, "name": "control_adapter"},
        ]
        if self.learnable_control_alpha is not None and self.learnable_control_alpha.requires_grad:
            groups.append({"params": [self.learnable_control_alpha], "lr": args.lr_control_adapter, "name": "control_alpha"})
        if stage == "stage2":
            groups.extend([
                {"params": [p for n, p in self.vae.named_parameters() if p.requires_grad and "lora" in n], "lr": args.lr_vae_lora, "name": "vae_lora"},
                {"params": [p for n, p in self.unet.named_parameters() if p.requires_grad and "lora" in n], "lr": args.lr_unet_lora, "name": "unet_lora"},
            ])
        return [group for group in groups if len(group["params"]) > 0]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        train_timestep: Optional[str] = None,
        infer_timestep: Optional[int] = None,
        global_step: int = 0,
        disable_control: bool = False,
        event_mode: str = "normal",
        fixed_prompt: str = "high quality, bright, detailed photo",
        neg_prompt: str = "",
    ) -> Dict[str, torch.Tensor]:
        low = batch["conditioning_pixel_values"].to(self.device)
        low_rgb_01 = batch.get("low_rgb_01", (low + 1.0) * 0.5).to(self.device)
        event_voxel = batch["event_values"].to(self.device)

        if event_mode == "zero":
            event_voxel = torch.zeros_like(event_voxel)
        elif event_mode == "shuffle" and event_voxel.shape[0] > 1:
            event_voxel = event_voxel[torch.roll(torch.arange(event_voxel.shape[0], device=event_voxel.device), shifts=1)]
        elif event_mode != "normal":
            raise ValueError(f"Unknown event_mode: {event_mode}")

        prompts = batch.get("prompt", None)
        if prompts is None or prompts == "":
            prompts = [fixed_prompt] * low.shape[0]
        if isinstance(prompts, str):
            prompts = [prompts] * low.shape[0]
        prompts = [prompt if prompt else fixed_prompt for prompt in prompts]
        prompt_embeds = self.encode_prompt(prompts).to(dtype=torch.float32)

        if infer_timestep is not None:
            timesteps = torch.full((low.shape[0],), int(infer_timestep), device=self.device, dtype=torch.long)
        else:
            timesteps = sample_train_timesteps(train_timestep or "500", low.shape[0], self.device)

        # VAE/LoRA 在当前训练脚本中保持自身参数 dtype；不要把输入强制成 fp16，
        # 否则 Windows dry-run 会出现 input half / bias float 的 conv2d mismatch。
        vae_dtype = next(self.vae.parameters()).dtype
        encoded_control = self.vae.encode(low.to(dtype=vae_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        hdr_features = self.lle_encoder(low_rgb_01.to(dtype=torch.float32), event_voxel.to(dtype=torch.float32))
        control_condition = self.pyramid_fusion(hdr_features, target_size=encoded_control.shape[2:]).to(dtype=encoded_control.dtype)

        control_alpha = self.get_control_alpha(global_step=global_step, disable_control=disable_control)
        if disable_control or control_alpha == 0.0:
            down_residuals = None
            mid_residual = None
        else:
            down_residuals, mid_residual = self.control_module(
                encoded_control,
                timesteps,
                prompt_embeds.to(device=self.device, dtype=encoded_control.dtype),
                control_condition,
                conditioning_scale=control_alpha,
            )

        model_pred = self.unet(
            encoded_control,
            timesteps,
            encoder_hidden_states=prompt_embeds.to(device=self.device, dtype=encoded_control.dtype),
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
        ).sample
        x_denoised = self.noise_scheduler.step(model_pred, timesteps, encoded_control, return_dict=True).prev_sample
        output_image = self.vae.decode(x_denoised.to(dtype=vae_dtype) / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        residual_norm = torch.tensor(0.0, device=self.device)
        if down_residuals is not None:
            residual_norm = torch.stack([res.float().norm() for res in down_residuals]).mean()
        return {
            "output": output_image,
            "latents": x_denoised,
            "prompt_embeds": prompt_embeds,
            "control_alpha": torch.tensor(control_alpha, device=self.device),
            "control_residual_norm": residual_norm,
        }
