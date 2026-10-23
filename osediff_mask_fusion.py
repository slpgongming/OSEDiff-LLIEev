"""idea6 独立生成器：6 号 RGB 基座 + 冻结 90 号事件增量 + 可训练 mask。"""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.event_encoder import EventEncoder
from models.rgb_event_mask_lora import (
    MaskFusionConfig, fusion_context, install_mask_fusion, iter_mask_layers,
    load_exact, load_mask_state, mask_state_dict,
)


CHECKPOINT_VERSION = "idea6_mask_fusion_v1"


def load_checkpoint(path):
    """项目 checkpoint 含配置等非张量字段；只读取用户指定的可信训练权重。"""
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_sources(rgb, event):
    """新权重自带两套冻结 adapter；丢弃源 checkpoint 的 VSD/优化器大对象。"""
    rgb_keys = ("rank_unet", "rank_vae", "state_dict_unet", "state_dict_vae")
    rgb_small = {key: rgb[key] for key in rgb_keys}
    event_small = {key: event[key] for key in ("event_lora_config", "state_dict_event_encoder")}
    event_small["event_lora_layer_configs"] = event.get("event_lora_layer_configs", {})
    event_small["state_dict_unet"] = {
        k: v for k, v in event["state_dict_unet"].items()
        if k.startswith("_event_lora_shared_heads_residual_cross_attn.")
        or k.endswith((".lora_down.weight", ".lora_up.weight"))
    }
    return rgb_small, event_small


def load_rgb_subset(module, state, predicate, label):
    """完整校验 6 号所有 LoRA/conv_in；缺参数立即失败，避免假 RGB 基线。"""
    own = module.state_dict()
    expected = {k for k in own if predicate(k)}
    if set(state) != expected:
        raise ValueError(f"{label}: missing={sorted(expected-set(state))}, extra={sorted(set(state)-expected)}")
    mismatch = [k for k in state if state[k].shape != own[k].shape]
    if mismatch:
        raise ValueError(f"{label} shape mismatch: {mismatch}")
    module.load_state_dict(state, strict=False)


def conditioned_unet(unet, latent, timestep, prompt, features, zeros,
                     enabled=True, use_checkpoint=False, collect_stats=False):
    """重算闭包捕获当前 batch 条件；每次重算重新 set/clear，避免 idea4 的状态问题。"""
    def run(x):
        with fusion_context(unet, features, zeros, enabled, collect_stats):
            return unet(x, timestep, encoder_hidden_states=prompt).sample
    if use_checkpoint and torch.is_grad_enabled():
        # 输入 latent 无梯度也能训练内部 mask；不使用要求输入带梯度的 reentrant 版本。
        return checkpoint(run, latent, use_reentrant=False)
    return run(latent)


class OSEDiffMaskFusion(nn.Module):
    def __init__(self, pretrained_path, rgb_source, event_source, config=None,
                 fixed_prompt="high quality, bright, detailed photo", timestep=500,
                 device="cuda", frozen_dtype=torch.float16, gradient_checkpointing=False):
        super().__init__()
        # 延迟导入完整扩散依赖，使迁移/梯度 CPU 单元测试无需下载 SD 模型。
        from osediff import initialize_unet, initialize_vae
        from transformers import AutoTokenizer, CLIPTextModel
        from diffusers import DDPMScheduler

        if not Path(pretrained_path).is_dir():
            raise FileNotFoundError(pretrained_path)
        self.fusion_config = config or MaskFusionConfig()
        self.rgb_source, self.event_source = compact_sources(rgb_source, event_source)
        self.fixed_prompt, self.timestep = fixed_prompt, int(timestep)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.provenance = {}
        self.unet, *_ = initialize_unet(SimpleNamespace(
            pretrained_model_name_or_path=pretrained_path, lora_rank=int(rgb_source["rank_unet"])))
        self.vae, _ = initialize_vae(SimpleNamespace(
            pretrained_model_name_or_path=pretrained_path, lora_rank=int(rgb_source["rank_vae"])))
        load_rgb_subset(self.unet, self.rgb_source["state_dict_unet"],
                        lambda k: "lora" in k or "conv_in" in k, "6 RGB UNet")
        load_rgb_subset(self.vae, self.rgb_source["state_dict_vae"], lambda k: "lora" in k, "6 VAE")
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
        self.vae.set_adapter(["default_encoder"])
        enc_state = self.event_source["state_dict_event_encoder"]
        stem = enc_state["stem.block.0.weight"]
        if stem.shape[1] != 5:
            raise ValueError("idea6 pilot requires the 5-channel event encoder from experiment 90")
        self.event_encoder = EventEncoder(in_channels=5, base_channels=int(stem.shape[0]))
        load_exact(self.event_encoder, enc_state, "90 event encoder")
        self.replaced_names = install_mask_fusion(self.unet, self.event_source,
                                                  self.event_encoder.out_channels_by_size, self.fusion_config)
        self.requires_grad_(False)
        # 先将所有冻结权重转半精度，再单独保留 mask/强度的 FP32 主参数。
        self.to(device=device, dtype=frozen_dtype)
        for _, layer in iter_mask_layers(self.unet):
            layer.mask.float().requires_grad_(True)
            layer.raw_strength.data = layer.raw_strength.data.float()
            layer.raw_strength.requires_grad_(True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder").to(
            device=device, dtype=frozen_dtype).eval()
        with torch.no_grad():
            tokens = tokenizer(fixed_prompt, max_length=tokenizer.model_max_length,
                               padding="max_length", truncation=True, return_tensors="pt").input_ids
            prompt = text_encoder(tokens.to(device))[0]
        # 固定 prompt 仅编码一次；不常驻额外的文本模型，更不创建 VSD UNet。
        self.register_buffer("prompt_embeds", prompt, persistent=False)
        del text_encoder, tokenizer
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
        if not 0 <= self.timestep < self.scheduler.config.num_train_timesteps:
            raise ValueError("timestep outside scheduler training range")
        self.scheduler.set_timesteps(1, device=device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        self.train()
        self.assert_trainable_scope()

    def train(self, mode=True):
        super().train(mode)
        # 冻结路径使用确定性的 eval；VAE decode 的输入梯度不受 eval 影响。
        self.unet.eval()
        self.vae.eval()
        self.event_encoder.eval()
        for _, layer in iter_mask_layers(self.unet):
            layer.mask.train(mode)
        return self

    def assert_trainable_scope(self):
        expected = {f"unet.{k}" for k in mask_state_dict(self.unet)}
        actual = {n for n, p in self.named_parameters() if p.requires_grad}
        if actual != expected or not actual:
            raise RuntimeError(f"Trainable scope mismatch: extra={actual-expected}, missing={expected-actual}")
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def encode_rgb(self, rgb, generator=None):
        rgb = rgb.to(dtype=next(self.vae.parameters()).dtype)
        return self.vae.encode(rgb).latent_dist.sample(generator=generator) * self.vae.config.scaling_factor

    @torch.no_grad()
    def encode_events(self, events):
        # 两次轻量 EventEncoder；f(0) 含已训练的 bias/GN，不能当成零张量。
        events = events.to(dtype=next(self.event_encoder.parameters()).dtype)
        return self.event_encoder(events), self.event_encoder(torch.zeros_like(events))

    def forward(self, rgb, events, latent=None, rgb_only=False, collect_stats=False):
        if rgb.shape[-2:] != (256, 256) or events.shape != (rgb.shape[0], 5, 256, 256):
            raise ValueError("idea6 pilot requires aligned RGB [B,3,256,256] and events [B,5,256,256]")
        if latent is None:
            latent = self.encode_rgb(rgb)
        features, zeros = self.encode_events(events)
        prompt = self.prompt_embeds.expand(rgb.shape[0], -1, -1)
        t = torch.tensor([self.timestep], dtype=torch.long, device=latent.device)
        prediction = conditioned_unet(self.unet, latent, t, prompt, features, zeros,
                                       enabled=not rgb_only,
                                       use_checkpoint=self.gradient_checkpointing and self.training,
                                       collect_stats=collect_stats)
        denoised = self.scheduler.step(prediction, self.timestep, latent).prev_sample
        def decode(z):
            return self.vae.decode(z.to(next(self.vae.parameters()).dtype) / self.vae.config.scaling_factor).sample
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            image = checkpoint(decode, denoised, use_reentrant=False)
        else:
            image = decode(denoised)
        return image.clamp(-1, 1)

    def debug_stats(self):
        return {f"{name}/{k}": float(v.item()) for name, layer in iter_mask_layers(self.unet)
                for k, v in layer.last_stats.items()}

    def export_checkpoint(self):
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "fusion_config": self.fusion_config.to_dict(),
            "rgb_source": self.rgb_source, "event_source": self.event_source,
            "mask_state_dict": mask_state_dict(self.unet),
            "fixed_prompt": self.fixed_prompt, "timestep": self.timestep,
            "image_size": 256, "clora_event_active_bins": 5,
            "provenance": self.provenance,
        }

    @classmethod
    def from_checkpoint(cls, pretrained_path, saved, **kwargs):
        if saved.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("Not an idea6 mask checkpoint")
        model = cls(pretrained_path, saved["rgb_source"], saved["event_source"],
                    config=MaskFusionConfig(**saved["fusion_config"]),
                    fixed_prompt=saved["fixed_prompt"], timestep=saved["timestep"], **kwargs)
        load_mask_state(model.unet, saved["mask_state_dict"])
        model.provenance = saved.get("provenance", {})
        return model
