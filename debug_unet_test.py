import torch
import sys
import argparse
sys.path.insert(0, '.')

from osediff import OSEDiff_test

args = argparse.Namespace(
    pretrained_model_name_or_path='D:/PretrainedModel/stable-diffusion-2-1-base',
    odediff_path='checkpoints/osediff_lle_5-2-0020/checkpoints/model_final.pkl',
    osediff_path='checkpoints/osediff_lle_5-2-0020/checkpoints/model_final.pkl',
    image_size=256,
    device='cuda',
    merge_and_unload_lora=False,
    vae_encoder_tiled_size=1024,
    vae_decoder_tiled_size=224,
    latent_tiled_size=96,
    latent_tiled_overlap=32,
    use_roformer=False,
    use_clipseg=False,
    mixed_precision='fp16',
    enable_rotate=False,
)

print("=== Loading model WITH LoRA ===")
model_lora = OSEDiff_test(args)

# Build lq_latent similar to real data: mean~0.035, std~0.19
torch.manual_seed(42)
lq_latent = torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16) * 0.19 + 0.035
t = torch.tensor([999], device='cuda')
prompt_embeds = torch.randn(1, 77, 1024, device='cuda', dtype=torch.float16)

print("\n=== Test 1: UNet 直接推理 (随机输入, std≈1) ===")
with torch.no_grad():
    x_rand = torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16)
    out1 = model_lora.unet(x_rand, t, encoder_hidden_states=prompt_embeds).sample
    print(f"UNet out  mean={out1.mean().item():.6f}, std={out1.std().item():.6f}")
    print(f"x_rand    mean={x_rand.mean().item():.6f}, std={x_rand.std().item():.6f}")
    print(f"|out-x|: {torch.abs(out1 - x_rand).mean().item():.6f}")

print("\n=== Test 2: UNet 推理 (lq_latent 分布, std≈0.19) ===")
with torch.no_grad():
    out2 = model_lora.unet(lq_latent, t, encoder_hidden_states=prompt_embeds).sample
    print(f"UNet out       mean={out2.mean().item():.6f}, std={out2.std().item():.6f}")
    print(f"lq_latent      mean={lq_latent.mean().item():.6f}, std={lq_latent.std().item():.6f}")
    print(f"|out-lq|: {torch.abs(out2 - lq_latent).mean().item():.6f}")

print("\n=== Test 3: 不带 LoRA (merge_and_unload_lora=True) ===")
args_no_lora = argparse.Namespace(
    pretrained_model_name_or_path='D:/PretrainedModel/stable-diffusion-2-1-base',
    odediff_path='checkpoints/osediff_lle_5-2-0020/checkpoints/model_final.pkl',
    osediff_path='checkpoints/osediff_lle_5-2-0020/checkpoints/model_final.pkl',
    image_size=256,
    device='cuda',
    merge_and_unload_lora=True,
    vae_encoder_tiled_size=1024,
    vae_decoder_tiled_size=224,
    latent_tiled_size=96,
    latent_tiled_overlap=32,
    use_roformer=False,
    use_clipseg=False,
    mixed_precision='fp16',
    enable_rotate=False,
)
model_merge = OSEDiff_test(args_no_lora)
with torch.no_grad():
    out3 = model_merge.unet(lq_latent, t, encoder_hidden_states=prompt_embeds).sample
    print(f"UNet (merged)  mean={out3.mean().item():.6f}, std={out3.std().item():.6f}")
    print(f"|out3 - lq|: {torch.abs(out3 - lq_latent).mean().item():.6f}")
    print(f"|out2 - out3| (LoRA effect): {torch.abs(out2 - out3).mean().item():.6f}")

print("\n=== Test 4: 对比 prompt_embeds vs zeros (prompt 有无效果) ===")
with torch.no_grad():
    zero_prompt = torch.zeros(1, 77, 1024, device='cuda', dtype=torch.float16)
    out_p = model_lora.unet(lq_latent, t, encoder_hidden_states=prompt_embeds).sample
    out_z = model_lora.unet(lq_latent, t, encoder_hidden_states=zero_prompt).sample
    print(f"with prompt  mean={out_p.mean().item():.6f}, std={out_p.std().item():.6f}")
    print(f"with zeros   mean={out_z.mean().item():.6f}, std={out_z.std().item():.6f}")
    print(f"|diff|: {torch.abs(out_p - out_z).mean().item():.6f}")

print("\n=== Test 5: 对比不同 timestep (t=500 vs t=999) ===")
with torch.no_grad():
    t500 = torch.tensor([500], device='cuda')
    out_t999 = model_lora.unet(lq_latent, t, encoder_hidden_states=prompt_embeds).sample
    out_t500 = model_lora.unet(lq_latent, t500, encoder_hidden_states=prompt_embeds).sample
    print(f"t=999  mean={out_t999.mean().item():.6f}, std={out_t999.std().item():.6f}")
    print(f"t=500  mean={out_t500.mean().item():.6f}, std={out_t500.std().item():.6f}")
    print(f"|diff|: {torch.abs(out_t999 - out_t500).mean().item():.6f}")

print("\n=== Test 6: LoRA weight magnitude详情 ===")
lora_norms = {}
for n, p in model_lora.unet.named_parameters():
    if "lora" in n:
        prefix = ".".join(n.split(".")[:4])
        if prefix not in lora_norms:
            lora_norms[prefix] = []
        lora_norms[prefix].append(p.abs().mean().item())

print("LoRA各模块的mean abs:")
for k, vals in sorted(lora_norms.items()):
    print(f"  {k}: mean_abs={sum(vals)/len(vals):.8f}, count={len(vals)}")

print("\nDone.")
