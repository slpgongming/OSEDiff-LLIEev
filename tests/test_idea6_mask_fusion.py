"""CPU 张量测试：验证真实事件头、源权重迁移、梯度和 checkpoint 条件重算。"""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from models.event_film_lora import EventLoRAConfig, SharedResidualEventHeads, SpatialRGBEventLoRALinear
from models.rgb_event_mask_lora import (
    MaskFusionConfig, fusion_context, install_mask_fusion, iter_mask_layers,
    load_mask_state, mask_state_dict,
)
from osediff_mask_fusion import OSEDiffMaskFusion, conditioned_unet, compact_sources, load_rgb_subset
from train_idea6_mask_fusion import (EpochSeededDataset, EpochShuffleSampler, event_edge_loss, parse_args,
                                   resolve_resume, register_checkpoint_hooks, restore_rank_rng, prepare_epoch_loader)


class TinyRGBLoRA(nn.Module):
    """保留 base + 2*BA 的真实线性映射，模拟原 RGB adapter 的缩放。"""
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(4, 4)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)

    def forward(self, x):
        return self.base_layer(x) + 2 * self.lora_B(self.lora_A(x))


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(block_out_channels=(4,))
        self.down_blocks = nn.ModuleList([nn.Module()])
        self.down_blocks[0].attn1 = nn.Module()
        self.down_blocks[0].attn1.to_v = TinyRGBLoRA()
        self.mid_block = nn.Module()
        self.mid_block.attn1 = nn.Module()
        self.mid_block.attn1.to_v = TinyRGBLoRA()
        self.forward_count = 0

    def forward(self, x, timestep=None, encoder_hidden_states=None):
        self.forward_count += 1
        x = self.down_blocks[0].attn1.to_v(x).tanh()
        return SimpleNamespace(sample=self.mid_block.attn1.to_v(x))


def fixture():
    torch.manual_seed(71)
    unet = TinyUNet().eval().requires_grad_(False)
    baseline = copy.deepcopy(unet)
    event_config = EventLoRAConfig(variant="residual_cross_attn", rank=2, alpha=2,
                                  mod_channels=8, mod_num_heads=2, mod_window_size=2,
                                  modulation_scale=0.3, zero_init_last=False)
    heads = SharedResidualEventHeads(event_config, {"4": 3})
    state = {"_event_lora_shared_heads_residual_cross_attn." + k: v.clone()
             for k, v in heads.state_dict().items()}
    for name in ("down_blocks.0.attn1.to_v", "mid_block.attn1.to_v"):
        state[f"{name}.lora_down.weight"] = torch.randn(2, 4)
        state[f"{name}.lora_up.weight"] = torch.randn(4, 2)
    saved = {"event_lora_config": event_config.to_dict(), "state_dict_unet": state}
    install_mask_fusion(unet, saved, {"4": 3}, MaskFusionConfig(mask_channels=3))
    x = torch.randn(2, 16, 4)
    # f(0) 刻意取非零值，验证代码确实减去编码后的零事件响应。
    features, zeros = {"4": torch.randn(2, 3, 4, 4)}, {"4": torch.randn(2, 3, 4, 4) + 1}
    return unet, baseline, saved, x, features, zeros


class MaskFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def run_unet(self, unet, x, features, zeros, **kw):
        return conditioned_unet(unet, x, torch.tensor([500]), None, features, zeros, **kw)

    def test_initial_output_is_exact_rgb_and_one_unet_call(self):
        unet, baseline, _, x, features, zeros = fixture()
        actual = self.run_unet(unet, x, features, zeros)
        self.assertTrue(torch.equal(actual, baseline(x).sample))
        self.assertEqual(unet.forward_count, 1)
        self.assertTrue(all(layer.context is None for _, layer in iter_mask_layers(unet)))

    def test_zero_events_identity_after_mask_and_strength_change(self):
        unet, baseline, _, x, features, zeros = fixture()
        with torch.no_grad():
            for _, layer in iter_mask_layers(unet):
                layer.raw_strength.fill_(0.7)
                layer.mask.fuse[-1].weight.normal_()
        self.assertTrue(torch.equal(self.run_unet(unet, x, zeros, zeros), baseline(x).sample))
        self.assertFalse(torch.allclose(self.run_unet(unet, x, features, zeros), baseline(x).sample))
        self.assertTrue(torch.equal(self.run_unet(unet, x, features, zeros, enabled=False), baseline(x).sample))

    def test_copied_event_delta_matches_legacy_response_difference(self):
        unet, _, saved, x, features, zeros = fixture()
        name, layer = next(iter_mask_layers(unet))
        source_config = EventLoRAConfig(**{**saved["event_lora_config"], "alpha": 8.0})
        shared = SharedResidualEventHeads(source_config, {"4": 3})
        prefix = "_event_lora_shared_heads_residual_cross_attn."
        shared.load_state_dict({k[len(prefix):]: v for k, v in saved["state_dict_unet"].items() if k.startswith(prefix)})
        legacy = SpatialRGBEventLoRALinear(copy.deepcopy(layer.rgb_layer.base_layer), source_config,
                                           {"4": 3}, shared)
        with torch.no_grad():
            legacy.lora_down.weight.copy_(layer.event_down.weight)
            legacy.lora_up.weight.copy_(layer.event_up.weight)
            layer.raw_strength.fill_(0.4)
        legacy.set_event_feats(features)
        positive = legacy(x)
        legacy.set_event_feats(zeros)
        negative = legacy(x)
        with fusion_context(unet, features, zeros):
            difference = layer(x) - layer.rgb_layer(x)
        torch.testing.assert_close(difference, layer.strength * 0.5 * (positive - negative), atol=3e-6, rtol=1e-4)

    def test_only_mask_strength_update_and_zero_init_can_start(self):
        unet, _, _, x, features, zeros = fixture()
        frozen = {n: p.clone() for n, p in unet.named_parameters() if not p.requires_grad}
        trainable = [(n, p) for n, p in unet.named_parameters() if p.requires_grad]
        self.assertTrue(all(".mask." in n or n.endswith("raw_strength") for n, _ in trainable))
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-2)
        self.run_unet(unet, x, features, zeros).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in trainable if n.endswith("raw_strength")))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.run_unet(unet, x, features, zeros).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in trainable if ".mask." in n))
        self.assertTrue(all(p.grad is None and torch.equal(p, frozen[n])
                            for n, p in unet.named_parameters() if n in frozen))

    def test_checkpoint_output_and_all_trainable_gradients_match(self):
        unet, _, _, x, features, zeros = fixture()
        for _, layer in iter_mask_layers(unet):
            layer.raw_strength.data.fill_(0.4)
        output = self.run_unet(unet, x, features, zeros)
        output.square().sum().backward()
        expected = {n: p.grad.clone() for n, p in unet.named_parameters() if p.requires_grad}
        unet.zero_grad(set_to_none=True)
        actual = self.run_unet(unet, x, features, zeros, use_checkpoint=True)
        self.assertTrue(all(layer.context is None for _, layer in iter_mask_layers(unet)))
        actual.square().sum().backward()
        torch.testing.assert_close(actual, output)
        for n, p in unet.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.grad, expected[n])
        self.assertTrue(all(layer.context is None for _, layer in iter_mask_layers(unet)))

    def test_checkpoint_handles_two_outstanding_forwards_with_different_events(self):
        unet, _, _, x, features, zeros = fixture()
        for _, layer in iter_mask_layers(unet):
            layer.raw_strength.data.fill_(0.4)
        changed = {"4": features["4"] * -2}
        (self.run_unet(unet, x, features, zeros) + self.run_unet(unet, x, changed, zeros)).square().sum().backward()
        expected = {n: p.grad.clone() for n, p in unet.named_parameters() if p.requires_grad}
        unet.zero_grad(set_to_none=True)
        a = self.run_unet(unet, x, features, zeros, use_checkpoint=True)
        b = self.run_unet(unet, x, changed, zeros, use_checkpoint=True)
        (a + b).square().sum().backward()
        for n, p in unet.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.grad, expected[n])

    def test_context_cleared_on_exception_and_missing_context_rejected(self):
        unet, _, _, x, features, zeros = fixture()
        with self.assertRaises(RuntimeError):
            unet(x)
        with self.assertRaises(ValueError):
            with fusion_context(unet, features, zeros):
                raise ValueError("synthetic failure")
        self.assertTrue(all(layer.context is None for _, layer in iter_mask_layers(unet)))

    def test_mask_checkpoint_round_trip_and_incomplete_weights_fail(self):
        unet, _, saved, x, features, zeros = fixture()
        for _, layer in iter_mask_layers(unet):
            layer.raw_strength.data.fill_(0.7)
        state = mask_state_dict(unet)
        other, _, _, _, _, _ = fixture()
        load_mask_state(other, state)
        torch.testing.assert_close(self.run_unet(unet, x, features, zeros), self.run_unet(other, x, features, zeros))
        broken = dict(state)
        broken.pop(next(iter(broken)))
        with self.assertRaises(ValueError):
            load_mask_state(other, broken)
        saved["state_dict_unet"].pop("down_blocks.0.attn1.to_v.lora_up.weight")
        with self.assertRaises(ValueError):
            install_mask_fusion(TinyUNet(), saved, {"4": 3}, MaskFusionConfig())

    def test_partial_rgb_checkpoint_fails_instead_of_silently_using_random_lora(self):
        module = TinyRGBLoRA()
        state = {k: v for k, v in module.state_dict().items() if "lora" in k}
        load_rgb_subset(module, state, lambda k: "lora" in k, "RGB")
        state.pop(next(iter(state)))
        with self.assertRaises(ValueError):
            load_rgb_subset(module, state, lambda k: "lora" in k, "RGB")

    def test_compact_event_source_does_not_include_rgb_or_regularizer(self):
        _, _, event, _, _, _ = fixture()
        event["state_dict_event_encoder"] = {"test": torch.ones(1)}
        event["state_dict_unet"]["conv_in.weight"] = torch.ones(4)
        event["state_dict_unet"]["bad.lora_A.default_encoder.weight"] = torch.ones(4)
        event["optimizer_reg"] = {"unused": torch.ones(3)}
        rgb = dict(rank_unet=2, rank_vae=2, state_dict_unet={}, state_dict_vae={})
        _, compact = compact_sources(rgb, event)
        self.assertNotIn("conv_in.weight", compact["state_dict_unet"])
        self.assertNotIn("bad.lora_A.default_encoder.weight", compact["state_dict_unet"])
        self.assertNotIn("optimizer_reg", compact)

    def test_edge_loss_has_gt_gradient_and_zero_event_behavior(self):
        pred = torch.randn(1, 3, 8, 8, requires_grad=True)
        gt, events = torch.randn_like(pred), torch.ones(1, 5, 8, 8)
        event_edge_loss(pred, gt, events).backward()
        self.assertGreater(pred.grad.abs().sum().item(), 0)
        self.assertEqual(event_edge_loss(pred, gt, events * 0).item(), 0)

    def test_training_defaults_and_latest_ignore_incomplete_checkpoints(self):
        args = parse_args(["--pretrained_model_name_or_path", "sd", "--train_data_root", "train",
                           "--rgb_checkpoint", "6.pkl", "--event_checkpoint", "90.pkl", "--output_dir", "out"])
        self.assertEqual((args.train_batch_size, args.gradient_accumulation_steps, args.generator_timestep), (1, 4, 500))
        self.assertEqual(args.clora_event_active_bins, 5)
        with tempfile.TemporaryDirectory() as tmp:
            for step in (10, 20, 30):
                path = Path(tmp) / f"checkpoint-{step}"
                path.mkdir()
                if step != 30:
                    (path / "progress.json").write_text("{}")
            args.output_dir, args.resume_from_checkpoint = tmp, "latest"
            self.assertEqual(resolve_resume(args).name, "checkpoint-20")

    def test_epoch_sampler_is_reproducible(self):
        data = list(range(20))
        a, b = EpochShuffleSampler(data, 42), EpochShuffleSampler(data, 42)
        self.assertEqual(list(a), list(b))
        a.set_epoch(1)
        self.assertNotEqual(list(a), list(b))
        b.set_epoch(1)
        self.assertEqual(list(a), list(b))

    @unittest.skipUnless(importlib.util.find_spec("peft") and importlib.util.find_spec("diffusers"),
                         "optional real PEFT/diffusers integration dependencies unavailable")
    def test_actual_peft_layer_and_tiny_diffusers_unet_forward_backward(self):
        from peft import LoraConfig
        from models.unet_2d_condition import UNet2DConditionModel
        unet = UNet2DConditionModel(sample_size=4, in_channels=4, out_channels=4,
                   down_block_types=("CrossAttnDownBlock2D",), up_block_types=("CrossAttnUpBlock2D",),
                   block_out_channels=(8,), layers_per_block=1, norm_num_groups=4,
                   cross_attention_dim=8, attention_head_dim=2)
        unet.add_adapter(LoraConfig(r=2, target_modules=["to_v", "to_q"]), adapter_name="rgb")
        unet.set_adapter("rgb")
        for name, param in unet.named_parameters():
            if "lora_B" in name:
                param.data.normal_(0, 0.1)
        unet.eval().requires_grad_(False)
        baseline = copy.deepcopy(unet)
        config = EventLoRAConfig(variant="residual_cross_attn", rank=2, mod_channels=8,
                                 mod_num_heads=2, mod_window_size=2, zero_init_last=False)
        heads = SharedResidualEventHeads(config, {"4": 3})
        state = {"_event_lora_shared_heads_residual_cross_attn."+k: v for k, v in heads.state_dict().items()}
        for name, layer in unet.named_modules():
            if name.endswith("attn1.to_v"):
                self.assertEqual(layer.lora_alpha["rgb"], 8)
                state[name+".lora_down.weight"] = torch.randn(2, layer.in_features)
                state[name+".lora_up.weight"] = torch.randn(layer.out_features, 2)
        install_mask_fusion(unet, {"event_lora_config": config.to_dict(), "state_dict_unet": state},
                            {"4": 3}, MaskFusionConfig(mask_channels=3))
        x, prompt = torch.randn(1, 4, 4, 4), torch.randn(1, 3, 8)
        features, zeros = {"4": torch.randn(1, 3, 4, 4)}, {"4": torch.ones(1, 3, 4, 4)}
        actual = conditioned_unet(unet, x, 500, prompt, features, zeros)
        torch.testing.assert_close(actual, baseline(x, 500, encoder_hidden_states=prompt).sample, rtol=0, atol=0)
        for _, layer in iter_mask_layers(unet):
            layer.raw_strength.data.fill_(0.5)
        actual = conditioned_unet(unet, x, 500, prompt, features, zeros, use_checkpoint=True)
        actual.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in unet.named_parameters() if ".mask." in n))
        self.assertTrue(all(p.grad is None for p in unet.parameters() if not p.requires_grad))

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "optional Accelerate unavailable")
    def test_accelerate_checkpoint_restores_mask_optimizer_and_rng(self):
        from accelerate import Accelerator
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet, _, _, self.x, self.features, self.zeros = fixture()
            def forward(self):
                return conditioned_unet(self.unet, self.x, 500, None, self.features, self.zeros)
            def export_checkpoint(self):
                return {"mask_state_dict": mask_state_dict(self.unet)}
        acc = Accelerator(cpu=True)
        model = Model()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        model, optimizer = acc.prepare(model, optimizer)
        register_checkpoint_hooks(acc)
        acc.backward(model().square().mean())
        optimizer.step()
        optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as tmp:
            acc.save_state(tmp)
            self.assertFalse((Path(tmp) / "model.safetensors").exists())
            expected_random = torch.rand(4)
            expected_state = copy.deepcopy(mask_state_dict(model.unet))
            expected_optim = copy.deepcopy(optimizer.state_dict())
            for p in model.parameters():
                if p.requires_grad:
                    p.data.add_(1)
            acc.load_state(tmp)
            restore_rank_rng(acc, tmp)
            torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
            for k, v in mask_state_dict(model.unet).items():
                torch.testing.assert_close(v, expected_state[k])
            for key, state in optimizer.state_dict()["state"].items():
                torch.testing.assert_close(state["exp_avg"], expected_optim["state"][key]["exp_avg"])
        acc.free_memory()

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "optional Accelerate unavailable")
    def test_resume_mid_later_epoch_keeps_remaining_sample_order(self):
        from accelerate import Accelerator
        acc = Accelerator(cpu=True, rng_types=[])
        dataset = EpochSeededDataset(list(range(24)), seed=91)
        sampler = EpochShuffleSampler(dataset, seed=91)
        loader = acc.prepare(torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler))
        full = [batch.tolist() for batch in prepare_epoch_loader(acc, loader, dataset, sampler, 3, 0)]
        resumed = [batch.tolist() for batch in prepare_epoch_loader(acc, loader, dataset, sampler, 3, 4)]
        self.assertEqual(resumed, full[4:])
        acc.free_memory()

    @unittest.skipUnless(importlib.util.find_spec("peft") and importlib.util.find_spec("diffusers"),
                         "optional real generator integration dependencies unavailable")
    def test_full_generator_tiny_local_sd_initialization_gradient_and_roundtrip(self):
        # 真实 UNet/VAE/PEFT/调度器和 EventEncoder；仅替换文本 tokenizer/encoder 下载。
        from models.unet_2d_condition import UNet2DConditionModel
        from models.autoencoder_kl import AutoencoderKL
        from models.event_encoder import EventEncoder
        from osediff import initialize_unet, initialize_vae
        from diffusers import DDPMScheduler
        from transformers import CLIPTextConfig, CLIPTextModel
        class Tokenizer:
            model_max_length = 8
            def __call__(self, *args, **kwargs):
                return SimpleNamespace(input_ids=torch.zeros(1, 8, dtype=torch.long))
        clip = CLIPTextModel(CLIPTextConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                          num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8))
        with tempfile.TemporaryDirectory() as tmp:
            unet = UNet2DConditionModel(sample_size=32, in_channels=4, out_channels=4,
                       down_block_types=("CrossAttnDownBlock2D",), up_block_types=("CrossAttnUpBlock2D",),
                       block_out_channels=(32,), layers_per_block=1, norm_num_groups=32,
                       cross_attention_dim=8, attention_head_dim=2)
            vae = AutoencoderKL(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                     down_block_types=("DownEncoderBlock2D",)*4, up_block_types=("UpDecoderBlock2D",)*4,
                     layers_per_block=1, sample_size=256)
            unet.save_pretrained(Path(tmp) / "unet")
            vae.save_pretrained(Path(tmp) / "vae")
            DDPMScheduler().save_pretrained(Path(tmp) / "scheduler")
            args = SimpleNamespace(pretrained_model_name_or_path=tmp, lora_rank=2)
            rgb_unet, *_ = initialize_unet(args)
            rgb_vae, _ = initialize_vae(args)
            rgb = dict(rank_unet=2, rank_vae=2,
                       state_dict_unet={k: v for k, v in rgb_unet.state_dict().items() if "lora" in k or "conv_in" in k},
                       state_dict_vae={k: v for k, v in rgb_vae.state_dict().items() if "lora" in k})
            encoder = EventEncoder(base_channels=8)
            config = EventLoRAConfig(variant="residual_cross_attn", rank=2, mod_channels=8,
                                     mod_num_heads=2, mod_window_size=2, zero_init_last=False)
            heads = SharedResidualEventHeads(config, encoder.out_channels_by_size)
            event_state = {"_event_lora_shared_heads_residual_cross_attn."+k: v for k, v in heads.state_dict().items()}
            for name, layer in rgb_unet.named_modules():
                if name.endswith("attn1.to_v"):
                    event_state[name+".lora_down.weight"] = torch.randn(2, layer.in_features)
                    event_state[name+".lora_up.weight"] = torch.randn(layer.out_features, 2)
            event = dict(event_lora_config=config.to_dict(), state_dict_unet=event_state,
                         state_dict_event_encoder=encoder.state_dict())
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
                 patch("transformers.CLIPTextModel.from_pretrained", side_effect=lambda *a, **kw: copy.deepcopy(clip)):
                model = OSEDiffMaskFusion(tmp, rgb, event, MaskFusionConfig(mask_channels=3), device="cpu",
                                          frozen_dtype=torch.float32, gradient_checkpointing=True)
                self.assertGreater(model.assert_trainable_scope(), 0)
                low, events = torch.randn(1, 3, 256, 256).clamp(-1, 1), torch.randn(1, 5, 256, 256)
                latent = model.encode_rgb(low)
                initial = model(low, events, latent=latent)
                base = model(low, events, latent=latent, rgb_only=True)
                torch.testing.assert_close(initial, base, rtol=0, atol=0)
                for _, layer in iter_mask_layers(model.unet):
                    layer.raw_strength.data.fill_(0.3)
                output = model(low, events, latent=latent)
                output.square().mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in model.named_parameters() if ".mask." in n))
                self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
                model.eval()
                with torch.no_grad():
                    expected = model(low, events, latent=latent)
                    restored = OSEDiffMaskFusion.from_checkpoint(tmp, model.export_checkpoint(), device="cpu",
                                                                  frozen_dtype=torch.float32).eval()
                    torch.testing.assert_close(restored(low, events, latent=latent), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
