"""idea8 CPU 回归：实际小型 UNet、事件时间锚点、训练恢复和四指标入口。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
from dataloaders.sde_event_structure_dataset import StructureEventConfig, SDEEventStructureDataset, structure_voxel
from models.event_structure_head import EventStructurePredictor, StructureHeadConfig, load_structure_checkpoint, structure_checkpoint
from models.event_structure_guidance import FullFeatureCorrection, GuidanceConfig, StructureObservation, optimize_features, restore_sample
from models.image_structure import StructureOperatorConfig, image_structure


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def geometry(size=16):
    return dict(scale=1.0, top=0, left=0, crop_h=size, crop_w=size, flip=False, image_size=size)


def test_fixed_time_bins_and_polarity_survive_spatial_crop():
    t = 10**15
    raw = torch.tensor([[t-40, 3, 3, 1], [t-40, 3, 3, -1], [t+40, 14, 14, 1]], dtype=torch.int64)
    config = StructureEventConfig("us", 50, 50)
    full = structure_voxel(raw, t, geometry(), config)
    crop_geometry = dict(geometry(8), top=1, left=1)
    cropped = structure_voxel(raw, t, crop_geometry, config)
    assert full[0, 3, 3] > 0 and full[5, 3, 3] > 0
    torch.testing.assert_close(cropped[:, 2, 2], full[:, 3, 3])
    assert cropped[0, 2, 2] > 0  # 裁剪移除末事件仍不重定义时间 bin。
    flipped = structure_voxel(raw, t, dict(geometry(), flip=True), config)
    torch.testing.assert_close(flipped, full.flip(-1))
    with pytest.raises(ValueError, match="raw"):
        structure_voxel(torch.zeros(5, 16, 16), t, geometry(), config)
    with pytest.raises(ValueError, match="timestamps"):
        structure_voxel(raw.float(), t, geometry(), config)


def make_dataset(root, scenes=("a", "b", "c"), frames=2):
    for scene in scenes:
        for folder in ("low", "normal"):
            (root / scene / folder).mkdir(parents=True)
        for index in range(frames):
            t = 10**15 + index * 1000
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[:, 16:] = 50
            Image.fromarray(image).save(root / scene / "low" / f"{t}.png")
            Image.fromarray(image * 3).save(root / scene / "normal" / f"{t+1}.png")
            events = np.array([[t-40, 16, 12, 1], [t-10, 16, 15, -1],
                               [t+10, 16, 18, 1], [t+40, 16, 21, -1]], dtype=np.int64)
            np.savez(root / scene / "low" / f"{t}.npz", events)


def test_extra_view_preserves_old_inputs_and_no_gt_loading(tmp_path):
    make_dataset(tmp_path, frames=1)
    args = argparse.Namespace(image_size=16)
    original = SDEEventFilmDataset("train", args, str(tmp_path))
    extended = SDEEventStructureDataset("train", args, StructureEventConfig("us", 50, 50), str(tmp_path))
    random.seed(9)
    a = original[0]
    random.seed(9)
    b = extended[0]
    for key in ("events", "events_refinement", "conditioning_pixel_values", "output_pixel_values"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert b["events_structure"].shape == (10, 16, 16)
    no_gt = SDEEventStructureDataset("test", argparse.Namespace(image_size=16, require_ground_truth=False),
                                     StructureEventConfig("us", 50, 50), str(tmp_path))
    before = no_gt[0]
    Path(original.samples[0].normal_path).write_bytes(b"not an image")
    after = no_gt[0]
    assert "output_pixel_values" not in after
    torch.testing.assert_close(before["conditioning_pixel_values"], after["conditioning_pixel_values"])
    torch.testing.assert_close(before["events_structure"], after["events_structure"])


def test_head_phases_checkpoint_and_operator_gradients(tmp_path):
    torch.manual_seed(4)
    head = EventStructurePredictor(StructureHeadConfig(base_channels=4))
    events = torch.rand(1, 10, 32, 32)
    head.set_training_phase("structure")
    edge, confidence = head(events)
    edge.mean().backward()
    assert head.edge.weight.grad is not None and head.confidence[0].weight.grad is None
    head.zero_grad(set_to_none=True)
    head.set_training_phase("confidence")
    _, confidence = head(events)
    confidence.mean().backward()
    assert head.edge.weight.grad is None and head.confidence[0].weight.grad is not None
    path = tmp_path / "head.pt"
    saved = structure_checkpoint(head, StructureOperatorConfig(), StructureEventConfig("us", 50, 50), confidence_trained=False)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="confidence"):
        load_structure_checkpoint(path)
    saved["confidence_trained"] = True
    torch.save(saved, path)
    restored, _, _ = load_structure_checkpoint(path)
    torch.testing.assert_close(restored(events)[0], head(events)[0])
    image = torch.rand(1, 3, 16, 16, requires_grad=True)
    gradient, = torch.autograd.grad(image_structure(image).sum(), image)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("targets", [("v",), ("q", "k", "v", "out")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_real_unet_zero_delta_gradient_and_frozen_parameters(targets, dtype):
    from models.unet_2d_condition import UNet2DConditionModel
    from models.event_film_lora import EventLoRAConfig
    from models.event_lora_utils import (replace_self_attn_projections_with_event_lora,
                                        set_event_feats_for_all_event_lora, iter_event_lora_modules)
    torch.manual_seed(8)
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), block_out_channels=(8, 16),
        layers_per_block=1, norm_num_groups=4, cross_attention_dim=12, attention_head_dim=2)
    replace_self_attn_projections_with_event_lora(unet,
        EventLoRAConfig(targets=targets, variant="residual_cross_attn", rank=4, mod_channels=8,
                        mod_num_heads=2, mod_window_size=4), {"8": 3, "4": 3})
    for _, layer in iter_event_lora_modules(unet):
        nn.init.normal_(layer.lora_up.weight, std=.1)  # 确保事件 LoRA 确实参与前向。
    unet.to(dtype=dtype).eval().requires_grad_(False)
    set_event_feats_for_all_event_lora(unet, {str(s): torch.randn(1, 3, s, s, dtype=dtype) for s in (8, 4)})
    state = {key: value.clone() for key, value in unet.state_dict().items()}
    latent, text = torch.randn(1, 4, 8, 8, dtype=dtype), torch.randn(1, 3, 12, dtype=dtype)
    expected = unet(latent, 500, text).sample
    hook = FullFeatureCorrection(torch.ones(1, 1, 64, 64))
    actual = unet(latent, 500, text, feature_guidance=hook).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert hook.delta.shape[1] > 4
    gradient, = torch.autograd.grad(actual.square().mean(), hook.delta)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    hook.delta = (hook.delta.detach() - gradient * 10).requires_grad_(True)
    changed = unet(latent, 500, text, feature_guidance=hook).sample
    assert not torch.equal(changed, expected)
    for key, value in unet.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert all(param.grad is None for param in unet.parameters())


def test_generator_bridge_actual_scheduler_vae_rng_and_state_restoration(monkeypatch):
    from diffusers import DDPMScheduler
    from models.autoencoder_kl import AutoencoderKL
    from models.unet_2d_condition import UNet2DConditionModel
    from osediff_event_film import DenoisingContext, OSEDiffEventFiLMGen
    from models.event_film_lora import EventLoRAConfig
    from models.event_lora_utils import replace_self_attn_projections_with_event_lora, iter_event_lora_modules

    torch.manual_seed(32)
    gen = object.__new__(OSEDiffEventFiLMGen)
    nn.Module.__init__(gen)
    gen.unet = UNet2DConditionModel(sample_size=32, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), block_out_channels=(4, 8),
        layers_per_block=1, norm_num_groups=1, cross_attention_dim=4, attention_head_dim=1)
    replace_self_attn_projections_with_event_lora(gen.unet,
        EventLoRAConfig(targets=("q", "v"), variant="residual_cross_attn", rank=4,
                        mod_channels=8, mod_num_heads=2, mod_window_size=4), {"32": 3, "16": 3})
    for _, layer in iter_event_lora_modules(gen.unet):
        nn.init.normal_(layer.lora_up.weight, std=.1)
    gen.vae = AutoencoderKL(block_out_channels=(4, 4, 4, 4),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4, layers_per_block=1,
        norm_num_groups=1, sample_size=256)
    gen.decoder_refinement_enabled = gen.decoder_fusion_enabled = False
    gen.refinement_fusion_mode, gen.decoder_fusion_weight = "both", 1.0
    gen.last_refinement_stats = {}
    gen.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    gen.noise_scheduler.set_timesteps(1)
    context = DenoisingContext(torch.randn(1, 4, 32, 32), torch.randn(1, 2, 4), torch.randn(1, 2, 4),
                              {str(s): torch.randn(1, 3, s, s) for s in (32, 16)})
    rgb, timestep = torch.zeros(1, 3, 256, 256), torch.tensor([500])
    gen.eval()
    rng = torch.get_rng_state()
    baseline, _ = restore_sample(gen, context, timestep, rgb, GuidanceConfig())
    observation = StructureObservation(torch.ones(1, 2, 256, 256), torch.zeros(1, 2, 256, 256))
    torch.set_rng_state(rng)
    fallback, stats = restore_sample(gen, context, timestep, rgb, GuidanceConfig(steps=1), observation)
    torch.testing.assert_close(fallback, baseline, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state()) and stats["forward_calls"] == 1

    # 非统一 training/requires_grad 状态与 checkpointing 开关均应逐项恢复。
    gen.train()
    gen.vae.eval()
    gen.unet.enable_gradient_checkpointing()
    next(gen.parameters()).requires_grad_(False)
    flags = [(m, m.training) for m in gen.modules()]
    requires = [(p, p.requires_grad, p.detach().clone()) for p in gen.parameters()]
    ckpt = [(m, m.gradient_checkpointing) for m in gen.modules() if hasattr(m, "gradient_checkpointing")]
    observation = StructureObservation(observation.edges, torch.ones_like(observation.confidence))
    output, stats = restore_sample(gen, context, timestep, rgb,
                                  GuidanceConfig(steps=1, backtracking_steps=1), observation)
    assert torch.isfinite(output).all() and stats["backward_calls"] == 1
    assert all(m.training == mode for m, mode in flags)
    assert all(m.gradient_checkpointing == enabled for m, enabled in ckpt)
    for param, enabled, value in requires:
        assert param.requires_grad == enabled and param.grad is None
        torch.testing.assert_close(param, value, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(layer.event_feats is None and not layer.last_aux for _, layer in iter_event_lora_modules(gen.unet))

    def fail(*args, **kwargs):
        raise RuntimeError("decode failed")
    monkeypatch.setattr(gen, "decode_denoised_latent", fail)
    with pytest.raises(RuntimeError, match="decode failed"):
        restore_sample(gen, context, timestep, rgb, GuidanceConfig(steps=1), observation)
    assert all(m.training == mode for m, mode in flags)
    assert all(p.requires_grad == enabled for p, enabled, _ in requires)
    assert all(m.gradient_checkpointing == enabled for m, enabled in ckpt)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(layer.event_feats is None and not layer.last_aux for _, layer in iter_event_lora_modules(gen.unet))


def test_optimizer_improves_structure_with_actual_gradient_and_trust_budget():
    h = torch.linspace(-0.2, 0.2, 32)[None, None, None].expand(1, 8, 32, 32).clone()
    target = h[:, :3].clone()
    target[:, :, :, 16:] += 0.2
    observation = StructureObservation(image_structure(target * .5 + .5).requires_grad_(True),
                                       torch.ones(1, 2, 32, 32, requires_grad=True))
    calls = []
    def render(hook):
        calls.append(hook)
        return (hook.apply(h) if hook else h)[:, :3]
    config = GuidanceConfig(steps=2, learning_rate=.05, trust_radius=.1, lambda_low=0,
                            lambda_keep=0, max_low_change=1, max_outside_change=1)
    output, stats = optimize_features(render, observation, config)
    assert stats["accepted_steps"] > 0 and stats["final_objective"] < stats["initial_objective"]
    assert stats["relative_delta"] <= config.trust_radius + 1e-6
    assert not torch.equal(output, h[:, :3])
    assert len(calls) == stats["forward_calls"]
    assert calls[0].delta is None  # finally 已清理临时状态。
    assert observation.edges.grad is None and observation.confidence.grad is None


def test_no_support_disabled_and_nonfinite_candidate_fallback():
    baseline = torch.rand(1, 3, 16, 16) * .2
    target = StructureObservation(image_structure(baseline*.5+.5), torch.zeros(1, 2, 16, 16))
    for config, observation in ((GuidanceConfig(), None), (GuidanceConfig(steps=1), target)):
        output, stats = optimize_features(lambda hook: baseline, observation, config)
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        assert stats["forward_calls"] == 1 and stats["backward_calls"] == 0
    calls = []
    def bad_render(hook):
        result = hook.apply(baseline)
        calls.append(1)
        return result if len(calls) == 1 else result * float("nan")
    target = StructureObservation(torch.ones(1, 2, 16, 16), torch.ones(1, 2, 16, 16))
    output, stats = optimize_features(bad_render, target, GuidanceConfig(steps=1))
    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    assert stats["accepted_steps"] == 0


def test_training_two_phases_and_epoch_resume(tmp_path):
    from train_event_structure_head import main, parse_args
    data = tmp_path / "data"
    make_dataset(data, frames=1)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(dict(train=["a"], calibration=["b"], validation=["c"])))
    common = ["--data_root", str(data), "--scene_split", str(split), "--base_channels", "4",
              "--batch_size", "1", "--device", "cpu"]
    output = tmp_path / "structure"
    main(parse_args(common + ["--output_dir", str(output), "--epochs", "1", "--structure_time_unit", "us",
                             "--structure_window_before", "50", "--structure_window_after", "50"]))
    main(parse_args(common + ["--output_dir", str(output), "--epochs", "2",
                             "--resume_from_checkpoint", str(output / "checkpoint_last.pt")]))
    checkpoint = torch.load(output / "checkpoint_last.pt", weights_only=False)
    assert checkpoint["epoch"] == 1 and not checkpoint["confidence_trained"]
    # 相同 seed 连续跑两轮，应与一轮后保存/恢复的结果一致。
    uninterrupted = tmp_path / "uninterrupted"
    main(parse_args(common + ["--output_dir", str(uninterrupted), "--epochs", "2", "--structure_time_unit", "us",
                             "--structure_window_before", "50", "--structure_window_after", "50"]))
    continuous = torch.load(uninterrupted / "checkpoint_last.pt", weights_only=False)
    for key in checkpoint["state_dict"]:
        torch.testing.assert_close(checkpoint["state_dict"][key], continuous["state_dict"][key], rtol=0, atol=0)
    calibrated = tmp_path / "confidence"
    main(parse_args(common + ["--output_dir", str(calibrated), "--epochs", "1", "--phase", "confidence",
                             "--init_structure_checkpoint", str(output / "checkpoint_last.pt")]))
    _, confidence_checkpoint, _ = load_structure_checkpoint(calibrated / "checkpoint_last.pt")
    assert confidence_checkpoint["confidence_trained"]
    for key in checkpoint["state_dict"]:
        if not key.startswith("confidence."):
            torch.testing.assert_close(checkpoint["state_dict"][key], confidence_checkpoint["state_dict"][key], rtol=0, atol=0)


def test_default_four_metrics_complete_evaluation_without_downloads(tmp_path, monkeypatch):
    # 本测试检查真实入口/CSV/FID 文件收集；指标网络用明确的测试替身，避免下载权重。
    pyiqa = types.ModuleType("pyiqa")
    monkeypatch.setitem(sys.modules, "pyiqa", pyiqa)
    import osediff_event_film
    from test_idea8_structure_dps import main, parse_args
    data = tmp_path / "data"
    make_dataset(data, scenes=("a",), frames=2)
    pretrained = tmp_path / "pretrained"
    pretrained.mkdir()
    checkpoint = tmp_path / "generator.pkl"
    checkpoint.touch()
    output = tmp_path / "result"
    metric_calls = []
    fid_paths = []
    class Metric(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
        def forward(self, prediction, target):
            if self.name == "fid":
                assert len(list(Path(prediction).glob("*.png"))) == 2
                assert len(list(Path(target).glob("*.png"))) == 2
                fid_paths.append(Path(prediction))
            else:
                assert prediction.min() >= 0 and prediction.max() <= 1
            return torch.tensor(1.0)
    def create(name, **kwargs):
        metric_calls.append((name, kwargs))
        return Metric(name)
    class Gen(nn.Module):
        def __init__(self):
            super().__init__()
            self.event_lora_config = types.SimpleNamespace(targets=("v",))
            self.noise_scheduler = types.SimpleNamespace(config={})
        def prepare_denoising_context(self, rgb, events, batch):
            return rgb
        def predict_denoised_latent_from_context(self, context, timesteps):
            return context, {}
        def decode_denoised_latent(self, latent, rgb, **kwargs):
            return latent
    def model(args):
        return types.SimpleNamespace(gen=Gen(), device=torch.device("cpu"), weight_dtype=torch.float32, timesteps=torch.tensor([500]))
    monkeypatch.setattr(pyiqa, "create_metric", create, raising=False)
    # 兼容整个 tests/ 一起执行时，其他测试已导入共用指标模块的情况。
    import test_event_film_lora_osediff
    monkeypatch.setattr(test_event_film_lora_osediff, "pyiqa", pyiqa)
    monkeypatch.setattr(osediff_event_film, "OSEDiffEventFiLMTest", model)
    monkeypatch.setattr(osediff_event_film, "resolve_decoder_refinement_args", lambda *args: None)
    args = parse_args(["--input_image", str(data), "--output_dir", str(output),
                       "--event_lora_checkpoint", str(checkpoint), "--pretrained_model_name_or_path", str(pretrained)])
    assert args.calculate_fid and args.guidance_steps == 0
    main(args)
    assert [call[0] for call in metric_calls] == ["psnr", "ssim", "lpips", "fid"]
    assert metric_calls[0][1]["test_y_channel"] and metric_calls[1][1]["test_y_channel"]
    assert (output / "idea8_summary.csv").read_text().splitlines()[0] == "mode,count,psnr,ssim,lpips,fid"
    assert not fid_paths[0].exists()  # finally 清理 FID 临时图。
