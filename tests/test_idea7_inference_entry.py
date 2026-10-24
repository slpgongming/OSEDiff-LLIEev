"""入口集成：真实数据读取/proxy/runner，轻量生成器代替需 CUDA 权重的 OSEDiff。"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from dataloaders.sde_event_pair_dataset import CONTRACT_KEYS
from models.event_observation_proxy import DEFAULT_PROXY_CONFIG
from test_idea7_dps import build_parser, frame_seed_indices, main, resolve_checkpoint_args, summarize_rows


@pytest.fixture
def entry_files(tmp_path):
    settings = dict(timestamp_unit="us", timestamp_anchor="center", clock_offset=0,
                    polarity_encoding="zero_one", spatial_mapping="identity", image_size=256,
                    event_bins=5, clora_event_active_bins=5, rgb_event_active_bins=5)
    source = dict(path="events.npz", coverage=[0, 40])
    condition = dict(sources=[source], boundary="all")
    np.savez(tmp_path / "events.npz", timestamp=np.array([15, 25], dtype=np.int64),
             x=[4, 4], y=[4, 4], polarity=[1, 1])
    pairs = []
    for index, stamp in enumerate([10, 20, 30]):
        Image.fromarray(np.full((8, 8, 3), 60 + index * 10, dtype=np.uint8)).save(tmp_path / f"{stamp}.png")
        previous = stamp - 10 if index else None
        pairs.append(dict(scene_id="s", prev_id=str(previous) if previous else None, curr_id=str(stamp),
                          low_prev=f"{previous}.png" if previous else None, low_curr=f"{stamp}.png",
                          t_prev=previous, t_curr=stamp, split="test", observation_group="A",
                          observation_conditions=dict(exposure="fixed", isp="fixed"),
                          condition_event_sources=dict(previous=condition if previous else None, current=condition),
                          measurement_event_sources=[source] if previous else []))
    manifest = dict(schema_version=1, settings=settings, pairs=pairs)
    proxy_config = copy.deepcopy(DEFAULT_PROXY_CONFIG)
    proxy_config.update(linearization="identity", min_valid_fraction=0., block_size=4)
    calibration = dict(schema_version=1, status="validated", proxy=proxy_config,
                       contract={key: settings[key] for key in CONTRACT_KEYS},
                       groups=dict(A=dict(status="validated", a=1., sigma=1., conditions=pairs[0]["observation_conditions"],
                                          dt_range=[10, 10], sensor_sizes=[[8, 8]],
                                          validated_supports=[dict(sensor_height=8, sensor_width=8, dt=10)],
                                          validation=dict(passed=True))))
    checkpoint = dict(event_lora_config=dict(variant="residual_cross_attn", rank=4, mod_channels=64),
                      train_unet_lora_active_parts=[], timestep_config=dict(generator_train_timestep=(500, 500)))
    (tmp_path / "pairs.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    torch.save(checkpoint, tmp_path / "model.pkl")
    (tmp_path / "pretrained").mkdir()
    argv = ["--data-root", str(tmp_path), "--pair-manifest", str(tmp_path / "pairs.json"),
            "--output-dir", str(tmp_path / "output"), "--event-lora-checkpoint", str(tmp_path / "model.pkl"),
            "--pretrained-model-name-or-path", str(tmp_path / "pretrained"), "--mixed-precision", "fp32"]
    return tmp_path, argv, manifest, calibration, checkpoint


class _Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = torch.nn.Module()
        self.unet.up_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        self.calls = 0
        self.cleanup_calls = 0

    def prepare_fixed_denoising_context(self, rgb, events, batch):
        return rgb.detach()

    def predict_denoised_latent_from_context(self, context, timesteps, feature_guidance=None):
        self.calls += 1
        feature = context * 0.1 + 1.
        if feature_guidance is not None:
            feature_guidance.bind_num_up_blocks(3)
        for index, block in enumerate(self.unet.up_blocks):
            feature = block(feature)
            if feature_guidance is not None:
                feature = feature_guidance.apply(index, feature)
        return ((feature - 1.) / 0.1).clamp(-1, 1), {}

    def decode_denoised_latent(self, latent, rgb, **kwargs):
        return latent

    def clear_feature_dps_caches(self):
        self.cleanup_calls += 1


class _Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gen = _Generator()
        self.device = torch.device("cpu")
        self.weight_dtype = torch.float32
        self.timesteps = torch.tensor([args.generator_timestep])


def _resolved_args(argv):
    args = build_parser().parse_args(argv)
    args.event_bins = args.clora_event_active_bins = args.rgb_event_active_bins = 5
    return args


def test_unique_timestep_and_checkpoint_activation_inherited(entry_files):
    _, argv, _, _, checkpoint = entry_files
    args = resolve_checkpoint_args(_resolved_args(argv), checkpoint)
    assert args.generator_timestep == 500
    assert args.unet_lora_parts == ()
    assert args.mod_channels == 64 and not args.decoder_refinement_enabled


def test_range_timestep_and_missing_activation_require_explicit_choice(entry_files):
    _, argv, _, _, checkpoint = entry_files
    checkpoint["timestep_config"]["generator_train_timestep"] = [400, 600]
    with pytest.raises(ValueError, match="generator-timestep"):
        resolve_checkpoint_args(_resolved_args(argv), checkpoint)
    del checkpoint["train_unet_lora_active_parts"]
    with pytest.raises(ValueError, match="activation"):
        resolve_checkpoint_args(_resolved_args(argv + ["--generator-timestep", "500"]), checkpoint)
    args = resolve_checkpoint_args(_resolved_args(argv + ["--generator-timestep", "500", "--unet-lora-parts", "none"]), checkpoint)
    assert args.unet_lora_parts == ()


def test_structure_active_bins_and_fusion_weight_checked(entry_files):
    _, argv, _, _, checkpoint = entry_files
    with pytest.raises(ValueError, match="conflicts"):
        resolve_checkpoint_args(_resolved_args(argv + ["--mod-channels", "32"]), checkpoint)
    checkpoint["additional_info"] = dict(event_active_bins=dict(clora=1, rgb_event_encoder=1, total_channels=5))
    with pytest.raises(ValueError, match="event_active_bins"):
        resolve_checkpoint_args(_resolved_args(argv), checkpoint)
    del checkpoint["additional_info"]
    checkpoint["state_dict_decoder_refinement"] = {}
    checkpoint["decoder_refinement_config"] = dict(variant="cgdrs", fusion_weight=0.25)
    args = resolve_checkpoint_args(_resolved_args(argv), checkpoint)
    assert args.decoder_refinement_enabled and args.decoder_fusion_weight == 0.25
    with pytest.raises(ValueError, match="fusion_weight"):
        resolve_checkpoint_args(_resolved_args(argv + ["--decoder-fusion-weight", "1"]), checkpoint)


def test_gt_free_entry_cache_counts_and_saved_baseline(entry_files):
    root, argv, _, _, _ = entry_files
    args = build_parser().parse_args(argv + ["--observation-calibration", str(root / "calibration.json")])
    models = []
    def factory(args):
        model = _Model(args)
        models.append(model)
        return model
    main(args, model_factory=factory)
    summary = json.loads((root / "output/summary.json").read_text())
    assert summary["count"] == 3 and summary["forward_calls_total"] == 5
    assert summary["backward_calls_total"] == 2 and summary["reference_forward_calls_total"] == 0
    assert summary["accepted"] == 2 and models[0].gen.calls == 5
    assert not (root / "normal").exists()
    last = json.loads((root / "output/last_sample.json").read_text())
    assert last["row"]["reference_cache_hit"] is True
    files = sorted((root / "output/images_baseline").glob("*.png"))
    assert len(files) == 3
    for path, expected in zip(files, [60, 70, 80]):
        assert np.array(Image.open(path)).mean() == expected


def test_k0_needs_no_calibration_or_measurement_files(entry_files):
    root, argv, manifest, _, _ = entry_files
    for pair in manifest["pairs"]:
        pair["measurement_event_sources"] = []
    (root / "pairs.json").write_text(json.dumps(manifest))
    main(build_parser().parse_args(argv + ["--dps-steps", "0"]), model_factory=_Model)
    summary = json.loads((root / "output/summary.json").read_text())
    assert summary["forward_calls_total"] == 3 and summary["backward_calls_total"] == 0
    assert summary["reference_forward_calls_total"] == 0


def test_shifted_feedback_records_actual_donor(entry_files):
    root, argv, _, _, _ = entry_files
    args = build_parser().parse_args(argv + ["--observation-calibration", str(root / "calibration.json"),
                                            "--feedback-mode", "shifted"])
    main(args, model_factory=_Model)
    last = json.loads((root / "output/last_sample.json").read_text())["row"]
    assert last["feedback_donor_id"] == "20" and last["backward_calls"] == 1


def test_shifted_feedback_excludes_declared_gap_donor(entry_files):
    root, argv, manifest, _, _ = entry_files
    manifest["pairs"][1]["skip_reason"] = "time_gap"
    (root / "pairs.json").write_text(json.dumps(manifest))
    args = build_parser().parse_args(argv + ["--observation-calibration", str(root / "calibration.json"),
                                            "--feedback-mode", "shifted"])
    main(args, model_factory=_Model)
    last = json.loads((root / "output/last_sample.json").read_text())["row"]
    assert last["skip_reason"] == "no_distinct_shift_donor" and last["backward_calls"] == 0


def test_preflight_does_not_load_checkpoint_or_model(entry_files):
    root, argv, _, _, _ = entry_files
    (root / "model.pkl").unlink()
    args = build_parser().parse_args(argv + ["--dps-steps", "0", "--preflight-only"])
    main(args, model_factory=lambda _: pytest.fail("Preflight must not construct the model"))
    assert json.loads((root / "output/preflight.json").read_text())["status"] == "passed"


def test_vae_then_scheduler_rng_matches_sequential_baseline(entry_files):
    root, argv, manifest, _, _ = entry_files
    class StochasticGenerator(_Generator):
        def prepare_fixed_denoising_context(self, rgb, events, batch):
            return rgb + 0.001 * torch.rand_like(rgb)

        def predict_denoised_latent_from_context(self, context, timesteps, feature_guidance=None):
            latent, info = super().predict_denoised_latent_from_context(context, timesteps, feature_guidance)
            return (latent + 0.002 * torch.rand_like(latent)).clamp(-1, 1), info

    class StochasticModel(_Model):
        def __init__(self, args):
            super().__init__(args)
            self.gen = StochasticGenerator()

    args = build_parser().parse_args(argv + ["--dps-steps", "0"])
    before = torch.get_rng_state().clone()
    main(args, model_factory=StochasticModel)
    assert torch.equal(before, torch.get_rng_state())
    # 独立按旧入口顺序播种→编码采样→scheduler随机项，比较落盘baseline。
    from dataloaders.sde_event_pair_dataset import SDEEventPairDataset
    from test_idea7_dps import _output_name
    dataset = SDEEventPairDataset(root / "pairs.json", root, require_measurement=False)
    expected_model = StochasticModel(args)
    for index in range(3):
        frame = dataset[index]["current"]
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.random.default_generator.manual_seed(args.seed + index)
            context = expected_model.gen.prepare_fixed_denoising_context(
                frame["conditioning_pixel_values"].unsqueeze(0), None, None)
            prediction, _ = expected_model.gen.predict_denoised_latent_from_context(context, expected_model.timesteps)
            expected = ((prediction * 0.5 + 0.5)[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        actual = np.array(Image.open(root / "output/images_baseline" / _output_name("s", frame["frame_id"])))
        np.testing.assert_array_equal(actual, expected)


def test_failed_calibration_rejected_before_model(entry_files):
    root, argv, _, calibration, _ = entry_files
    calibration["status"] = "failed"
    (root / "calibration.json").write_text(json.dumps(calibration))
    args = build_parser().parse_args(argv + ["--observation-calibration", str(root / "calibration.json")])
    with pytest.raises(ValueError, match="validated"):
        main(args, model_factory=lambda _: pytest.fail("Bad observation must fail before model loading"))


def test_summary_includes_skips_and_actual_costs():
    result = summarize_rows([dict(selected_iteration=0, skip_reason="first_frame", forward_calls=1),
                             dict(selected_iteration=1, skip_reason="", forward_calls=2, backward_calls=1)])
    assert result["acceptance_rate"] == 0.5 and result["forward_calls_total"] == 3


def test_explicit_seed_indices_survive_manifest_subset_and_reordering():
    pairs = [dict(scene_id="s", prev_id="a", curr_id="b", seed_indices=dict(previous=7, current=8)),
             dict(scene_id="s", prev_id="b", curr_id="c", seed_indices=dict(previous=8, current=9))]
    assert frame_seed_indices(pairs) == frame_seed_indices(list(reversed(pairs)))
    assert frame_seed_indices(pairs[1:])[("s", "b")] == 8
    pairs[1]["seed_indices"]["previous"] = 4
    with pytest.raises(ValueError, match="inconsistent"):
        frame_seed_indices(pairs)
