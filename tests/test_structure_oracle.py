"""目标替换诊断：只改变目标、显式固定三种权重，并保留原默认优化结果。"""
from dataclasses import replace

import pytest
import torch

from models.event_structure_guidance import (GuidanceConfig, StructureObservation, _losses,
                                            freeze_observation_masks, optimize_features)
from models.image_structure import StructureOperatorConfig, image_structure


def test_explicit_original_masks_reproduce_default_optimizer():
    torch.manual_seed(19)
    feature = torch.rand(1, 3, 16, 16) - .5
    observation = StructureObservation(image_structure((feature.roll(2, -1) + 1) / 2),
                                       torch.full((1, 2, 16, 16), .6))
    def render(correction):
        return feature if correction is None else correction.apply(feature)
    implicit, stats0 = optimize_features(render, observation, GuidanceConfig(steps=1))
    explicit, stats1 = optimize_features(render, freeze_observation_masks(observation), GuidanceConfig(steps=1))
    torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)
    assert stats0 == stats1


def test_reweighting_structure_does_not_change_fixed_masks_or_protection():
    confidence = torch.full((1, 2, 12, 12), .4)
    frozen = freeze_observation_masks(StructureObservation(torch.zeros_like(confidence), confidence))
    changed = replace(frozen, confidence=torch.ones_like(confidence))
    baseline = torch.zeros(1, 3, 12, 12)
    raw = baseline + .02
    _, old = _losses(raw, baseline, frozen, StructureOperatorConfig(), GuidanceConfig())
    _, new = _losses(raw, baseline, changed, StructureOperatorConfig(), GuidanceConfig())
    for name in ('keep', 'penalty_keep', 'low', 'penalty_low'):
        torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
    torch.testing.assert_close(changed.feature_mask, confidence.mean(1, keepdim=True))
    torch.testing.assert_close(changed.protection_mask, 1 - confidence.mean(1, keepdim=True))


def test_oracle_uses_same_operator_and_preserves_event_masks():
    from test_idea8_structure_dps import diagnostic_observation
    torch.manual_seed(3)
    gt = torch.rand(1, 3, 16, 16) * 2 - 1
    confidence = torch.rand(1, 2, 16, 16)
    observed = StructureObservation(torch.zeros_like(confidence), confidence)
    event = diagnostic_observation(observed, 'event')
    oracle = diagnostic_observation(observed, 'gt_oracle', gt, StructureOperatorConfig())
    torch.testing.assert_close(oracle.edges, image_structure((gt + 1) / 2), rtol=0, atol=0)
    assert not oracle.edges.requires_grad
    for name in ('confidence', 'feature_mask', 'protection_mask'):
        torch.testing.assert_close(getattr(event, name), getattr(oracle, name), rtol=0, atol=0)
    with pytest.raises(ValueError, match='ground-truth'):
        diagnostic_observation(observed, 'gt_oracle')


def test_oracle_cli_requires_explicit_offline_positive_guidance():
    from test_idea8_structure_dps import parse_args
    common = ['--input_image', 'data', '--output_dir', 'out', '--pretrained_model_name_or_path', 'sd',
              '--event_lora_checkpoint', 'generator.pkl']
    assert parse_args(common).guidance_target_source == 'event'
    for extra in ([], ['--inference_only']):
        with pytest.raises(SystemExit):
            parse_args(common + ['--guidance_target_source', 'gt_oracle'] + extra)
    args = parse_args(common + ['--guidance_target_source', 'gt_oracle', '--guidance_steps', '1',
                                '--structure_checkpoint', 'head.pt'])
    assert args.require_ground_truth


def test_invalid_independent_mask_is_rejected():
    observation = StructureObservation(torch.zeros(1, 2, 4, 4), torch.ones(1, 2, 4, 4),
                                       feature_mask=torch.ones(1, 2, 4, 4))
    with pytest.raises(ValueError, match='Independent guidance masks'):
        observation.validate()
