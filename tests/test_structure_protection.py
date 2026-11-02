"""回归实际发现的 L1 零更新问题、硬预算含义以及候选诊断的无副作用性。"""
from dataclasses import replace
import json
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from models.event_structure_guidance import GuidanceConfig, StructureObservation, _losses, optimize_features
from models.image_structure import StructureOperatorConfig, image_structure


def example():
    torch.manual_seed(7)
    feature = torch.rand(1, 3, 16, 16) * 1.5 - .75
    observation = StructureObservation(image_structure((feature.roll(2, -1) + 1) / 2),
                                       torch.full((1, 2, 16, 16), .6))
    def render(correction):
        return feature if correction is None else correction.apply(feature)
    return feature, observation, render


def test_soft_penalty_does_not_change_l1_budget_units_and_legacy_formula():
    baseline = torch.zeros(1, 3, 16, 16)
    # 显示域变化是 0.03：旧 L1 惩罚为 0.03，新二次惩罚为 0.0009。
    raw = baseline + .06
    observation = StructureObservation(torch.zeros(1, 2, 16, 16), torch.full((1, 2, 16, 16), .5))
    quadratic = GuidanceConfig()
    legacy = replace(quadratic, protection_mode="l1", lambda_low=2, lambda_keep=1)
    new_total, new = _losses(raw, baseline, observation, StructureOperatorConfig(), quadratic)
    old_total, old = _losses(raw, baseline, observation, StructureOperatorConfig(), legacy)
    for key in ("low", "keep"):
        assert float(new[key]) == pytest.approx(.03, abs=1e-6)
        torch.testing.assert_close(new[key], old[key])
    assert float(new["penalty_low"]) == pytest.approx(.0009, abs=1e-7)
    assert float(new["penalty_keep"]) == pytest.approx(.0009, abs=1e-7)
    assert float(new_total) == pytest.approx(.00135, abs=1e-6)
    assert float(old_total) == pytest.approx(.09, abs=1e-6)


def test_quadratic_allows_structure_update_that_original_l1_rejects():
    feature, observation, render = example()
    legacy = GuidanceConfig(steps=1, protection_mode="l1", lambda_low=2, lambda_keep=1)
    old_image, old_stats = optimize_features(render, observation, legacy)
    image, stats = optimize_features(render, observation, GuidanceConfig(steps=1))
    assert old_stats["accepted_steps"] == 0
    torch.testing.assert_close(old_image, feature, rtol=0, atol=0)
    assert all("objective_not_lower" in c["rejection_reasons"] for c in old_stats["candidate_history"])
    assert stats["accepted_steps"] == 1
    assert stats["final_losses"]["structure"] < stats["initial_losses"]["structure"]
    assert not torch.equal(image, feature)
    assert stats["candidate_history"][-1]["accepted"]
    assert stats["candidate_history"][-1]["losses"]["low"] <= .02
    assert stats["candidate_history"][-1]["losses"]["keep"] <= .02
    json.dumps(stats, allow_nan=False)


def test_quadratic_still_rejects_hard_budget_violation():
    feature, observation, render = example()
    image, stats = optimize_features(render, observation,
        GuidanceConfig(steps=1, max_low_change=0, max_outside_change=0))
    assert stats["accepted_steps"] == 0
    torch.testing.assert_close(image, feature, rtol=0, atol=0)
    assert all("low_budget" in c["rejection_reasons"] and "keep_budget" in c["rejection_reasons"]
               for c in stats["candidate_history"])


def test_candidate_trace_does_not_change_optimizer_output():
    _, observation, render = example()
    config = GuidanceConfig(steps=2)
    traced, traced_stats = optimize_features(render, observation, config)
    plain, plain_stats = optimize_features(render, observation, replace(config, trace_candidates=False))
    torch.testing.assert_close(traced, plain, rtol=0, atol=0)
    for key in ("accepted_steps", "forward_calls", "backward_calls", "final_objective"):
        assert traced_stats[key] == plain_stats[key]
    assert "candidate_history" not in plain_stats
    assert len(traced_stats["candidate_history"]) == traced_stats["forward_calls"] - 1


def test_nonfinite_candidate_trace_is_valid_json():
    feature, observation, _ = example()
    calls = 0
    def render(correction):
        nonlocal calls
        calls += 1
        image = correction.apply(feature)
        return image if calls == 1 else image * float("nan")
    image, stats = optimize_features(render, observation, GuidanceConfig(steps=1))
    torch.testing.assert_close(image, feature, rtol=0, atol=0)
    assert stats["accepted_steps"] == 0
    assert all(c["rejection_reasons"] == ["nonfinite_candidate"] for c in stats["candidate_history"])
    json.dumps(stats, allow_nan=False)
