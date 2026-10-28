"""全分辨率路径的空间尺寸、梯度、冻结与旧权重兼容回归。"""
from dataclasses import asdict

import pytest
import torch

from models.event_structure_head import (EventStructurePredictor, StructureHeadConfig,
                                        load_structure_checkpoint, structure_checkpoint)
from models.image_structure import StructureOperatorConfig
from dataloaders.sde_event_structure_dataset import StructureEventConfig


def test_legacy_checkpoint_and_shared_initialization(tmp_path):
    torch.manual_seed(42)
    old = EventStructurePredictor(StructureHeadConfig(base_channels=4))
    torch.manual_seed(42)
    new = EventStructurePredictor(StructureHeadConfig(base_channels=4, variant="full_detail"))
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    payload = structure_checkpoint(old, StructureOperatorConfig(), StructureEventConfig(), confidence_trained=True)
    # 模拟旧 v1 权重：缺少 variant 时必须恢复原 half 结构，输出逐位一致。
    payload["head_config"].pop("variant")
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    restored, _, _ = load_structure_checkpoint(path)
    events = torch.rand(1, 10, 32, 32)
    assert restored.config.variant == "half"
    for actual, expected in zip(restored(events), old(events)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_detail_resolution_gradients_and_confidence_freeze(tmp_path):
    head = EventStructurePredictor(StructureHeadConfig(base_channels=4, variant="full_detail"))
    seen = []
    hook = head.edge.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[-2:]))
    events = torch.rand(2, 10, 32, 40)
    head.set_training_phase("structure")
    edge, confidence = head(events)
    assert edge.shape == confidence.shape == (2, 2, 32, 40)
    assert seen == [torch.Size([32, 40])]
    edge.square().mean().backward()
    for branch in (head.detail, head.detail_fuse, head.enc3):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        assert any(p.grad.abs().sum() > 0 for p in branch.parameters())
    before = {k: v.clone() for k, v in head.state_dict().items()}
    head.zero_grad(set_to_none=True)
    head.set_training_phase("confidence")
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=1e-3)
    head(events)[1].square().mean().backward()
    optimizer.step()
    for key, value in before.items():
        if not key.startswith("confidence."):
            assert torch.equal(value, head.state_dict()[key])
    assert any(not torch.equal(v, head.state_dict()[k]) for k, v in before.items() if k.startswith("confidence."))
    payload = structure_checkpoint(head, StructureOperatorConfig(), StructureEventConfig(), confidence_trained=True)
    path = tmp_path / "full.pt"
    torch.save(payload, path)
    restored, _, _ = load_structure_checkpoint(path)
    assert restored.config.variant == "full_detail"
    for actual, expected in zip(restored(events), head(events)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    hook.remove()


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        StructureHeadConfig(variant="unknown")
