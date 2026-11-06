"""独立验证正则到事件输入的梯度、AMP 点积溢出和数值检查。"""
import unittest
import tempfile
from datetime import timedelta
from pathlib import Path
import torch
from torch import nn

from models.event_film_lora import (
    EVENT_LORA_VARIANTS, EventLoRAConfig, SharedResidualEventHeads,
    SpatialRGBEventLoRALinear, WindowRGBEventCrossAttentionCore,
)
from models.event_lora_utils import collect_event_lora_regularization
from models.training_finite_checks import require_finite


class LocalAccelerator:
    device = torch.device("cpu")
    process_index = 0

    def reduce(self, value, reduction):
        return value


def distributed_failure_worker(rank, rendezvous):
    # 真正的双进程归约：只有 rank 1 含 NaN，但两边都必须收到异常。
    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank,
                                         world_size=2, timeout=timedelta(seconds=30))
    try:
        accelerator = LocalAccelerator()
        accelerator.process_index = rank
        def reduce(value, reduction):
            torch.distributed.all_reduce(value)
            return value
        accelerator.reduce = reduce
        caught = False
        try:
            require_finite(accelerator, [("input", torch.tensor(float("nan") if rank else 1.))],
                           phase="distributed_test", step=1)
        except FloatingPointError:
            caught = True
        assert caught, "Every rank must fail before the next backward"
    finally:
        torch.distributed.destroy_process_group()


class StabilityTests(unittest.TestCase):
    def test_two_rank_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            rendezvous = (Path(folder) / "store").as_uri()
            torch.multiprocessing.spawn(distributed_failure_worker, args=(rendezvous,), nprocs=2)

    def test_both_regularizers_reach_events_for_every_variant(self):
        # 事件输入不属于参数 L2 路径：其非零梯度能证明增量正则没有 detach。
        for variant in EVENT_LORA_VARIANTS:
            for term in (0, 1):
                with self.subTest(variant=variant, term=term):
                    torch.manual_seed(42)
                    cfg = EventLoRAConfig(variant=variant, rank=2, mod_channels=8,
                                          zero_init_last=False)
                    heads = SharedResidualEventHeads(cfg, {"4": 3})
                    layer = SpatialRGBEventLoRALinear(nn.Linear(6, 6), cfg, {"4": 3}, heads)
                    nn.init.normal_(layer.lora_up.weight)
                    layer.lora_down.requires_grad_(False)
                    layer.lora_up.requires_grad_(False)
                    events = torch.randn(1, 3, 4, 4, requires_grad=True)
                    layer.set_event_feats({"4": events})
                    layer(torch.randn(1, 16, 6))
                    layer.clear_event_feats()  # 模拟生成器前向结束的特征清理。
                    losses = collect_event_lora_regularization(nn.Sequential(layer), "cpu")
                    losses[term].backward()
                    self.assertTrue(torch.isfinite(events.grad).all())
                    self.assertGreater(events.grad.abs().sum().item(), 0)
                    self.assertIsNone(layer.base_linear.weight.grad)
                    self.assertIsNone(layer.lora_down.weight.grad)
                    self.assertIsNone(layer.lora_up.weight.grad)
                    layer.clear_regularization_cache()
                    self.assertIsNone(layer.last_z_delta)
                    self.assertEqual(layer.last_aux, {})

    def test_large_finite_qk_under_autocast(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        core = WindowRGBEventCrossAttentionCore(2, 3, dim=8, num_heads=2, window_size=4).to(device)
        with torch.no_grad():
            for projection in (core.q_proj, core.k_proj):
                projection.weight.zero_()
                projection.bias.fill_(300)
        query = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        events = torch.randn(1, 3, 4, 4, device=device, requires_grad=True)
        with torch.autocast(device_type=device, dtype=torch.float16):
            # 旧写法即使输入为 float，点积也会溢出。
            probe = torch.full((4, 4), 300., device=device)
            self.assertFalse(torch.isfinite(probe.float() @ probe.float()).all())
            output = core(query, events)
        self.assertTrue(torch.isfinite(output).all())
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(events.grad).all())

    def test_nonfinite_checks_and_remote_failure(self):
        accelerator = LocalAccelerator()
        require_finite(accelerator, [("ok", torch.ones(2))], phase="test", step=1)
        for bad in (float("nan"), float("inf")):
            with self.assertRaisesRegex(FloatingPointError, "local_bad=.*bad"):
                require_finite(accelerator, [("bad", torch.tensor(bad))], phase="test", step=2)
        accelerator.reduce = lambda value, reduction: torch.ones_like(value)
        with self.assertRaisesRegex(FloatingPointError, "another rank"):
            require_finite(accelerator, [("ok", torch.ones(2))], phase="test", step=3)


if __name__ == "__main__":
    unittest.main()
