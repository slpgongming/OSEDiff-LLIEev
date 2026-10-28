"""真实双进程 CPU/Gloo 回归，不以模拟梯度同步替代 DDP；无需 GPU 或模型下载。"""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, root, store_path, loss_mode, head_variant):
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      USE_LIBUV="0")
    torch.set_num_threads(1)
    import train_event_structure_head as training

    # 在各 rank 完成每个 epoch 时逐参数比较实际模型，确保两个进程没有各自独立训练。
    last_model = []
    original_loss, original_capture = training.phase_loss, training.capture_rng

    def checked_loss(model, *args, **kwargs):
        last_model[:] = [model.module if hasattr(model, "module") else model]
        return original_loss(model, *args, **kwargs)

    def checked_capture(device):
        state = {key: value.detach().cpu() for key, value in last_model[0].state_dict().items()}
        states = [None, None]
        dist.all_gather_object(states, state)
        for key in state:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
        return original_capture(device)

    training.phase_loss, training.capture_rng = checked_loss, checked_capture
    root = Path(root)
    common = ["--data_root", str(root / "data"), "--scene_split", str(root / "split.json"),
              "--base_channels", "4", "--batch_size", "2", "--device", "cpu", "--num_workers", "1",
              "--structure_loss", loss_mode, "--head_variant", head_variant]
    # Windows TCPStore 的本地主机名解析不稳定；只将 rendezvous 换成临时 FileStore，
    # 梯度/指标仍经真实 Gloo 通信。测试复用进程组，避免无外部同步地销毁/重建进程组。
    store = dist.FileStore(store_path, 2)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2)

    def run(name, epochs, extra=()):
        args = training.parse_args(common + ["--output_dir", str(root / name), "--epochs", str(epochs)] + list(extra))
        with (root / f"stdout_rank{rank}.log").open("a", encoding="utf-8") as handle, redirect_stdout(handle):
            training.train_main(args, rank, 2, torch.device("cpu"))

    run("structure", 1)
    run("structure", 2, ["--resume_from_checkpoint", str(root / "structure/checkpoint_last.pt"), "--log_interval", "1"])
    run("continuous", 2)
    init = ["--phase", "confidence", "--init_structure_checkpoint", str(root / "structure/checkpoint_last.pt")]
    run("confidence", 1, init)
    run("confidence", 2, ["--phase", "confidence", "--resume_from_checkpoint", str(root / "confidence/checkpoint_last.pt"),
                          "--log_interval", "1"])
    run("confidence_continuous", 2, init)

    # 只在 rank 1 产生非有限 loss，两个 rank 都必须在 backward 前退出，不能互相等待。
    def bad_loss(model, *args, **kwargs):
        loss, edge, confidence = original_loss(model, *args, **kwargs)
        return loss * (float("nan") if rank else 1), edge, confidence
    training.phase_loss = bad_loss
    try:
        run("nonfinite", 1)
    except FloatingPointError:
        (root / f"nonfinite_rank{rank}.txt").write_text("caught", encoding="utf-8")
    else:
        raise AssertionError("Non-finite loss was not rejected on all ranks")
    dist.destroy_process_group()
    assert not dist.is_initialized()


def _dataset(root):
    # train/calibration 的奇数样本检查 sampler 补齐；validation 只有 1 张，rank 1 为空。
    for scene, count in (("a", 3), ("b", 3), ("c", 1)):
        for folder in ("low", "normal"):
            (root / "data" / scene / folder).mkdir(parents=True)
        for index in range(count):
            tick = 10**15 + index * 1000
            image = np.zeros((264, 264, 3), dtype=np.uint8)
            image[:, 130:] = 30 + index * 10
            Image.fromarray(image).save(root / "data" / scene / "low" / f"{tick}.png")
            Image.fromarray(image * 3).save(root / "data" / scene / "normal" / f"{tick+1}.png")
            event = np.array([[tick-40,130,20,1],[tick-10,130,100,-1],[tick+10,130,150,1],[tick+40,130,220,-1]],dtype=np.int64)
            np.savez(root / "data" / scene / "low" / f"{tick}.npz", event)
    (root / "split.json").write_text(json.dumps(dict(train=["a"],calibration=["b"],validation=["c"])),encoding="utf-8")


@pytest.mark.parametrize("head_variant", ["half", "full_detail"])
@pytest.mark.parametrize("loss_mode", ["smooth_l1", "region_balanced"])
def test_two_process_training_resume_and_validation(tmp_path, monkeypatch, loss_mode, head_variant):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo is unavailable")
    _dataset(tmp_path)
    # 当前 Windows PyTorch FileStore 不支持中文路径；只把临时 rendezvous 放系统 temp，
    # 自动清理；数据、checkpoint 和验证记录仍放本次任务的 tmp_path。
    with tempfile.TemporaryDirectory(prefix="idea8_ddp_store_") as store_dir:
        context = mp.spawn(_worker, args=(str(tmp_path), str(Path(store_dir) / "store"), loss_mode, head_variant), nprocs=2, join=False)
        deadline = time.monotonic() + 240
        try:
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    raise TimeoutError("DDP regression timed out (possible collective deadlock)")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
    load = lambda name: torch.load(tmp_path / name / "checkpoint_last.pt", weights_only=False)
    structure, confidence = load("structure"), load("confidence")
    for name, last in (("structure", structure), ("confidence", confidence)):
        best = torch.load(tmp_path / name / "checkpoint_best.pt", weights_only=False)
        assert best["epoch"] == last["selection_state"]["best_epoch"]
        assert best["metrics"][best["selection_config"]["metric"]] == last["selection_state"]["best_score"]
        assert last["supervision_config"]["mode"] == loss_mode
        assert last["metrics"]["by_scene"]["c"]["samples"] == 1
    for resumed, uninterrupted in ((structure, load("continuous")), (confidence, load("confidence_continuous"))):
        for key in resumed["state_dict"]:
            torch.testing.assert_close(resumed["state_dict"][key], uninterrupted["state_dict"][key], rtol=0, atol=0)
        assert resumed["world_size"] == 2 and len(resumed["rng_states"]) == 2
        assert resumed["rng_states"][0]["random_state"] != resumed["rng_states"][1]["random_state"]
        assert not any(key.startswith("module.") for key in resumed["state_dict"])
        assert resumed["metrics"]["train_samples"] == 4  # 3 个原始样本补齐到 4 个。
        assert resumed["metrics"]["validation_samples"] == 1
    for key in structure["state_dict"]:
        if not key.startswith("confidence."):
            torch.testing.assert_close(structure["state_dict"][key], confidence["state_dict"][key], rtol=0, atol=0)
    for name in ("structure", "confidence"):
        rows = (tmp_path / name / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(row)["epoch"] for row in rows] == [0, 1]  # 只有 rank 0 写日志。
        config = json.loads((tmp_path / name / "run_config.json").read_text(encoding="utf-8"))
        assert config["global_batch_size"] == 4 and config["world_size"] == 2
        assert config["log_interval"] == 1  # 续训可改日志间隔，结果仍逐参数一致。
    rank0 = (tmp_path / "stdout_rank0.log").read_text(encoding="utf-8")
    rank1 = (tmp_path / "stdout_rank1.log").read_text(encoding="utf-8")
    for tag in ("[Progress]", "[Train]", "[Checkpoint]", "[Epoch]"):
        assert tag in rank0 and tag not in rank1
    for phase in ("structure", "confidence"):
        for stage in ("train", "validation"):
            assert f"[Progress] phase={phase} stage={stage} epoch=2/2 step=1/1" in rank0
    assert "status=waiting_for_ranks" in rank0 and "status=complete" in rank0
    assert all((tmp_path / f"nonfinite_rank{rank}.txt").exists() for rank in range(2))
    assert not (tmp_path / "nonfinite/checkpoint_last.pt").exists()

    # 校验分布式汇总等于完整验证集的单进程测量；DDP checkpoint 可直接用于普通推理。
    import train_event_structure_head as training
    from models.event_structure_head import load_structure_checkpoint
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    head, checkpoint, operator = load_structure_checkpoint(tmp_path / "confidence/checkpoint_last.pt")
    args = training.parse_args(["--data_root", str(tmp_path / "data"), "--scene_split", str(tmp_path / "split.json"),
                                "--output_dir", str(tmp_path / "invalid"), "--device", "cpu"])
    data = training.SDEEventStructureDataset("test", args,
        training.StructureEventConfig.from_dict(checkpoint["event_config"]), args.data_root, ["c"])
    with torch.no_grad():
        metrics = training.phase_loss(head, training.collate_structure([data[0]]), "confidence", operator, .1, "cpu")
    for key, actual in zip(("validation_loss", "edge_mae", "confidence_mae"), metrics):
        assert float(actual) == pytest.approx(checkpoint["metrics"][key], abs=1e-7)
    args.resume_from_checkpoint = str(tmp_path / "structure/checkpoint_last.pt")
    with pytest.raises(ValueError, match="world_size mismatch"):
        training.main(args)
