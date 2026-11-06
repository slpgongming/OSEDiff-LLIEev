"""训练数值检查：所有 rank 先归约标志，再共同报错，避免单卡退出造成等待。"""
import torch


def require_finite(accelerator, tensors, *, phase, step, batch=None):
    # tensors 可为生成器，避免为梯度复制大张量；每项只生成一个布尔标量。
    names, flags = [], []
    for name, tensor in tensors:
        if tensor is not None:
            names.append(name)
            flags.append(torch.isfinite(tensor.detach()).all())
    local = torch.stack(flags) if flags else torch.ones(1, device=accelerator.device, dtype=torch.bool)
    failed = (~local.all()).to(device=accelerator.device, dtype=torch.int32)
    failed = accelerator.reduce(failed, reduction="sum")
    if failed.item():
        bad = [name for name, ok in zip(names, local.cpu().tolist()) if not ok]
        context = {key: batch.get(key) for key in ("scene_name", "frame_id", "low_path")} if batch else {}
        raise FloatingPointError(
            f"NaN/Inf: step={step}, phase={phase}, rank={accelerator.process_index}, "
            f"local_bad={bad or 'another rank'}, samples={context}"
        )
