"""用独立 train/validation 场景标定 idea7 的固定 RGB 事件统计代理。

正常光只在本程序中由独立 --gt-manifest 读取。test 场景不能进入拟合或验收。
验收失败也输出带 failed 状态的 JSON；推理代理会拒绝加载失败配置。
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import torch

from dataloaders.sde_event_pair_dataset import (
    CONTRACT_KEYS, SDEEventPairDataset, content_hash, load_rgb, resolve_path,
)
from models.event_observation_proxy import (
    DEFAULT_PROXY_CONFIG, FixedObservationEnergy, log_luminance, validate_proxy_config,
)


def fit_positive_scale(x, y, weights=None, sigma_floor=1e-3, iterations=30):
    """单参数稳健回归，过原点且 a>0；无信息或负相关不伪装为有效标定。"""
    x, y = torch.as_tensor(x, dtype=torch.float64).flatten(), torch.as_tensor(y, dtype=torch.float64).flatten()
    weights = torch.ones_like(x) if weights is None else torch.as_tensor(weights, dtype=torch.float64).flatten()
    if x.shape != y.shape or x.shape != weights.shape or not x.numel():
        raise ValueError("Calibration arrays must be nonempty with equal shape")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Calibration arrays must be finite, weights nonnegative")
    # 无效块不能通过大量零残差把 MAD 压到下限。
    valid = weights > 0
    x, y, weights = x[valid], y[valid], weights[valid]
    if not x.numel():
        raise ValueError("No valid weighted calibration blocks")
    denominator = (weights*x*x).sum()
    if denominator <= 1e-12 or weights.sum() <= 0:
        raise ValueError("No informative image differences for positive-scale calibration")
    a = ((weights*x*y).sum()/denominator).clamp_min(1e-8)
    for _ in range(iterations):
        residual = y-a*x
        median = residual.median()
        scale = (1.4826*(residual-median).abs().median()).clamp_min(sigma_floor)
        robust = (1.345*scale/residual.abs().clamp_min(1e-12)).clamp_max(1)
        effective = weights*robust
        updated = ((effective*x*y).sum()/(effective*x*x).sum().clamp_min(1e-12)).clamp_min(1e-8)
        if float((updated-a).abs()) <= 1e-10*max(float(a), 1.):
            a = updated
            break
        a = updated
    residual = y-a*x
    sigma = max(float(1.4826*(residual-residual.median()).abs().median()), sigma_floor)
    return float(a), sigma


def check_proxy_gradient(config):
    """小型合成梯度方向/有限差分检查，不代替真实图像质量验证。"""
    scalar = torch.tensor(0.4, requires_grad=True)
    reference = torch.full((1, 3, 2, 2), 0.3)
    offset = log_luminance(reference, config).mean()
    base = log_luminance(scalar.expand(1, 3, 2, 2), config).mean()-offset
    target = base.detach()+0.2
    loss = (base-target).square()
    grad, = torch.autograd.grad(loss, scalar)
    step = 1e-4
    def evaluate(value):
        return float((log_luminance(torch.full((1, 3, 2, 2), value), config).mean()-offset-target).square())
    finite_difference = (evaluate(0.4+step)-evaluate(0.4-step))/(2*step)
    passed = bool(torch.isfinite(grad)) and float(grad) < 0 and abs(float(grad)-finite_difference) < max(1e-3, abs(finite_difference)*0.01)
    return dict(passed=passed, gradient=float(grad), finite_difference=finite_difference,
                scope="synthetic_rgb_proxy_only")


def _weighted_mae(prediction, target, weights):
    return float(((prediction-target).abs()*weights).sum()/weights.sum().clamp_min(1e-12))


def validate_group(records, a, min_improvement, min_wrong_time_improvement,
                   min_common_fraction=0.1):
    if len(records) < 2:
        return dict(passed=False, reason="Need at least two usable validation frame pairs for wrong-time control")
    # 错时对照只在同一几何/块布局内轮换，绝不插值原始测量去凑形状。
    by_shape = defaultdict(list)
    for record in records:
        by_shape[tuple(record["x"].shape)].append(record)
    if any(len(items) < 2 for items in by_shape.values()):
        return dict(passed=False, reason="Every validation block geometry needs two pairs for wrong-time control")
    xs, ys, ws, wrongs = [], [], [], []
    for items in by_shape.values():
        for index, record in enumerate(items):
            donor = items[(index+1) % len(items)]
            # 错时 target 在 donor 的无效块中是占位零，不能用这些零夸大对照误差。
            # 正确配对与错时对照严格共享 current/donor 块有效性的交集和权重。
            common = (record["w"] > 0) & (donor["w"] > 0)
            fraction = float(common.sum()) / max(int((record["w"] > 0).sum()), 1)
            if not bool(common.any()) or fraction < min_common_fraction:
                return dict(passed=False, reason="Insufficient common valid blocks for wrong-time control",
                            pair=record.get("pair"), common_valid_fraction=fraction)
            xs.append(record["x"].flatten())
            ys.append(record["y"].flatten())
            ws.append((record["w"]*common).flatten())
            wrongs.append(donor["y"].flatten())
    x, y, weights, wrong = map(torch.cat, (xs, ys, ws, wrongs))
    prediction = a*x
    mae = _weighted_mae(prediction, y, weights)
    zero_mae = _weighted_mae(torch.zeros_like(y), y, weights)
    wrong_mae = _weighted_mae(prediction, wrong, weights)
    valid_sign = (y.abs() > 1e-6) & (weights > 0)
    sign = float((torch.sign(prediction[valid_sign]) == torch.sign(y[valid_sign])).float().mean()) if valid_sign.any() else None
    relative = (zero_mae-mae)/max(zero_mae, 1e-12)
    wrong_relative = (wrong_mae-mae)/max(wrong_mae, 1e-12)
    passed = zero_mae > 1e-8 and wrong_mae > 1e-8 and relative >= min_improvement and wrong_relative >= min_wrong_time_improvement
    activities = sorted(r["activity"] for r in records)
    median_activity = activities[len(activities)//2]
    strata = {}
    for label, subset in (("lower_activity", [r for r in records if r["activity"] <= median_activity]),
                           ("higher_activity", [r for r in records if r["activity"] > median_activity])):
        if subset:
            strata[label] = dict(pairs=len(subset), mae=_weighted_mae(
                a*torch.cat([r["x"].flatten() for r in subset]),
                torch.cat([r["y"].flatten() for r in subset]),
                torch.cat([r["w"].flatten() for r in subset])))
    return dict(passed=passed, pairs=len(records), mae=mae, zero_prediction_mae=zero_mae,
                wrong_time_mae=wrong_mae, relative_improvement=relative,
                wrong_time_relative_improvement=wrong_relative, sign_agreement=sign,
                activity_strata=strata, reason="" if passed else "Held-out proxy failed zero/wrong-time comparison")


def calibrate(pair_manifest, gt_manifest, data_root, output, proxy_config=None,
              min_improvement=0.05, min_wrong_time_improvement=0.01, report_path=None):
    config = copy.deepcopy(DEFAULT_PROXY_CONFIG)
    if proxy_config:
        config.update(proxy_config)
    validate_proxy_config(config)
    if not 0 < min_improvement < 1 or not 0 < min_wrong_time_improvement < 1:
        raise ValueError("Validation improvement thresholds must be fixed positive fractions below one")
    dataset = SDEEventPairDataset(pair_manifest, data_root, validate_files=True)
    if any(pair["split"] == "test" for pair in dataset.pairs):
        raise ValueError("Calibration input must contain train/validation only; never test GT")
    with Path(gt_manifest).open(encoding="utf-8") as handle:
        gt_document = json.load(handle)
    if gt_document.get("schema_version") != 1:
        raise ValueError("GT manifest requires schema_version=1")
    gt_by_key = {}
    for item in gt_document.get("pairs", []):
        key = (item["scene_id"], item["curr_id"])
        if key in gt_by_key:
            raise ValueError("Duplicate GT pair entry")
        gt_by_key[key] = item
    allowed = {(p["scene_id"], p["curr_id"]) for p in dataset.pairs}
    if set(gt_by_key)-allowed:
        raise ValueError("GT manifest contains frames outside declared calibration splits")
    groups = defaultdict(lambda: dict(train=[], validation=[], metadata=[], skipped=[]))
    for index, pair in enumerate(dataset.pairs):
        bucket = groups[pair["observation_group"]]
        bucket["metadata"].append(pair)
        sample = dataset[index]
        if sample["skip_reason"]:
            bucket["skipped"].append(dict(pair=sample["metadata"]["pair_key"], reason=sample["skip_reason"]))
            continue
        key = (pair["scene_id"], pair["curr_id"])
        if key not in gt_by_key:
            raise ValueError(f"Missing explicit normal-light pair for calibration: {key}")
        target = gt_by_key[key]
        # GT 先走与模型输入相同的中心裁剪/缩放，energy 内再回到 native crop 网格。
        previous = load_rgb(resolve_path(data_root, target["normal_prev"]), pair["geometry"]).unsqueeze(0)
        current = load_rgb(resolve_path(data_root, target["normal_curr"]), pair["geometry"]).unsqueeze(0)
        energy = FixedObservationEnergy(previous, current, sample["measurement"], sample["metadata"],
                                         config, dict(a=1., sigma=config["sigma_floor"]))
        if energy.skip_reason:
            bucket["skipped"].append(dict(pair=sample["metadata"]["pair_key"], reason=energy.skip_reason))
            continue
        record = dict(x=energy.features(current).detach().double(), y=energy.target.double(),
                      w=energy.weights.double(), activity=energy.diagnostics["event_activity"],
                      pair=sample["metadata"]["pair_key"], metadata=copy.deepcopy(pair))
        bucket[pair["split"]].append(record)
    gradient_check = check_proxy_gradient(config)
    results = {}
    for name, bucket in groups.items():
        pairs = bucket["metadata"]
        conditions = pairs[0]["observation_conditions"]
        if any(p["observation_conditions"] != conditions for p in pairs):
            raise ValueError(f"Mixed acquisition/ISP conditions inside observation_group {name}")
        if not any(p["t_prev"] is not None and not p.get("skip_reason") for p in pairs):
            results[name] = dict(status="failed", reason="No usable intervals")
            continue
        # 适用范围只来自真正进入留出验证的记录；train-only/被跳过样本不扩张范围。
        support_tuples = sorted({(r["metadata"]["geometry"]["sensor_height"],
                                  r["metadata"]["geometry"]["sensor_width"],
                                  r["metadata"]["t_curr"]-r["metadata"]["t_prev"])
                                 for r in bucket["validation"]})
        supports = [dict(sensor_height=h, sensor_width=w, dt=dt) for h, w, dt in support_tuples]
        duration = [v[2] for v in support_tuples]
        metadata = dict(conditions=conditions, dt_range=[min(duration), max(duration)] if duration else [],
                        sensor_sizes=[list(v) for v in sorted({(h, w) for h, w, _ in support_tuples})],
                        validated_supports=supports,
                        skipped=bucket["skipped"], train_pairs=len(bucket["train"]))
        try:
            if not bucket["train"]:
                raise ValueError("No usable training frame pairs")
            x, y, weights = (torch.cat([r[key].flatten() for r in bucket["train"]]) for key in ("x", "y", "w"))
            a, sigma = fit_positive_scale(x, y, weights, config["sigma_floor"])
            validation = validate_group(bucket["validation"], a, min_improvement, min_wrong_time_improvement)
            passed = validation["passed"] and gradient_check["passed"]
            results[name] = dict(metadata, a=a, sigma=sigma, validation=validation,
                                 status="validated" if passed else "failed")
        except ValueError as exc:
            results[name] = dict(metadata, status="failed", reason=str(exc))
    status = "validated" if results and all(g["status"] == "validated" for g in results.values()) else "failed"
    document = dict(schema_version=1, status=status,
                    scientific_scope="empirical_processed_RGB_net_event_proxy_not_RAW_likelihood",
                    proxy=config, groups=results,
                    contract={key: dataset.settings[key] for key in CONTRACT_KEYS},
                    manifest_hash=dataset.manifest["manifest_hash"], gt_manifest_hash=content_hash(gt_document),
                    scenes={split: sorted({p["scene_id"] for p in dataset.pairs if p["split"] == split}) for split in ("train", "validation")},
                    acceptance=dict(min_relative_improvement=min_improvement,
                                    min_wrong_time_relative_improvement=min_wrong_time_improvement,
                                    min_wrong_time_common_valid_fraction=0.1,
                                    wrong_time_control="cyclic_pair_measurement_swap_within_same_block_geometry"),
                    gradient_check=gradient_check, actual_model_quality_evaluated=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(document, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if report_path:
        lines = ["# idea7 事件观测代理标定报告", "", f"状态：{status}", "",
                 "这是 processed RGB 的经验代理验证；没有运行 OSEDiff，也没有证明纠错改善图像质量。", ""]
        for name, group in results.items():
            lines.extend([f"## {name}", "", "```json", json.dumps(group, ensure_ascii=False, indent=2), "```", ""])
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--gt-manifest", required=True, help="独立JSON：schema_version=1,pairs:[scene_id,curr_id,normal_prev,normal_curr]")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--proxy-config", help="可选JSON覆盖已声明的物理近似/可靠性参数")
    parser.add_argument("--min-improvement", type=float, default=0.05)
    parser.add_argument("--min-wrong-time-improvement", type=float, default=0.01)
    args = parser.parse_args(argv)
    proxy_config = json.loads(Path(args.proxy_config).read_text(encoding="utf-8")) if args.proxy_config else None
    document = calibrate(args.pair_manifest, args.gt_manifest, args.data_root, args.output,
                         proxy_config, args.min_improvement, args.min_wrong_time_improvement, args.report)
    print(json.dumps(dict(status=document["status"], groups=list(document["groups"]), output=args.output), ensure_ascii=False))
    return 0 if document["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
