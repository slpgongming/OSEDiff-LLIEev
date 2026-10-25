"""从 SDE 的 train/test/<scene>/{low,normal} 自动准备 idea7 输入并标定。

RGB、GT、条件分片分别按整数文件名排序后按索引配对，保留旧 dataset 语义。
反馈始终查询 low/lowlight_event.npz，不能拼接有重叠的逐帧条件分片。
默认 SDE 时间约定为 us / center / offset=0；这些是可覆盖的采集约定，
不是从时间戳量级推断的物理标定。仅 train 场景参与拟合和留出验证。
"""

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_event_observation import calibrate
from dataloaders.sde_event_pair_dataset import (
    _read_raw_events_cached, read_raw_events, validate_manifest,
)
from models.event_observation_proxy import EventObservationProxy
from scripts.build_event_pair_manifest import build_manifest_from_spec


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def timestamp_files(directory, suffix, exclude=()):
    files = [p for p in directory.iterdir() if p.is_file()
             and p.suffix.lower() == suffix and p.name not in exclude]
    for path in files:
        if not path.stem.isascii() or not path.stem.isdigit():
            raise ValueError(f"Expected integer timestamp filename: {path}")
    files.sort(key=lambda p: int(p.stem))
    if len({int(p.stem) for p in files}) != len(files):
        raise ValueError(f"Duplicate numeric timestamps: {directory} ({suffix})")
    return files


def discover_scenes(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"SDE split directory not found: {directory}")
    scenes = []
    for path in sorted(directory.iterdir()):
        if not path.is_dir():
            continue
        if (path / "low").is_dir():
            if not (path / "normal").is_dir():
                raise ValueError(f"Missing normal/ for index-paired GT: {path}")
            scenes.append(path)
    if not scenes:
        raise ValueError(f"No SDE scene directories containing low/ and normal/: {directory}")
    return scenes


def partition_scenes(train_scenes, validation_names=None, fraction=0.2, seed=42):
    """确定性的整场景留出，不把同一视频的相邻帧随机分到 train/validation。"""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    names = {scene.name for scene in train_scenes}
    if validation_names:
        selected = set(validation_names)
        if selected - names:
            raise ValueError(f"Unknown validation scenes: {sorted(selected - names)}")
    else:
        if len(train_scenes) < 2:
            raise ValueError("Need at least two train scenes or a separate --validation-dir; never use test GT")
        ordered = sorted(names, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
        count = min(len(ordered) - 1, max(1, math.ceil(len(ordered) * fraction)))
        selected = set(ordered[:count])
    if not selected or selected == names:
        raise ValueError("Training and validation must both contain at least one whole scene")
    return {name: "validation" if name in selected else "train" for name in names}


def _relative(path, root):
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"All SDE sources must be inside --data-root for portable manifests: {path}") from exc


def _condition_spec(path, root, settings):
    events = read_raw_events(path, settings["polarity_encoding"])
    # boundary=all 的 coverage 只包住文件负载，绝不用于声称反馈区间已完整采集。
    start = int(events.t[0]) if len(events) else int(path.stem)
    end = max(start + 1, int(events.t[-1])) if len(events) else start + 1
    source = dict(path=_relative(path, root), coverage=[start, end], format="raw",
                  coverage_basis="payload_extent_condition_only")
    return dict(sources=[source], boundary="all")


def scan_scene(scene, split, root, settings, group, seed_offset=0, gap_factor=1.5):
    """解析一个场景，返回无 GT 的帧对、独立 GT 索引及采集范围诊断。"""
    low = timestamp_files(scene / "low", ".png")
    normal = timestamp_files(scene / "normal", ".png")
    fragments = timestamp_files(scene / "low", ".npz", ("lowlight_event.npz", "normalight_event.npz"))
    if len(low) < 2 or len(normal) != len(low):
        raise ValueError(f"{scene.name}: need >=2 low PNGs and exactly as many normal PNGs; "
                         f"got {len(low)} low / {len(normal)} normal (no silent truncation)")
    if fragments and len(fragments) != len(low):
        raise ValueError(f"{scene.name}: partial condition fragments ({len(fragments)} vs {len(low)} frames)")
    stream_path = scene / "low" / "lowlight_event.npz"
    if not stream_path.is_file():
        raise FileNotFoundError(f"Missing complete raw stream: {stream_path}; overlapping frame fragments cannot be concatenated")
    # 场景切换时释放旧整流；每个场景内缓存三份来源，避免反复读大文件。
    _read_raw_events_cached.cache_clear()
    stream = read_raw_events(stream_path, settings["polarity_encoding"])
    if len(stream) < 2 or stream.t[0] == stream.t[-1]:
        raise ValueError(f"Cannot establish a conservative observed stream span: {stream_path}")
    lo, hi = int(stream.t[0]), int(stream.t[-1])
    source = dict(path=_relative(stream_path, root), coverage=[lo, hi], format="raw",
                  coverage_basis="observed_full_stream_span_conservative")
    times = [int(p.stem) for p in low]
    differences = [b - a for a, b in zip(times, times[1:])]
    median_dt = float(np.median(differences))
    conditions, sizes = [], []
    for index, (image_path, normal_path) in enumerate(zip(low, normal)):
        with Image.open(image_path) as image, Image.open(normal_path) as target:
            if image.size != target.size:
                raise ValueError(f"RGB/GT geometry mismatch: {image_path} / {normal_path}")
            sizes.append(image.size)
        if fragments:
            conditions.append(_condition_spec(fragments[index], root, settings))
        else:
            start = lo if index == 0 else times[index - 1] + settings["clock_offset"]
            end = hi if index == len(low) - 1 else times[index + 1] + settings["clock_offset"]
            # 和旧 whole-stream 条件通路一致；缺采集覆盖时不把边缘补成零事件。
            start, end = max(start, lo), min(end, hi)
            if start >= end:
                raise ValueError(f"No whole-stream condition window for {image_path}")
            conditions.append(dict(sources=[source], window=[start, end], boundary="open_open"))
    if len(set(sizes)) != 1:
        raise ValueError(f"Sensor geometry changes inside scene {scene.name}")
    width, height = sizes[0]
    if (np.any(stream.x < 0) or np.any(stream.x >= width)
            or np.any(stream.y < 0) or np.any(stream.y >= height)):
        raise ValueError(f"Raw stream has coordinates outside RGB geometry: {stream_path}")
    pairs, gt_pairs, evaluation, gaps = [], [], [], []
    for index, image_path in enumerate(low):
        previous = index - 1 if index else None
        first = previous is None
        pair = dict(scene_id=scene.name, curr_id=image_path.stem,
                    prev_id=low[previous].stem if not first else None,
                    low_curr=_relative(image_path, root),
                    low_prev=_relative(low[previous], root) if not first else None,
                    t_curr=times[index], t_prev=times[previous] if not first else None,
                    split=split, **copy.deepcopy(group),
                    seed_indices=dict(current=seed_offset + index,
                                      previous=seed_offset + previous if not first else None),
                    condition_event_sources=dict(current=conditions[index],
                                                 previous=conditions[previous] if not first else None),
                    measurement_event_sources=[] if first else [source])
        if not first:
            start, end = times[previous] + settings["clock_offset"], times[index] + settings["clock_offset"]
            reason = ("outside_observed_stream_span" if start < lo or end > hi
                      else "frame_time_gap" if end - start > gap_factor * median_dt else "")
            if reason:
                pair.update(skip_reason="time_gap", measurement_event_sources=[])
                gaps.append(dict(curr_id=image_path.stem, reason=reason))
            gt_pairs.append(dict(scene_id=scene.name, curr_id=image_path.stem,
                                 normal_prev=_relative(normal[previous], root),
                                 normal_curr=_relative(normal[index], root)))
        pairs.append(pair)
        evaluation.append(dict(scene_id=scene.name, frame_id=image_path.stem,
                               normal_path=_relative(normal[index], root)))
    offsets = [int(target.stem) - int(image.stem) for image, target in zip(low, normal)]
    report = dict(scene_id=scene.name, split=split, frames=len(low), sensor_size=[height, width],
                  frame_dt_histogram=dict(sorted(Counter(differences).items())),
                  normal_minus_low_timestamp_range=[min(offsets), max(offsets)],
                  condition_mode="sorted_index_fragments_all" if fragments else "whole_stream_neighbor_window",
                  measurement_source=source, raw_event_count=len(stream), skipped_intervals=gaps,
                  observation_group=group["observation_group"])
    return pairs, gt_pairs, evaluation, report


def _group_for_scene(config, name):
    group = config.get("scenes", {}).get(name, config.get("default"))
    if not isinstance(group, dict) or set(group) != {"observation_group", "observation_conditions"}:
        raise ValueError(f"Group config for {name} requires observation_group and observation_conditions")
    if not isinstance(group["observation_group"], str) or not group["observation_group"]:
        raise ValueError(f"Empty observation_group for {name}")
    if not isinstance(group["observation_conditions"], dict) or not group["observation_conditions"]:
        raise ValueError(f"Empty observation_conditions for {name}")
    return group


def check_test_support(calibration, manifest):
    if calibration["status"] != "validated":
        return dict(compatible=False, reason="calibration_failed", unsupported=[])
    proxy = EventObservationProxy(calibration)
    issues = {}
    for pair in manifest["pairs"]:
        try:
            proxy.validate_metadata(dict(pair, settings=manifest["settings"]))
        except ValueError as exc:
            key = (pair["scene_id"], str(exc), None if pair["t_prev"] is None else pair["t_curr"] - pair["t_prev"])
            issues[key] = issues.get(key, 0) + 1
    return dict(compatible=not issues, unsupported=[dict(scene_id=s, reason=r, dt=dt, frames=n)
                                                  for (s, r, dt), n in issues.items()])


def write_supported_subset(report, manifest, evaluation, root, output):
    """只按已标定的采集范围筛选完整场景，不读取测试 GT 像素或图像质量指标。"""
    excluded = {item["scene_id"] for item in report["test_compatibility"]["unsupported"]}
    included = {p["scene_id"] for p in manifest["pairs"]} - excluded
    report["supported_test_scenes"] = sorted(included)
    report["excluded_test_scenes"] = sorted(excluded)
    if not included:
        return
    selected = copy.deepcopy(manifest)
    selected["pairs"] = [p for p in selected["pairs"] if p["scene_id"] in included]
    paths = set()
    for pair in selected["pairs"]:
        paths.update(p for p in (pair["low_curr"], pair["low_prev"]) if p)
        for condition in pair["condition_event_sources"].values():
            if condition:
                paths.update(s["path"] for s in condition["sources"])
        paths.update(s["path"] for s in pair["measurement_event_sources"])
    selected["file_fingerprints"] = {p: value for p, value in selected.get("file_fingerprints", {}).items() if p in paths}
    # 来源已经全量预检；对子集重新计算清单 hash，保留原 seed_indices。
    selected = validate_manifest(selected, root, check_files=False)
    write_json(output / "pairs_test_supported.json", selected)
    write_json(output / "evaluation_test_supported.json", dict(
        schema_version=1, frames=[item for item in evaluation if item["scene_id"] in included]))
    report["files"].update(supported_pair_manifest="pairs_test_supported.json",
                           supported_evaluation_manifest="evaluation_test_supported.json")
    report["supported_subset_inference"] = dict(
        data_root=str(root), pair_manifest=str(output / "pairs_test_supported.json"),
        evaluation_manifest=str(output / "evaluation_test_supported.json"),
        observation_calibration=str(output / "calibration.json"))


def prepare(args):
    """扫描、预检、标定，再核对测试范围；各阶段结果保存在独立 JSON 中。"""
    root, output = Path(args.data_root).resolve(), Path(args.output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not math.isfinite(args.gap_factor) or args.gap_factor <= 1:
        raise ValueError("gap_factor must be finite and greater than one")
    test_scenes = discover_scenes(root / args.test_dir)
    jobs = [(scene, "test") for scene in test_scenes]
    if not args.test_only:
        train_path = (root / args.train_dir).resolve()
        if train_path.name.lower() in {"test", "testing"}:
            raise ValueError("A test directory cannot be used as calibration training data")
        train_scenes = discover_scenes(train_path)
        validation_dir = args.validation_dir
        if not validation_dir and not args.validation_scenes:
            found = [name for name in ("validation", "val") if (root / name).is_dir()]
            if len(found) > 1:
                raise ValueError("Both val/ and validation/ exist; select --validation-dir explicitly")
            validation_dir = found[0] if found else None
        if validation_dir:
            if args.validation_scenes:
                raise ValueError("Use either --validation-dir or --validation-scenes")
            val_path = (root / validation_dir).resolve()
            if val_path.name.lower() in {"test", "testing"}:
                raise ValueError("A test directory cannot be used as calibration validation data")
            jobs += [(scene, "train") for scene in train_scenes]
            jobs += [(scene, "validation") for scene in discover_scenes(val_path)]
        else:
            assignment = partition_scenes(train_scenes, args.validation_scenes,
                                          args.validation_fraction, args.split_seed)
            jobs += [(scene, assignment[scene.name]) for scene in train_scenes]
    if len({scene.name for scene, _ in jobs}) != len(jobs) or len({scene.resolve() for scene, _ in jobs}) != len(jobs):
        raise ValueError("Scene leakage: scene names/paths must be disjoint across train, validation and test")
    settings = dict(timestamp_unit=args.timestamp_unit, timestamp_anchor=args.timestamp_anchor,
                    clock_offset=args.clock_offset, polarity_encoding=args.polarity_encoding,
                    spatial_mapping="identity", image_size=256, event_bins=args.event_bins,
                    clora_event_active_bins=args.clora_event_active_bins,
                    rgb_event_active_bins=args.rgb_event_active_bins)
    if args.group_config:
        groups = json.loads(Path(args.group_config).read_text(encoding="utf-8"))
    else:
        # 未知曝光/ISP 如实记录；同组只是默认实验假设，不是硬件参数已被识别。
        groups = dict(default=dict(observation_group="sde_default", observation_conditions=dict(
            dataset="SDE", exposure="unknown", isp="unknown", grouping="assumed_shared")))
    unknown = set(groups.get("scenes", {})) - {scene.name for scene, _ in jobs}
    if unknown:
        raise ValueError(f"Group config names unknown scenes: {sorted(unknown)}")
    documents = {name: dict(schema_version=1, settings=settings, pairs=[]) for name in ("test", "trainval")}
    gt, evaluation, reports, offsets = [], [], [], {}
    for scene, split in jobs:
        print(f"[scan] {split}/{scene.name}", flush=True)
        origin = str(scene.parent.resolve())  # 留出后仍保留旧 train/test 数据集的全局 seed 索引。
        pairs, target_pairs, frames, report = scan_scene(
            scene, split, root, settings, _group_for_scene(groups, scene.name),
            offsets.get(origin, 0), args.gap_factor)
        offsets[origin] = offsets.get(origin, 0) + len(pairs)
        documents["test" if split == "test" else "trainval"]["pairs"].extend(pairs)
        if split == "test":
            evaluation.extend(frames)
        else:
            gt.extend(target_pairs)
        reports.append(report)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="building", data_root=str(root), settings=settings, scenes=reports,
                  pairing="numeric_sorted_index_not_equal_filenames", group_config=groups,
                  calibration_uses_test_gt=False, split_seed=args.split_seed,
                  assumptions=["Timestamp unit/anchor/offset use the declared SDE profile.",
                               "Frames and events are already spatially registered.",
                               "Whole raw stream is continuous inside its observed first/last timestamp."])
    if not args.group_config:
        report["assumptions"].append("Unknown exposure/ISP: all scenes provisionally share one observation group.")
    write_json(output / "preparation_report.json", report)
    for name, document in documents.items():
        if not document["pairs"]:
            continue
        print(f"[preflight] {name}: {len(document['pairs'])} frames; checking files and hashing sources", flush=True)
        documents[name] = build_manifest_from_spec(document, root, output / f"pairs_{name}.json")
    write_json(output / "evaluation_test.json", dict(schema_version=1, frames=evaluation))
    report["files"] = dict(pair_manifest="pairs_test.json", evaluation_manifest="evaluation_test.json")
    if not args.test_only:
        write_json(output / "gt_trainval.json", dict(schema_version=1, pairs=gt))
        report["files"].update(calibration_pairs="pairs_trainval.json", calibration_gt="gt_trainval.json")
    report["inference"] = dict(data_root=str(root), pair_manifest=str(output / "pairs_test.json"),
                               evaluation_manifest=str(output / "evaluation_test.json"))
    code = 0
    if args.test_only or args.skip_calibration:
        report.update(status="manifests_ready", calibration_status="not_run")
    else:
        print("[calibration] fit on train scenes, evaluate on held-out validation scenes", flush=True)
        config = json.loads(Path(args.proxy_config).read_text(encoding="utf-8")) if args.proxy_config else None
        calibration = calibrate(output / "pairs_trainval.json", output / "gt_trainval.json", root,
                                output / "calibration.json", config, args.min_improvement,
                                args.min_wrong_time_improvement, output / "calibration_report.md")
        report["calibration_status"] = calibration["status"]
        report["files"].update(calibration="calibration.json", calibration_report="calibration_report.md")
        report["test_compatibility"] = check_test_support(calibration, documents["test"])
        if calibration["status"] != "validated":
            report["status"], code = "calibration_failed", 2
        elif not report["test_compatibility"]["compatible"]:
            report["status"], code = "calibrated_but_test_incompatible", 3
            write_supported_subset(report, documents["test"], evaluation, root, output)
        else:
            report["status"] = "ready"
            report["inference"]["observation_calibration"] = str(output / "calibration.json")
    write_json(output / "preparation_report.json", report)
    return report, code


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="SDE release root containing train/ and test/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-dir", default="train", help="Relative to data-root")
    parser.add_argument("--test-dir", default="test", help="Relative to data-root")
    parser.add_argument("--validation-dir", help="Separate validation split; auto-detect val/ or validation/ if present")
    parser.add_argument("--validation-scenes", nargs="+", help="Hold out these scene names from train/")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-only", action="store_true", help="Only test/evaluation manifests; no calibration or train GT")
    parser.add_argument("--skip-calibration", action="store_true", help="Generate all manifests without fitting")
    parser.add_argument("--timestamp-unit", choices=("s", "ms", "us", "ns"), default="us")
    parser.add_argument("--timestamp-anchor", choices=("start", "center", "end"), default="center")
    parser.add_argument("--clock-offset", type=int, default=0, help="Low-RGB to event clock offset, NOT normal-RGB offset")
    parser.add_argument("--polarity-encoding", choices=("zero_one", "minus_plus"), default="zero_one")
    parser.add_argument("--event-bins", type=int, default=5)
    parser.add_argument("--clora-event-active-bins", type=int, default=5, help="Must match the selected checkpoint")
    parser.add_argument("--rgb-event-active-bins", type=int, default=5)
    parser.add_argument("--gap-factor", type=float, default=1.5, help="Skip feedback when dt exceeds this times scene median")
    parser.add_argument("--group-config", help="JSON: default/scenes -> {observation_group, observation_conditions}")
    parser.add_argument("--proxy-config", help="Optional fixed observation-proxy config JSON")
    parser.add_argument("--min-improvement", type=float, default=0.05)
    parser.add_argument("--min-wrong-time-improvement", type=float, default=0.01)
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU preparation/calibration; no GPU or model weights needed")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    try:
        report, code = prepare(args)
    except (ValueError, KeyError, OSError) as exc:
        report, code = dict(status="failed", error=str(exc)), 2
        # 保留已写入的场景报告，失败状态不能沿用上一轮的 ready 标记。
        path = Path(args.output_dir) / "preparation_report.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            report = {**previous, **report}
        write_json(path, report)
    finally:
        _read_raw_events_cached.cache_clear()
    print(json.dumps(dict(status=report["status"], output_dir=str(Path(args.output_dir).resolve()),
                          error=report.get("error")), ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
