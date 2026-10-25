"""检查显式帧对规格并生成规范化 manifest；不从排序/时间戳量级猜元数据。

先用 --write-example 输出带占位路径的 JSON 规格，再填写实际采集信息。
--spec 的格式就是最终 manifest 格式，但 geometry/hash 可以省略，由预检生成。
相接分片的边界事件只能出现在其中一个来源；两边都有时拒绝所有权歧义，不自动去重。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# 允许从任意工作目录执行 python scripts/build_event_pair_manifest.py。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.sde_event_pair_dataset import validate_manifest, resolve_path


def example_manifest():
    source = dict(path="scene/low/events.npz", coverage=[0, 300], format="raw")
    return dict(schema_version=1, settings=dict(
        timestamp_unit="us", timestamp_anchor="center", clock_offset=0,
        polarity_encoding="zero_one", spatial_mapping="identity", image_size=256,
        event_bins=5, clora_event_active_bins=5, rgb_event_active_bins=5), pairs=[dict(
            scene_id="scene", prev_id="previous", curr_id="current",
            low_prev="scene/low/previous.png", low_curr="scene/low/current.png",
            t_prev=100, t_curr=200, split="train", observation_group="camera_isp_exposure_A",
            observation_conditions=dict(exposure="REPLACE_WITH_ACQUISITION_VALUE",
                                        isp="REPLACE_WITH_ACQUISITION_VALUE"),
            condition_event_sources=dict(
                previous=dict(sources=[source], window=[0, 200], boundary="open_open"),
                current=dict(sources=[source], window=[100, 300], boundary="open_open")),
            measurement_event_sources=[source])])


def build_manifest(spec_path, data_root, output_path, require_measurement=True):
    with Path(spec_path).open(encoding="utf-8") as handle:
        spec = json.load(handle)
    return build_manifest_from_spec(spec, data_root, output_path, require_measurement)


def build_manifest_from_spec(spec, data_root, output_path, require_measurement=True):
    """同时供显式 JSON 与 SDE 自动扫描器使用，统一校验和文件指纹语义。"""
    if not Path(data_root).is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    result = validate_manifest(spec, data_root, check_files=True,
                               require_measurement=require_measurement)
    # 内容指纹包含原始文件字节，记录预检时的数据身份，不把文件存在误作真实标定。
    paths = set()
    for pair in result["pairs"]:
        paths.add(pair["low_curr"])
        if pair["low_prev"]:
            paths.add(pair["low_prev"])
        for condition in pair["condition_event_sources"].values():
            if condition:
                paths.update(s["path"] for s in condition["sources"])
        if require_measurement and not pair.get("skip_reason"):
            paths.update(s["path"] for s in pair["measurement_event_sources"])
    fingerprints = {}
    for relative in sorted(paths):
        digest = hashlib.sha256()
        path = resolve_path(data_root, relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024*1024), b""):
                digest.update(block)
        fingerprints[relative] = dict(size_bytes=path.stat().st_size, sha256=digest.hexdigest())
    result["file_fingerprints"] = fingerprints
    # 文件指纹也纳入 manifest hash；验证器负责 canonical JSON 的一致性。
    result = validate_manifest(result, data_root, check_files=False,
                               require_measurement=require_measurement)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", help="显式 JSON 配对规格；不猜文件名单位/事件覆盖")
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    parser.add_argument("--report", help="可选预检成功/失败 JSON 报告")
    parser.add_argument("--write-example", metavar="PATH", help="只写模板，不把占位模板当作有效数据")
    parser.add_argument("--baseline-only", action="store_true", help="仅验证模型条件通路，不读取反馈测量")
    args = parser.parse_args(argv)
    if args.write_example:
        Path(args.write_example).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_example).write_text(json.dumps(example_manifest(), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        return 0
    if not all((args.spec, args.data_root, args.output)):
        parser.error("--spec, --data-root and --output are required")
    try:
        result = build_manifest(args.spec, args.data_root, args.output, not args.baseline_only)
        report = dict(status="preflight_passed", pairs=len(result["pairs"]),
                      manifest_hash=result["manifest_hash"], calibration_performed=False)
        code = 0
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        report, code = dict(status="failed", error=str(exc)), 2
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
