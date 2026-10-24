"""idea7 的无 GT 帧对读取与严格区间测量。

条件 voxel 复现旧数据集的 test 中心裁剪/缩放/归一化；反馈计数则始终保留在
原始中心裁剪网格，不经过 voxel 归一化。时间、极性和文件覆盖必须显式声明。
"""

import copy
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from dataloaders.event_voxel_layout import concentrate_event_voxel


SCHEMA_VERSION = 1
TIME_UNITS = {"s", "ms", "us", "ns"}
POLARITIES = {"zero_one", "minus_plus"}
CONTRACT_KEYS = ("timestamp_unit", "timestamp_anchor", "clock_offset",
                 "polarity_encoding", "spatial_mapping", "image_size")


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an explicitly declared integer timestamp/value")
    return int(value)


def _safe_timestamps(values, path):
    """绝对时戳禁止经过 float32；浮点文件仅接受无歧义的整数时戳。"""
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"Invalid timestamp values: {path}")
    if np.issubdtype(values.dtype, np.floating):
        exact_limit = 2 ** (24 if values.dtype.itemsize <= 4 else 53)
        if np.any(np.abs(values) >= exact_limit) or np.any(values != np.floor(values)):
            raise ValueError(f"Timestamp precision/integer contract violated: {path}")
    if np.any(values >= 2**63) or np.any(values < -(2**63)):
        raise ValueError(f"Timestamp outside int64: {path}")
    return values.astype(np.int64, copy=False)


@dataclass
class RawEvents:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray

    def select(self, mask):
        return RawEvents(*(getattr(self, key)[mask] for key in ("t", "x", "y", "p")))

    def interval(self, start, end):
        # 时戳已在读取时确认有序；二分切片避免对整段流反复生成巨大布尔数组。
        lower = int(np.searchsorted(self.t, start, side="right"))
        upper = int(np.searchsorted(self.t, end, side="right"))
        return self.select(slice(lower, upper))

    def __len__(self):
        return len(self.t)


def read_raw_events(path, polarity_encoding, keys=None):
    path = Path(path).resolve()
    stat = path.stat()  # 文件被删时不允许由缓存掩盖错误。
    return _read_raw_events_cached(str(path), polarity_encoding,
                                   json.dumps(keys, sort_keys=True), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=2)
def _read_raw_events_cached(path, polarity_encoding, keys_json, size_bytes, mtime_ns):
    """支持字段 NPZ、结构化 arr_0 或 [N,4]；不允许 pickle 或预体素替代测量。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping = json.loads(keys_json) or dict(timestamp="timestamp", x="x", y="y", polarity="polarity")
    with np.load(path, allow_pickle=False) as data:
        if all(mapping[k] in data for k in ("timestamp", "x", "y", "polarity")):
            columns = [np.asarray(data[mapping[k]]) for k in ("timestamp", "x", "y", "polarity")]
        else:
            name = "arr_0" if "arr_0" in data else (data.files[0] if len(data.files) == 1 else None)
            if name is None:
                raise ValueError(f"Declare raw NPZ keys explicitly: {path}")
            arr = np.asarray(data[name])
            if arr.dtype.fields:
                columns = [arr[mapping[k]] for k in ("timestamp", "x", "y", "polarity")]
            elif arr.ndim == 2 and arr.shape[1] == 4:
                columns = [arr[:, i] for i in range(4)]
            else:
                raise ValueError(f"Measurement requires raw [t,x,y,p], not a voxel: {path}")
    if any(c.ndim != 1 for c in columns) or len({len(c) for c in columns}) != 1:
        raise ValueError(f"Event columns must have equal one-dimensional lengths: {path}")
    t = _safe_timestamps(columns[0], path)
    x, y, p = (np.asarray(c) for c in columns[1:])
    if not all(np.isfinite(c).all() for c in (x, y, p)):
        raise ValueError(f"Nonfinite event coordinates/polarity: {path}")
    if np.any(x != np.floor(x)) or np.any(y != np.floor(y)):
        raise ValueError(f"Identity registration requires integer pixel coordinates: {path}")
    allowed = (0, 1) if polarity_encoding == "zero_one" else (-1, 1)
    if polarity_encoding not in POLARITIES or not np.isin(p, allowed).all():
        raise ValueError(f"Polarity encoding mismatch: {path}")
    if len(t) > 1 and np.any(t[1:] < t[:-1]):
        raise ValueError(f"Event timestamps must be sorted: {path}")
    signed = np.where(p == 1, 1, -1).astype(np.int8)
    return RawEvents(t, x.astype(np.int64), y.astype(np.int64), signed)


def _source_coverage(source):
    coverage = source.get("coverage")
    if not isinstance(coverage, list) or len(coverage) != 2:
        raise ValueError("Every event source requires coverage=[start,end], including empty files")
    lo, hi = (_integer(v, "coverage") for v in coverage)
    if lo >= hi:
        raise ValueError("Event source coverage must have positive duration")
    return lo, hi


def load_event_sources(sources, root, settings, interval=None):
    """文件 coverage 是采集元数据，不能用首末事件时间推断无事件时段。"""
    if not sources:
        raise ValueError("Missing raw event sources")
    paths = [str(resolve_path(root, s["path"]).resolve()) for s in sources]
    if len(set(paths)) != len(paths):
        raise ValueError("Repeated event source would double-count measurements")
    ordered = sorted(sources, key=lambda s: _source_coverage(s)[0])
    bounds = [_source_coverage(s) for s in ordered]
    for left, right in zip(bounds, bounds[1:]):
        if left[1] > right[0]:
            raise ValueError("Overlapping source coverage would double-count measurements")
    if interval is not None:
        start, end = interval
        cursor = start
        for lo, hi in bounds:
            if hi <= cursor:
                continue
            if lo > cursor:
                break
            cursor = hi
        if cursor < end:
            raise ValueError(f"Event source coverage does not cover ({start},{end}]")
    chunks = []
    previous_end, previous_last_event = None, None
    for source, (lo, hi) in zip(ordered, bounds):
        if source.get("format", "raw") != "raw":
            raise ValueError("Measurement source must use raw events")
        events = read_raw_events(resolve_path(root, source["path"]),
                                 settings["polarity_encoding"], source.get("keys"))
        # coverage 只声明采集覆盖，不声明边界事件属于左还是右文件。
        if len(events) and (events.t[0] < lo or events.t[-1] > hi):
            raise ValueError(f"Events outside declared coverage: {source['path']}")
        if (previous_end == lo and previous_last_event == lo
                and len(events) and events.t[0] == lo):
            # 不能按 (t,x,y,p) 去重，因为完全相同的事件也可能是真实重复触发。
            raise ValueError("Ambiguous boundary ownership: adjacent sources both contain shared-boundary events")
        previous_end = hi
        previous_last_event = int(events.t[-1]) if len(events) else None
        if interval is not None:
            # 唯一的左开右闭约束来自观测查询窗口；保留仅存在右分片左端的事件。
            events = events.interval(start, end)
        chunks.append(events)
    return RawEvents(*(np.concatenate([getattr(c, k) for c in chunks])
                       for k in ("t", "x", "y", "p")))


def center_geometry(width, height, image_size):
    side = min(width, height)
    return dict(sensor_width=width, sensor_height=height,
                crop=[(height-side)//2, (width-side)//2, side, side],
                image_size=image_size, resize_mode="bilinear_align_corners_false")


def load_rgb(path, geometry):
    """返回 [3,S,S] 的 [0,1] RGB，严格复现旧 test 图像几何。"""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (geometry["sensor_width"], geometry["sensor_height"]):
            raise ValueError(f"RGB/geometry size mismatch: {path}")
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.
    top, left, height, width = geometry["crop"]
    tensor = tensor[:, top:top+height, left:left+width]
    size = geometry["image_size"]
    if (height, width) != (size, size):
        tensor = F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="bilinear",
                               align_corners=False)[0]
    return tensor


def count_interval(events, geometry):
    """计数只在原网格上裁剪；N+、N- 单独保存，抵消不等于不活动。"""
    top, left, height, width = geometry["crop"]
    if len(events) and (np.any(events.x < 0) or np.any(events.y < 0)
                       or np.any(events.x >= geometry["sensor_width"])
                       or np.any(events.y >= geometry["sensor_height"])):
        raise ValueError("Event coordinates outside declared sensor geometry")
    selected = events.select((events.x >= left) & (events.x < left+width)
                             & (events.y >= top) & (events.y < top+height))
    positive = torch.zeros((1, height, width), dtype=torch.float32)
    negative = torch.zeros_like(positive)
    for polarity, target in ((1, positive), (-1, negative)):
        subset = selected.select(selected.p == polarity)
        if len(subset):
            index = (torch.from_numpy(subset.y-top), torch.from_numpy(subset.x-left))
            target[0].index_put_(index, torch.ones(len(subset)), accumulate=True)
    return dict(positive=positive, negative=negative,
                valid_mask=torch.ones_like(positive, dtype=torch.bool))


def _condition_voxel(spec, root, settings, geometry):
    sources = spec.get("sources", [])
    boundary = spec.get("boundary")
    if boundary not in {"open_open", "open_closed", "all"}:
        raise ValueError("Condition boundary must explicitly be open_open/open_closed/all")
    bins = settings["event_bins"]
    top, left, height, width = geometry["crop"]
    if len(sources) == 1 and sources[0].get("format") == "voxel":
        if boundary != "all":
            raise ValueError("A condition voxel has no raw timestamps; boundary must be all")
        path = resolve_path(root, sources[0]["path"])
        with np.load(path, allow_pickle=False) as data:
            arr = np.asarray(data["arr_0" if "arr_0" in data else data.files[0]])
        voxel = torch.from_numpy(arr.copy()).float()
        if voxel.ndim != 3:
            raise ValueError("Condition voxel must be three-dimensional")
        if voxel.shape[0] != bins and voxel.shape[-1] == bins:
            voxel = voxel.permute(2, 0, 1).contiguous()
        if tuple(voxel.shape) != (bins, geometry["sensor_height"], geometry["sensor_width"]):
            raise ValueError("Condition voxel geometry/channels mismatch")
        if not torch.isfinite(voxel).all():
            raise ValueError("Nonfinite condition voxel")
        voxel = voxel[:, top:top+height, left:left+width]
    else:
        # all 模式保留旧分片读取的全部原始事件，包括来源起点上的事件。
        if boundary == "all" and len(sources) == 1:
            source = sources[0]
            if source.get("format", "raw") != "raw":
                raise ValueError("Unknown condition source format")
            lo, hi = _source_coverage(source)
            events = read_raw_events(resolve_path(root, source["path"]),
                                     settings["polarity_encoding"], source.get("keys"))
            if len(events) and (events.t[0] < lo or events.t[-1] > hi):
                raise ValueError(f"Condition events outside declared coverage: {source['path']}")
        else:
            window = spec.get("window")
            if boundary == "all" or not isinstance(window, list) or len(window) != 2:
                raise ValueError("Explicit condition window is required for raw stream slicing")
            start, end = (_integer(t, "condition window") for t in window)
            if start >= end:
                raise ValueError("Condition window must have positive duration")
            events = load_event_sources(sources, root, settings, (start, end))
            if boundary == "open_open":
                events = events.select(events.t < end)
        if len(events) and (np.any(events.x < 0) or np.any(events.y < 0)
                           or np.any(events.x >= geometry["sensor_width"])
                           or np.any(events.y >= geometry["sensor_height"])):
            raise ValueError("Condition event coordinates outside sensor geometry")
        events = events.select((events.x >= left) & (events.x < left+width)
                               & (events.y >= top) & (events.y < top+height))
        voxel = torch.zeros((bins, height, width), dtype=torch.float32)
        if len(events):
            # 与旧实现相同：裁剪后用实际首末事件定位 bin，先 int64 相减。
            relative = torch.from_numpy(events.t-events.t[0]).float()
            denominator = max(float(events.t[-1]-events.t[0]), 1.)
            channel = (relative / denominator * bins).long().clamp_(0, bins-1)
            indices = (channel, torch.from_numpy(events.y-top), torch.from_numpy(events.x-left))
            voxel.index_put_(indices, torch.from_numpy(events.p).float(), accumulate=True)
    size = geometry["image_size"]
    if (height, width) != (size, size):
        voxel = F.interpolate(voxel.unsqueeze(0), size=(size, size), mode="nearest")[0]
    return voxel


def condition_inputs(spec, root, settings, geometry):
    raw = _condition_voxel(spec, root, settings, geometry)
    clora = concentrate_event_voxel(raw, settings["clora_event_active_bins"])
    clora = (clora / clora.abs().mean((1, 2), keepdim=True).clamp_min(1e-6)).clamp(-5, 5) / 5
    refinement_raw = concentrate_event_voxel(raw, settings["rgb_event_active_bins"])
    refinement = torch.zeros_like(refinement_raw)
    nonzero = refinement_raw != 0
    values = refinement_raw[nonzero]
    if values.numel() and torch.isfinite(values.std(unbiased=False)) and values.std(unbiased=False) >= 1e-6:
        refinement[nonzero] = (values-values.mean()) / (values.std(unbiased=False)+1e-6)
    return clora, refinement


def validate_manifest(manifest, root, check_files=True, require_measurement=True):
    """返回规范化副本；禁止在推理清单里夹带正常光路径。"""
    result = copy.deepcopy(manifest)
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported pair manifest schema_version")
    settings = result.get("settings", {})
    required = (*CONTRACT_KEYS, "event_bins", "clora_event_active_bins", "rgb_event_active_bins")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"Missing explicit manifest settings: {missing}")
    if settings["timestamp_unit"] not in TIME_UNITS or settings["timestamp_anchor"] not in {"start", "center", "end"}:
        raise ValueError("Unknown timestamp unit/anchor")
    _integer(settings["clock_offset"], "clock_offset")
    if settings["polarity_encoding"] not in POLARITIES or settings["spatial_mapping"] != "identity":
        raise ValueError("First version requires explicit polarity and identity registration")
    for key in ("event_bins", "image_size", "clora_event_active_bins", "rgb_event_active_bins"):
        if _integer(settings[key], key) <= 0:
            raise ValueError(f"{key} must be positive")
    if any(settings[k] > settings["event_bins"] for k in ("clora_event_active_bins", "rgb_event_active_bins")):
        raise ValueError("active bins cannot exceed event_bins")
    if not result.get("pairs"):
        raise ValueError("Pair manifest must not be empty")
    seen, frame_contracts = set(), {}
    scene_splits = {}
    for pair in result["pairs"]:
        for key in ("scene_id", "curr_id", "prev_id", "low_curr", "low_prev", "t_curr", "t_prev",
                    "split", "observation_group", "observation_conditions", "condition_event_sources",
                    "measurement_event_sources"):
            if key not in pair:
                raise ValueError(f"Missing pair field: {key}")
        if any(k in pair for k in ("normal_prev", "normal_curr", "gt_prev", "gt_curr", "normal_path")):
            raise ValueError("GT paths belong in a separate calibration/evaluation manifest")
        if pair["split"] not in {"train", "validation", "test"}:
            raise ValueError("split must be train/validation/test")
        if any(not isinstance(pair[k], str) or not pair[k] for k in ("scene_id", "curr_id", "observation_group")):
            raise ValueError("Scene/frame/group identifiers must be nonempty strings")
        if pair["prev_id"] is not None and (not isinstance(pair["prev_id"], str) or not pair["prev_id"]):
            raise ValueError("Previous frame identifier must be a nonempty string or null")
        if not pair["observation_group"] or not isinstance(pair["observation_conditions"], dict) or not pair["observation_conditions"]:
            raise ValueError("Declare observation_group and acquisition/ISP conditions explicitly")
        scene = pair["scene_id"]
        if scene in scene_splits and scene_splits[scene] != pair["split"]:
            raise ValueError("Scene leakage across manifest splits")
        scene_splits[scene] = pair["split"]
        key = (scene, pair["curr_id"])
        if key in seen:
            raise ValueError(f"Duplicate current frame: {key}")
        seen.add(key)
        _integer(pair["t_curr"], "t_curr")
        first = pair["prev_id"] is None
        if first:
            if pair["low_prev"] is not None or pair["t_prev"] is not None or pair["measurement_event_sources"]:
                raise ValueError("First frame must have no previous frame or measurement")
            pair["skip_reason"] = "first_frame"
        else:
            if pair["low_prev"] is None or _integer(pair["t_prev"], "t_prev") >= pair["t_curr"]:
                raise ValueError("Pair timestamps must increase with an explicit previous RGB")
            if pair.get("skip_reason", "") not in {"", "time_gap", "scene_boundary"}:
                raise ValueError("Unsupported declared pair skip_reason")
        for role in (("current",) if first else ("previous", "current")):
            if role not in pair["condition_event_sources"]:
                raise ValueError(f"Missing {role} condition source contract")
        if check_files:
            with Image.open(resolve_path(root, pair["low_curr"])) as image:
                geometry = center_geometry(*image.size, settings["image_size"])
            if "geometry" in pair and pair["geometry"] != geometry:
                raise ValueError("Geometry must match original center-square test preprocessing")
            pair["geometry"] = geometry
            if not first:
                with Image.open(resolve_path(root, pair["low_prev"])) as image:
                    if image.size != (geometry["sensor_width"], geometry["sensor_height"]):
                        raise ValueError("Adjacent RGB frames must share registered dimensions")
            # 全文件预检会读取原始时戳，而不是只检查路径存在。
            for role in (("current",) if first else ("previous", "current")):
                _condition_voxel(pair["condition_event_sources"][role], root, settings, geometry)
            if require_measurement and not first and not pair.get("skip_reason"):
                interval = (pair["t_prev"]+settings["clock_offset"], pair["t_curr"]+settings["clock_offset"])
                count_interval(load_event_sources(pair["measurement_event_sources"], root, settings, interval), geometry)
        elif "geometry" not in pair:
            raise ValueError("Prevalidated manifest must include geometry when check_files=False")
        for role, ident, low in (("current", pair["curr_id"], pair["low_curr"]),
                                 ("previous", pair["prev_id"], pair["low_prev"])):
            if ident is None:
                continue
            timestamp = pair["t_curr"] if role == "current" else pair["t_prev"]
            contract = content_hash(dict(low=low, timestamp=timestamp, geometry=pair["geometry"],
                                         condition=pair["condition_event_sources"][role]))
            frame_key = (scene, ident)
            if frame_key in frame_contracts and frame_contracts[frame_key] != contract:
                raise ValueError("Same frame has inconsistent geometry/condition inputs across pairs")
            frame_contracts[frame_key] = contract
    result.pop("manifest_hash", None)
    result["manifest_hash"] = content_hash(result)
    return result


class SDEEventPairDataset(Dataset):
    """不扫描或加载 normal/；独立评估/标定程序通过另一份清单读取 GT。"""
    def __init__(self, manifest_path, data_root=None, args=None, validate_files=True,
                 require_measurement=True):
        self.manifest_path = Path(manifest_path)
        self.root = Path(data_root) if data_root is not None else self.manifest_path.parent
        with self.manifest_path.open(encoding="utf-8") as handle:
            source = json.load(handle)
        self.require_measurement = bool(require_measurement)
        self.manifest = validate_manifest(source, self.root, check_files=validate_files,
                                          require_measurement=self.require_measurement)
        self.settings = self.manifest["settings"]
        self.pairs = self.manifest["pairs"]
        self.prompt = getattr(args, "fixed_prompt", "high quality, bright, detailed photo")
        # 传入的 checkpoint 条件参数必须与 manifest 一致，不能静默覆盖。
        if args is not None:
            for key in ("image_size", "event_bins", "clora_event_active_bins", "rgb_event_active_bins"):
                value = getattr(args, key, None)
                if value is not None and int(value) != self.settings[key]:
                    raise ValueError(f"Checkpoint/manifest condition preprocessing mismatch: {key}")
            if getattr(args, "event_channels", self.settings["event_bins"]) != self.settings["event_bins"]:
                raise ValueError("event_channels must equal manifest event_bins")

    def __len__(self):
        return len(self.pairs)

    def _frame(self, pair, role):
        current = role == "current"
        path, ident = (pair["low_curr"], pair["curr_id"]) if current else (pair["low_prev"], pair["prev_id"])
        if ident is None:
            return None
        rgb = load_rgb(resolve_path(self.root, path), pair["geometry"])
        spec = pair["condition_event_sources"][role]
        event, refinement = condition_inputs(spec, self.root, self.settings, pair["geometry"])
        cache_key = content_hash(dict(scene=pair["scene_id"], frame=ident, path=str(resolve_path(self.root, path).resolve()),
                                      geometry=pair["geometry"], condition=spec, settings=self.settings,
                                      prompt=self.prompt))
        return dict(conditioning_pixel_values=rgb*2-1, events=event, events_refinement=refinement,
                    prompt=self.prompt, frame_id=ident, low_path=str(resolve_path(self.root, path)),
                    baseline_cache_key=cache_key)

    def __getitem__(self, index):
        pair = self.pairs[index]
        skip = pair.get("skip_reason", "")
        if not skip and not self.require_measurement:
            skip = "guidance_disabled"
        measurement = None
        if not skip:
            interval = (pair["t_prev"]+self.settings["clock_offset"], pair["t_curr"]+self.settings["clock_offset"])
            events = load_event_sources(pair["measurement_event_sources"], self.root, self.settings, interval)
            measurement = count_interval(events, pair["geometry"])
            if not bool((measurement["positive"]+measurement["negative"]).sum()):
                skip = "empty_event_interval"
        metadata = dict(pair)
        metadata["settings"] = dict(self.settings)
        metadata["manifest_hash"] = self.manifest["manifest_hash"]
        metadata["pair_key"] = f"{pair['scene_id']}/{pair['curr_id']}"
        return dict(previous=self._frame(pair, "previous"), current=self._frame(pair, "current"),
                    measurement=measurement, metadata=metadata, skip_reason=skip)
