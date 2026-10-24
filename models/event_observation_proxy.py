"""冻结的 processed-RGB → 区间净事件统计代理，不是 RAW 相机似然。

先逐像素 log 亮度差再池化；a、sigma、mask 和权重在同一样本的候选之间固定。
仅候选 RGB 可求导，目标事件只出现在测量和固定可靠性判断中。
"""

import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from dataloaders.sde_event_pair_dataset import CONTRACT_KEYS, content_hash


DEFAULT_PROXY_CONFIG = dict(
    rgb_range="zero_one", linearization="srgb", gamma=2.2,
    luminance_weights=[0.2126, 0.7152, 0.0722], epsilon=1e-3,
    block_size=16, min_activity=1.0, min_valid_fraction=0.01,
    min_valid_blocks=1, min_pixels_per_block=1,
    saturation_low=0.005, saturation_high=0.995,
    sigma_floor=1e-3, huber_delta=1.0,
    count_units="events_per_pixel_per_interval",
    interval="left_open_right_closed",
    geometry_mode="native_center_crop_restore_bilinear",
    mask_rule="fixed_valid_active_reference_baseline_unsaturated",
    weight_rule="valid_pixel_count",
)


def _check_rgb(rgb):
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[1] != 3:
        raise ValueError("Proxy RGB must be [1,3,H,W], in [0,1]")
    if not torch.isfinite(rgb).all() or bool((rgb.detach() < -1e-5).any()) or bool((rgb.detach() > 1+1e-5).any()):
        raise ValueError("Proxy RGB range/nonfinite mismatch; decode to [0,1] before calling")


def _native(rgb, geometry):
    _check_rgb(rgb)
    size = geometry["image_size"]
    if tuple(rgb.shape[-2:]) != (size, size):
        raise ValueError("Proxy RGB does not match declared model image_size")
    height, width = geometry["crop"][2:]
    return F.interpolate(rgb.float(), size=(height, width), mode="bilinear", align_corners=False)


def log_luminance(rgb, config):
    """gamma/sRGB 只是假设并持久化的 RGB 代理变换，不宣称恢复真实光强。"""
    rgb = rgb.float()
    mode = config["linearization"]
    if mode == "srgb":
        rgb = torch.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055).pow(2.4))
    elif mode == "gamma":
        rgb = rgb.clamp_min(0).pow(config["gamma"])
    elif mode != "identity":
        raise ValueError("linearization must be identity/srgb/gamma")
    coeff = rgb.new_tensor(config["luminance_weights"]).view(1, 3, 1, 1)
    return ((rgb*coeff).sum(1, keepdim=True) + config["epsilon"]).log()


def block_sum(tensor, block_size):
    """右/下补零后求和，边界块用真实 mask 像素数归一化。"""
    height, width = tensor.shape[-2:]
    pad_h, pad_w = (-height) % block_size, (-width) % block_size
    padded = F.pad(tensor, (0, pad_w, 0, pad_h))
    return F.avg_pool2d(padded, block_size, stride=block_size) * (block_size**2)


def _as_map(tensor, device, shape):
    tensor = torch.as_tensor(tensor, device=device).detach()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != (1, 1, *shape):
        raise ValueError("Measurement must remain [1,Hcrop,Wcrop] or [1,1,Hcrop,Wcrop]")
    return tensor


class FixedObservationEnergy:
    """可调用的固定 energy；构建一次后只能替换 candidate RGB。"""
    def __init__(self, reference, baseline, measurement, metadata, config, group):
        self.config = copy.deepcopy(config)
        self.geometry = copy.deepcopy(metadata["geometry"])
        self.a = float(group["a"])
        self.sigma = float(group["sigma"])
        self.skip_reason = ""
        with torch.no_grad():
            reference_native = _native(reference.detach(), self.geometry)
            baseline_native = _native(baseline.detach(), self.geometry)
            shape = reference_native.shape[-2:]
            positive = _as_map(measurement["positive"], baseline.device, shape).float()
            negative = _as_map(measurement["negative"], baseline.device, shape).float()
            valid = _as_map(measurement["valid_mask"], baseline.device, shape).bool()
            if not torch.isfinite(positive).all() or not torch.isfinite(negative).all() or (positive < 0).any() or (negative < 0).any():
                raise ValueError("Raw measurement counts must be finite and nonnegative")
            activity = positive+negative
            low, high = config["saturation_low"], config["saturation_high"]
            # 可靠性只由固定 reference/baseline 确定，不能随候选迭代移动。
            ref_ok = ((reference_native > low) & (reference_native < high)).all(1, keepdim=True)
            base_ok = ((baseline_native > low) & (baseline_native < high)).all(1, keepdim=True)
            self.mask = (valid & (activity >= config["min_activity"]) & ref_ok & base_ok).float().detach()
            self.reference_log = log_luminance(reference_native, config).detach()
            block = config["block_size"]
            self.pixel_count = block_sum(self.mask, block).detach()
            block_valid = self.pixel_count >= config["min_pixels_per_block"]
            self.weights = (self.pixel_count*block_valid).detach()
            self.target = (block_sum((positive-negative)*self.mask, block) / self.pixel_count.clamp_min(1)).detach()
            self.weight_sum = self.weights.sum().detach()
            fraction = float(self.mask.mean())
            blocks = int(block_valid.sum())
            if not bool(activity.sum()):
                self.skip_reason = "empty_event_interval"
            elif fraction < config["min_valid_fraction"] or blocks < config["min_valid_blocks"] or not bool(self.weight_sum):
                self.skip_reason = "insufficient_observation_support"
            self.diagnostics = dict(valid_fraction=fraction, valid_blocks=blocks,
                                    event_activity=float(activity.sum()),
                                    positive_count=float(positive.sum()), negative_count=float(negative.sum()),
                                    saturated_fraction=float((~(ref_ok & base_ok)).float().mean()),
                                    a=self.a, sigma=self.sigma, count_units=config["count_units"])

    def features(self, candidate):
        current = log_luminance(_native(candidate, self.geometry), self.config)
        # 注意顺序：逐像素 log 差 → 固定 mask → 块求和/有效面积。
        difference = current-self.reference_log
        return block_sum(difference*self.mask, self.config["block_size"]) / self.pixel_count.clamp_min(1)

    def prediction(self, candidate):
        return self.a*self.features(candidate)

    def __call__(self, candidate):
        if self.skip_reason:
            # runner 应读取 skip_reason 并跳过；这里仍返回连通图的有限零值。
            return candidate.float().sum()*0
        normalized = (self.prediction(candidate)-self.target)/self.sigma
        loss = F.huber_loss(normalized, torch.zeros_like(normalized), reduction="none",
                            delta=self.config["huber_delta"])
        return (loss*self.weights).sum()/self.weight_sum


def validate_proxy_config(config):
    unknown = set(config)-set(DEFAULT_PROXY_CONFIG)
    if unknown:
        raise ValueError(f"Unknown observation proxy config fields: {sorted(unknown)}")
    missing = set(DEFAULT_PROXY_CONFIG)-set(config)
    if missing:
        raise ValueError(f"Missing observation proxy config fields: {sorted(missing)}")
    fixed = ("rgb_range", "count_units", "interval", "geometry_mode", "mask_rule", "weight_rule")
    for key in fixed:
        if config[key] != DEFAULT_PROXY_CONFIG[key]:
            raise ValueError(f"Unsupported observation convention: {key}")
    if config["linearization"] not in {"identity", "srgb", "gamma"}:
        raise ValueError("Unknown linearization")
    for key in ("gamma", "epsilon", "sigma_floor", "huber_delta", "min_activity"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Proxy {key} must be finite positive")
    for key in ("block_size", "min_valid_blocks", "min_pixels_per_block"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"Proxy {key} must be a positive integer")
    coeff = config["luminance_weights"]
    if len(coeff) != 3 or any(not math.isfinite(v) or v <= 0 for v in coeff) or abs(sum(coeff)-1) > 1e-5:
        raise ValueError("Luminance weights must be three positive values summing to one")
    if not 0 <= config["saturation_low"] < config["saturation_high"] <= 1:
        raise ValueError("Invalid fixed saturation limits")
    if not 0 <= config["min_valid_fraction"] <= 1:
        raise ValueError("min_valid_fraction must lie in [0,1]")


class EventObservationProxy:
    def __init__(self, calibration):
        self.calibration = copy.deepcopy(calibration)
        if calibration.get("schema_version") != 1 or calibration.get("status") != "validated":
            raise ValueError("Observation calibration must be schema_version=1 and status=validated")
        self.config = calibration.get("proxy", {})
        validate_proxy_config(self.config)
        if not calibration.get("groups") or not calibration.get("contract"):
            raise ValueError("Calibration requires fixed groups and acquisition contract")
        for name, group in calibration["groups"].items():
            if group.get("status") != "validated":
                raise ValueError(f"Observation group is not validated: {name}")
            for key in ("a", "sigma"):
                if not math.isfinite(group.get(key, float("nan"))) or group[key] <= 0:
                    raise ValueError(f"Invalid calibrated {key}: {name}")
            if group["sigma"] < self.config["sigma_floor"]:
                raise ValueError("Calibration sigma smaller than declared floor")
            for key in ("conditions", "dt_range", "sensor_sizes", "validated_supports", "validation"):
                if key not in group:
                    raise ValueError(f"Missing calibration group metadata: {key}")
            if group["validation"].get("passed") is not True:
                raise ValueError("Calibration group has no successful held-out validation")
            duration = group["dt_range"]
            if len(duration) != 2 or not all(isinstance(v, int) and v > 0 for v in duration) or duration[0] > duration[1]:
                raise ValueError("Invalid calibrated dt_range")
            supports = group["validated_supports"]
            if not supports or any(set(s) != {"sensor_height", "sensor_width", "dt"}
                                   or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in s.values())
                                   for s in supports):
                raise ValueError("Calibration requires exact validated sensor-size/dt combinations")
            # JSON 中尺寸为列表；统一成 tuple 核对，防止报告范围与支持清单矛盾。
            if sorted(map(tuple, group["sensor_sizes"])) != sorted({(s["sensor_height"], s["sensor_width"]) for s in supports}):
                raise ValueError("sensor_sizes differs from validated_supports")
            durations = [s["dt"] for s in supports]
            if duration != [min(durations), max(durations)]:
                raise ValueError("dt_range differs from validated_supports")
        self.fingerprint = content_hash(calibration)

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def validate_metadata(self, metadata):
        settings = metadata["settings"]
        contract = self.calibration["contract"]
        for key in CONTRACT_KEYS:
            if settings.get(key) != contract.get(key):
                raise ValueError(f"Observation calibration/manifest mismatch: {key}")
        group_id = metadata["observation_group"]
        if group_id not in self.calibration["groups"]:
            raise ValueError(f"Unknown calibrated observation_group: {group_id}")
        group = self.calibration["groups"][group_id]
        if metadata["observation_conditions"] != group["conditions"]:
            raise ValueError(f"Exposure/ISP acquisition conditions mismatch: {group_id}")
        geometry = metadata["geometry"]
        if [geometry["sensor_height"], geometry["sensor_width"]] not in group["sensor_sizes"]:
            raise ValueError("Sensor geometry has not been validated for this observation group")
        if metadata["t_prev"] is not None and not metadata.get("skip_reason"):
            dt = metadata["t_curr"]-metadata["t_prev"]
            support = dict(sensor_height=geometry["sensor_height"], sensor_width=geometry["sensor_width"], dt=dt)
            if support not in group["validated_supports"]:
                raise ValueError("Sensor-size/dt combination has no held-out validated support")
        return group

    def validate_manifest(self, manifest):
        for pair in manifest["pairs"]:
            self.validate_metadata(dict(pair, settings=manifest["settings"]))

    def prepare_energy(self, reference_rgb, baseline_rgb, measurement, metadata):
        group = self.validate_metadata(metadata)
        if measurement is None:
            raise ValueError("Observation measurement is required; first frames must skip DPS in caller")
        return FixedObservationEnergy(reference_rgb, baseline_rgb, measurement, metadata,
                                      self.config, group)
