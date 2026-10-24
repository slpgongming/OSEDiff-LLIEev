"""CPU 合成契约测试：不使用真实数据/权重，不把通过测试当作研究收益。"""

import copy
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from calibrate_event_observation import calibrate, check_proxy_gradient, fit_positive_scale, validate_group
from dataloaders.sde_event_pair_dataset import (
    CONTRACT_KEYS, SDEEventPairDataset, center_geometry, condition_inputs, count_interval,
    load_event_sources, load_rgb, read_raw_events, validate_manifest,
)
from models.event_observation_proxy import (
    DEFAULT_PROXY_CONFIG, EventObservationProxy, FixedObservationEnergy, log_luminance,
)
from scripts.build_event_pair_manifest import build_manifest


class EventMeasurementTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.settings = dict(timestamp_unit="us", timestamp_anchor="center", clock_offset=0,
                             polarity_encoding="zero_one", spatial_mapping="identity", image_size=4,
                             event_bins=5, clora_event_active_bins=5, rgb_event_active_bins=5)
        self.geometry = center_geometry(8, 6, 4)
        for name in ("prev", "curr"):
            Image.fromarray(np.full((6, 8, 3), 90, dtype=np.uint8)).save(self.root/f"{name}.png")
        self.save_events("events.npz", [10, 11, 12, 20, 21], [2]*5, [2]*5, [1, 1, 0, 1, 1])
        source = dict(path="events.npz", coverage=[0, 30])
        condition = dict(sources=[source], boundary="all")
        self.pair = dict(scene_id="scene", prev_id="prev", curr_id="curr", low_prev="prev.png", low_curr="curr.png",
                         t_prev=10, t_curr=20, split="test", observation_group="A",
                         observation_conditions=dict(exposure="fixed", isp="fixed"),
                         condition_event_sources=dict(previous=condition, current=condition),
                         measurement_event_sources=[source])
        self.manifest = dict(schema_version=1, settings=self.settings, pairs=[self.pair])
        self.manifest_path = self.write_json("pairs.json", self.manifest)
        self.config = copy.deepcopy(DEFAULT_PROXY_CONFIG)
        self.config.update(linearization="identity", block_size=2, min_valid_fraction=0.)

    def tearDown(self):
        self.directory.cleanup()

    def write_json(self, name, value):
        path = self.root/name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def save_events(self, name, t, x, y, p):
        np.savez(self.root/name, timestamp=np.asarray(t, dtype=np.int64), x=np.asarray(x),
                 y=np.asarray(y), polarity=np.asarray(p))

    def metadata(self):
        return dict(self.pair, settings=self.settings, geometry=self.geometry)

    def measurement(self, count=1.):
        return dict(positive=torch.full((1, 6, 6), count), negative=torch.zeros(1, 6, 6),
                    valid_mask=torch.ones(1, 6, 6, dtype=torch.bool))

    def calibration(self):
        return dict(schema_version=1, status="validated", proxy=self.config,
                    contract={k: self.settings[k] for k in CONTRACT_KEYS},
                    groups=dict(A=dict(status="validated", a=2., sigma=1.,
                                       conditions=self.pair["observation_conditions"], dt_range=[10, 10],
                                       sensor_sizes=[[6, 8]], validated_supports=[dict(sensor_height=6, sensor_width=8, dt=10)],
                                       validation=dict(passed=True))))

    def test_left_open_right_closed_and_adjacent_no_duplicates(self):
        source = self.pair["measurement_event_sources"]
        first = load_event_sources(source, self.root, self.settings, (10, 20))
        second = load_event_sources(source, self.root, self.settings, (20, 30))
        self.assertEqual(first.t.tolist(), [11, 12, 20])
        self.assertEqual(second.t.tolist(), [21])

    def test_large_timestamp_int64_resolution(self):
        base = 10**15
        self.save_events("large.npz", [base, base+1, base+2], [2]*3, [2]*3, [1]*3)
        raw = load_event_sources([dict(path="large.npz", coverage=[base-1, base+3])], self.root,
                                 self.settings, (base, base+1))
        self.assertEqual(raw.t.tolist(), [base+1])

    def test_float32_absolute_timestamps_rejected(self):
        np.savez(self.root/"bad.npz", timestamp=np.asarray([10**15], dtype=np.float32),
                 x=[2], y=[2], polarity=[1])
        with self.assertRaisesRegex(ValueError, "precision"):
            read_raw_events(self.root/"bad.npz", "zero_one")

    def test_whole_and_split_stream_counts_match(self):
        self.save_events("a.npz", [10, 11, 12], [2]*3, [2]*3, [1, 1, 0])
        self.save_events("b.npz", [20, 21], [2]*2, [2]*2, [1, 1])
        split = [dict(path="a.npz", coverage=[0, 15]), dict(path="b.npz", coverage=[15, 30])]
        whole = load_event_sources(self.pair["measurement_event_sources"], self.root, self.settings, (10, 20))
        divided = load_event_sources(split, self.root, self.settings, (10, 20))
        self.assertEqual(whole.t.tolist(), divided.t.tolist())
        torch.testing.assert_close(count_interval(whole, self.geometry)["positive"],
                                   count_interval(divided, self.geometry)["positive"])

    def test_right_source_owns_boundary_event_without_loss(self):
        self.save_events("whole.npz", [11, 15, 20], [2]*3, [2]*3, [1]*3)
        self.save_events("left.npz", [11], [2], [2], [1])
        self.save_events("right.npz", [15, 20], [2]*2, [2]*2, [1]*2)
        whole = load_event_sources([dict(path="whole.npz", coverage=[0, 30])], self.root, self.settings, (10, 20))
        split = load_event_sources([dict(path="left.npz", coverage=[0, 15]),
                                   dict(path="right.npz", coverage=[15, 30])], self.root, self.settings, (10, 20))
        self.assertEqual(whole.t.tolist(), split.t.tolist())
        self.assertEqual(split.t.tolist(), [11, 15, 20])

    def test_both_sources_contain_boundary_events_is_ambiguous(self):
        self.save_events("left.npz", [11, 15], [2]*2, [2]*2, [1]*2)
        self.save_events("right.npz", [15, 20], [3]*2, [2]*2, [1]*2)
        with self.assertRaisesRegex(ValueError, "boundary ownership"):
            load_event_sources([dict(path="left.npz", coverage=[0, 15]),
                                dict(path="right.npz", coverage=[15, 30])], self.root, self.settings, (10, 20))

    def test_polarity_cancellation_keeps_activity(self):
        raw = load_event_sources(self.pair["measurement_event_sources"], self.root, self.settings, (10, 12))
        measured = count_interval(raw, self.geometry)
        self.assertEqual(float((measured["positive"]-measured["negative"]).sum()), 0.)
        self.assertEqual(float((measured["positive"]+measured["negative"]).sum()), 2.)

    def test_empty_is_valid_but_missing_file_fails(self):
        self.save_events("empty.npz", [], [], [], [])
        self.assertEqual(len(load_event_sources([dict(path="empty.npz", coverage=[0, 30])], self.root,
                                                self.settings, (10, 20))), 0)
        with self.assertRaises(FileNotFoundError):
            load_event_sources([dict(path="missing.npz", coverage=[0, 30])], self.root, self.settings, (10, 20))

    def test_coverage_gap_and_overlap_fail(self):
        for sources in ([dict(path="events.npz", coverage=[11, 30])],
                        [dict(path="events.npz", coverage=[0, 30]), dict(path="other.npz", coverage=[10, 40])]):
            with self.assertRaisesRegex(ValueError, "coverage"):
                load_event_sources(sources, self.root, self.settings, (10, 20))

    def test_voxel_cannot_be_measurement(self):
        np.savez(self.root/"voxel.npz", np.zeros((5, 6, 8), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "raw"):
            read_raw_events(self.root/"voxel.npz", "zero_one")

    def test_native_crop_count_is_not_resized(self):
        self.save_events("crop.npz", [11]*3, [0, 1, 7], [2]*3, [1]*3)
        raw = load_event_sources([dict(path="crop.npz", coverage=[0, 30])], self.root, self.settings, (10, 20))
        measured = count_interval(raw, self.geometry)
        self.assertEqual(tuple(measured["positive"].shape), (1, 6, 6))
        self.assertEqual(float(measured["positive"].sum()), 1.)

    def test_dataset_no_gt_and_condition_settings_do_not_change_measurement(self):
        first = SDEEventPairDataset(self.manifest_path, self.root)[0]
        self.assertFalse((self.root/"normal").exists())
        changed = copy.deepcopy(self.manifest)
        changed["settings"]["clora_event_active_bins"] = 1
        second = SDEEventPairDataset(self.write_json("changed.json", changed), self.root)[0]
        torch.testing.assert_close(first["measurement"]["positive"], second["measurement"]["positive"])
        self.assertEqual(tuple(first["current"]["events"].shape), (5, 4, 4))
        self.assertEqual(tuple(first["current"]["conditioning_pixel_values"].shape), (3, 4, 4))

    def test_condition_open_open_and_all_preserve_old_window_rules(self):
        # 选择缩放后仍被 nearest 采到的位置，避免本测试退化成两份全零 voxel。
        self.save_events("events.npz", [10, 11, 12, 20, 21], [2]*5, [1]*5, [1, 1, 0, 1, 1])
        all_inputs, _ = condition_inputs(self.pair["condition_event_sources"]["current"], self.root, self.settings, self.geometry)
        spec = copy.deepcopy(self.pair["condition_event_sources"]["current"])
        spec.update(boundary="open_open", window=[10, 20])
        sliced, _ = condition_inputs(spec, self.root, self.settings, self.geometry)
        self.assertFalse(torch.equal(all_inputs, sliced))

    def test_condition_preprocessing_matches_legacy_dataset_methods(self):
        # 抽取仓库旧方法本体，避免为这项纯预处理回归强制导入 torchvision。
        path = Path(__file__).resolve().parents[1]/"dataloaders/sde_event_film_dataset.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        old_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SDEEventFilmDataset")
        old_class.bases = []
        names = {"_crop_event", "_generate_voxel_grid", "_normalize_event_voxel", "_normalize_event_voxel_refinement"}
        old_class.body = [n for n in old_class.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=[old_class], type_ignores=[]), str(path), "exec"), namespace)
        legacy = namespace["SDEEventFilmDataset"]()
        legacy.event_bins = legacy.event_channels = 5
        self.save_events("events.npz", [10, 11, 12, 20, 21], [2]*5, [1]*5, [1, 1, 0, 1, 1])
        raw = read_raw_events(self.root/"events.npz", "zero_one")
        stacked = torch.from_numpy(np.stack([raw.t, raw.x, raw.y, raw.p], axis=1))
        voxel = legacy._crop_event(stacked, *self.geometry["crop"])
        voxel = F.interpolate(voxel.unsqueeze(0), size=(4, 4), mode="nearest")[0]
        actual, refinement = condition_inputs(self.pair["condition_event_sources"]["current"], self.root, self.settings, self.geometry)
        torch.testing.assert_close(actual, legacy._normalize_event_voxel(voxel))
        torch.testing.assert_close(refinement, legacy._normalize_event_voxel_refinement(voxel))

    def test_disabled_measurement_accepts_no_raw_measurement(self):
        self.manifest["pairs"][0]["measurement_event_sources"] = [dict(path="missing.npz", coverage=[0, 30])]
        path = self.write_json("disabled.json", self.manifest)
        sample = SDEEventPairDataset(path, self.root, require_measurement=False)[0]
        self.assertIsNone(sample["measurement"])
        self.assertEqual(sample["skip_reason"], "guidance_disabled")

    def test_first_frame_needs_no_reference(self):
        pair = self.manifest["pairs"][0]
        pair.update(prev_id=None, low_prev=None, t_prev=None, measurement_event_sources=[])
        pair["condition_event_sources"]["previous"] = None
        sample = SDEEventPairDataset(self.write_json("first.json", self.manifest), self.root)[0]
        self.assertIsNone(sample["previous"])
        self.assertEqual(sample["skip_reason"], "first_frame")

    def test_gt_in_inference_manifest_rejected(self):
        self.manifest["pairs"][0]["normal_curr"] = "forbidden.png"
        with self.assertRaisesRegex(ValueError, "GT paths"):
            validate_manifest(self.manifest, self.root)

    def test_scene_split_leakage_rejected(self):
        duplicate = copy.deepcopy(self.pair)
        duplicate.update(curr_id="other", split="train")
        self.manifest["pairs"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Scene leakage"):
            validate_manifest(self.manifest, self.root)

    def test_cli_config_conflict_rejected(self):
        with self.assertRaisesRegex(ValueError, "preprocessing mismatch"):
            SDEEventPairDataset(self.manifest_path, self.root, SimpleNamespace(event_bins=4))

    def test_preflight_writes_hash_and_fingerprints(self):
        result = build_manifest(self.manifest_path, self.root, self.root/"built.json")
        self.assertIn("events.npz", result["file_fingerprints"])
        reloaded = SDEEventPairDataset(self.root/"built.json", self.root).manifest
        self.assertEqual(result["manifest_hash"], reloaded["manifest_hash"])

    def test_log_before_pool_and_fixed_mask(self):
        geometry = center_geometry(4, 4, 4)
        metadata = dict(self.metadata(), geometry=geometry)
        measurement = {key: value[..., :4, :4] for key, value in self.measurement().items()}
        reference = torch.full((1, 3, 4, 4), 0.2)
        current = reference.clone()
        current[:, :, :, ::2] = 0.1
        current[:, :, :, 1::2] = 0.7
        energy = FixedObservationEnergy(reference, current, measurement, metadata, self.config, dict(a=1., sigma=1.))
        expected = (torch.log(torch.tensor(0.1)+self.config["epsilon"])+torch.log(torch.tensor(0.7)+self.config["epsilon"]))/2-torch.log(torch.tensor(0.2)+self.config["epsilon"])
        torch.testing.assert_close(energy.features(current), torch.full((1, 1, 2, 2), float(expected)))
        wrong = torch.log(torch.tensor(0.4)+self.config["epsilon"])-torch.log(torch.tensor(0.2)+self.config["epsilon"])
        self.assertGreater(abs(float(expected-wrong)), 0.1)
        saved_mask, saved_weights = energy.mask.clone(), energy.weights.clone()
        energy(torch.zeros_like(current))
        torch.testing.assert_close(energy.mask, saved_mask)
        torch.testing.assert_close(energy.weights, saved_weights)

    def test_proxy_gradient_is_finite_and_reference_detached(self):
        proxy = EventObservationProxy(self.calibration())
        reference = torch.full((1, 3, 4, 4), 0.2, requires_grad=True)
        baseline = torch.full((1, 3, 4, 4), 0.3)
        energy = proxy.prepare_energy(reference, baseline, self.measurement(), self.metadata())
        candidate = baseline.clone().requires_grad_(True)
        grad, = torch.autograd.grad(energy(candidate), candidate)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertLess(float(grad.mean()), 0.)
        self.assertFalse(energy.reference_log.requires_grad)
        self.assertTrue(check_proxy_gradient(self.config)["passed"])

    def test_proxy_skips_empty_or_saturated(self):
        proxy = EventObservationProxy(self.calibration())
        image = torch.full((1, 3, 4, 4), 0.3)
        empty = proxy.prepare_energy(image, image, self.measurement(0), self.metadata())
        self.assertEqual(empty.skip_reason, "empty_event_interval")
        saturated = proxy.prepare_energy(image, torch.ones_like(image), self.measurement(), self.metadata())
        self.assertEqual(saturated.skip_reason, "insufficient_observation_support")

    def test_proxy_rejects_wrong_range_conditions_units_and_unknown_group(self):
        proxy = EventObservationProxy(self.calibration())
        for mutation in ("group", "conditions", "unit", "duration"):
            metadata = copy.deepcopy(self.metadata())
            if mutation == "group": metadata["observation_group"] = "unknown"
            if mutation == "conditions": metadata["observation_conditions"]["isp"] = "different"
            if mutation == "unit": metadata["settings"]["timestamp_unit"] = "ns"
            if mutation == "duration": metadata["t_curr"] = 21
            with self.assertRaises(ValueError):
                proxy.validate_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "range"):
            proxy.prepare_energy(torch.full((1, 3, 4, 4), -1.), torch.zeros(1, 3, 4, 4), self.measurement(), self.metadata())

    def test_calibration_failure_is_not_loadable(self):
        bad = self.calibration()
        bad["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "validated"):
            EventObservationProxy(bad)

    def test_runtime_rejects_unvalidated_cross_product_of_size_and_dt(self):
        calibration = self.calibration()
        group = calibration["groups"]["A"]
        group.update(dt_range=[10, 20], sensor_sizes=[[6, 8], [10, 12]],
                     validated_supports=[dict(sensor_height=6, sensor_width=8, dt=10),
                                         dict(sensor_height=10, sensor_width=12, dt=20)])
        proxy = EventObservationProxy(calibration)
        metadata = self.metadata()
        metadata["t_curr"] = 30
        with self.assertRaisesRegex(ValueError, "validated support"):
            proxy.validate_metadata(metadata)

    def test_robust_scale_fit(self):
        x = torch.linspace(-2, 2, 101)
        y = 3*x
        y[0] = 100.
        a, sigma = fit_positive_scale(x, y)
        self.assertAlmostEqual(a, 3., places=3)
        self.assertGreaterEqual(sigma, 1e-3)

    def test_zero_weight_blocks_do_not_shrink_calibrated_sigma(self):
        x = torch.tensor([1., 2., 3., 4., 5.])
        y = 2*x+torch.tensor([0.3, -0.4, 0.2, -0.3, 0.1])
        reference = fit_positive_scale(x, y)
        padded = fit_positive_scale(torch.cat([x, torch.zeros(1000)]),
                                    torch.cat([y, torch.zeros(1000)]),
                                    torch.cat([torch.ones(5), torch.zeros(1000)]))
        self.assertAlmostEqual(reference[0], padded[0])
        self.assertAlmostEqual(reference[1], padded[1])
        self.assertGreater(padded[1], 0.01)

    def test_wrong_time_control_rejects_disjoint_valid_blocks(self):
        records = [dict(x=torch.tensor([1., 0.]), y=torch.tensor([2., 0.]), w=torch.tensor([1., 0.]), activity=1.),
                   dict(x=torch.tensor([0., 2.]), y=torch.tensor([0., 4.]), w=torch.tensor([0., 1.]), activity=1.)]
        result = validate_group(records, 2., 0.05, 0.01)
        self.assertFalse(result["passed"])
        self.assertIn("common valid blocks", result["reason"])

    def make_calibration_data(self):
        pairs, gt = [], []
        for index, count in enumerate((1, 3, 2, 4)):
            scene = f"scene_{index}"
            previous_value = 64
            current_value = round(255*((previous_value/255+self.config["epsilon"])*np.exp(count/5)-self.config["epsilon"]))
            previous_name, current_name = f"{scene}_prev.png", f"{scene}_curr.png"
            for name, value in ((previous_name, previous_value), (current_name, current_value)):
                Image.fromarray(np.full((6, 8, 3), value, dtype=np.uint8)).save(self.root/name)
            timestamps, xs, ys, ps = [], [], [], []
            for tick in range(count):
                for y in range(6):
                    for x in range(8):
                        timestamps.append(11+tick); xs.append(x); ys.append(y); ps.append(1)
            event_name = f"{scene}.npz"
            self.save_events(event_name, timestamps, xs, ys, ps)
            source = dict(path=event_name, coverage=[0, 30])
            condition = dict(sources=[source], boundary="all")
            pair = copy.deepcopy(self.pair)
            pair.update(scene_id=scene, low_prev=previous_name, low_curr=current_name,
                        split="train" if index < 2 else "validation", measurement_event_sources=[source],
                        condition_event_sources=dict(previous=condition, current=condition))
            pairs.append(pair)
            gt.append(dict(scene_id=scene, curr_id="curr", normal_prev=previous_name, normal_curr=current_name))
        manifest = dict(schema_version=1, settings=self.settings, pairs=pairs)
        return self.write_json("calibration_pairs.json", manifest), self.write_json("gt.json", dict(schema_version=1, pairs=gt))

    def test_full_calibration_round_trip_and_heldout_controls(self):
        pairs, gt = self.make_calibration_data()
        output = self.root/"calibration.json"
        result = calibrate(pairs, gt, self.root, output, self.config, report_path=self.root/"report.md")
        self.assertEqual(result["status"], "validated", result)
        self.assertAlmostEqual(result["groups"]["A"]["a"], 5., delta=0.1)
        self.assertGreater(result["groups"]["A"]["validation"]["wrong_time_mae"], result["groups"]["A"]["validation"]["mae"])
        self.assertFalse(result["actual_model_quality_evaluated"])
        proxy = EventObservationProxy.load(output)
        dataset = SDEEventPairDataset(pairs, self.root)
        proxy.validate_manifest(dataset.manifest)

    def test_calibration_rejects_test_split(self):
        pairs, gt = self.make_calibration_data()
        payload = json.loads(pairs.read_text(encoding="utf-8"))
        payload["pairs"][-1]["split"] = "test"
        pairs.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "never test GT"):
            calibrate(pairs, gt, self.root, self.root/"forbidden.json", self.config)

    def test_train_only_and_skipped_ranges_are_not_marked_validated(self):
        pairs, gt = self.make_calibration_data()
        payload = json.loads(pairs.read_text(encoding="utf-8"))
        # 训练样本的新间隔/尺寸可用于拟合，但不能扩张未验证的部署支持范围。
        payload["pairs"][0]["t_curr"] = 30
        for key in ("low_prev", "low_curr"):
            path = self.root/payload["pairs"][0][key]
            with Image.open(path) as old:
                value = old.getpixel((0, 0))[0]
            Image.fromarray(np.full((8, 10, 3), value, dtype=np.uint8)).save(path)
        skipped = copy.deepcopy(payload["pairs"][1])
        skipped.update(scene_id="skipped_scene", t_curr=1000, split="validation", skip_reason="time_gap")
        payload["pairs"].append(skipped)
        pairs.write_text(json.dumps(payload), encoding="utf-8")
        result = calibrate(pairs, gt, self.root, self.root/"support.json", self.config)
        self.assertEqual(result["status"], "validated", result)
        group = result["groups"]["A"]
        self.assertEqual(group["dt_range"], [10, 10])
        self.assertEqual(group["sensor_sizes"], [[6, 8]])
        self.assertEqual(group["validated_supports"], [dict(sensor_height=6, sensor_width=8, dt=10)])
        proxy = EventObservationProxy(result)
        dataset = SDEEventPairDataset(pairs, self.root)
        with self.assertRaisesRegex(ValueError, "Sensor geometry"):
            proxy.validate_metadata(dataset[0]["metadata"])


if __name__ == "__main__":
    unittest.main()
