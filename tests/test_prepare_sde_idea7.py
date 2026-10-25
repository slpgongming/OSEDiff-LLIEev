"""SDE 自动准备工具的对齐、测量、隔离和完整标定回归测试。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from dataloaders.sde_event_pair_dataset import (
    SDEEventPairDataset, _read_raw_events_cached, load_event_sources, read_raw_events,
)
from models.event_observation_proxy import EventObservationProxy
from scripts.prepare_sde_idea7 import (
    build_parser, check_test_support, main, partition_scenes, prepare, scan_scene,
)


class PrepareSDETest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data = self.root / "SDE"
        self.output = self.root / "prepared"
        self.settings = dict(timestamp_unit="us", timestamp_anchor="center", clock_offset=0,
                             polarity_encoding="zero_one", spatial_mapping="identity", image_size=256,
                             event_bins=5, clora_event_active_bins=5, rgb_event_active_bins=5)
        self.group = dict(observation_group="sde_default", observation_conditions=dict(sensor="synthetic"))
        torch.set_num_threads(1)

    def tearDown(self):
        _read_raw_events_cached.cache_clear()
        self.directory.cleanup()

    def make_scene(self, split, name, dt=100, short_start=False, fragments=True):
        scene = self.data / split / name
        (scene / "low").mkdir(parents=True)
        (scene / "normal").mkdir()
        times = [10**15 + 100 + dt * i for i in range(6)]
        levels = [40, 80, 40, 80, 40, 80]
        # 每个区间每像素 +/-14 个事件，与正常光的对数亮度翻倍/减半对应。
        rows = [[times[0] - 50, 0, 0, 1]]
        for index in range(1, len(times)):
            for k in range(1, 15):
                t = times[index - 1] + max(1, dt * k // 14)
                rows.extend([t, x, y, int(levels[index] > levels[index - 1])]
                            for y in range(6) for x in range(8))
        rows.append([times[-1] + 50, 0, 0, 1])
        raw = np.array(rows, dtype=np.int64)
        if short_start:
            raw = raw[raw[:, 0] > times[0] + dt // 2]
        structured = np.zeros(len(raw), dtype=[("timestamp", "<i8"), ("x", "<i2"),
                                                ("y", "<i2"), ("polarity", "i1")])
        for index, field in enumerate(structured.dtype.names):
            structured[field] = raw[:, index]
        np.savez(scene / "low" / "lowlight_event.npz", structured)
        for index, timestamp in enumerate(times):
            Image.fromarray(np.full((6, 8, 3), levels[index] // 4, dtype=np.uint8)).save(scene / "low" / f"{timestamp}.png")
            Image.fromarray(np.full((6, 8, 3), levels[index], dtype=np.uint8)).save(scene / "normal" / f"{timestamp + 5000}.png")
            if fragments:
                start = raw[0, 0] - 1 if index == 0 else times[index - 1]
                end = raw[-1, 0] + 1 if index == len(times) - 1 else times[index + 1]
                selected = raw[(raw[:, 0] > start) & (raw[:, 0] < end)]
                np.savez(scene / "low" / f"{timestamp}.npz", selected)
        return scene, times

    def args(self, *extra):
        return build_parser().parse_args(["--data-root", str(self.data), "--output-dir", str(self.output), *extra])

    def test_numeric_index_pairing_and_distinct_condition_measurement(self):
        scene, times = self.make_scene("test", "i_1")
        pairs, gt, frames, report = scan_scene(scene, "test", self.data, self.settings, self.group)
        self.assertEqual(gt[0]["normal_curr"], f"test/i_1/normal/{times[1] + 5000}.png")
        self.assertEqual(report["normal_minus_low_timestamp_range"], [5000, 5000])
        self.assertEqual(pairs[1]["t_prev"], times[0])
        self.assertEqual(pairs[1]["seed_indices"], dict(current=1, previous=0))
        self.assertEqual(pairs[1]["condition_event_sources"]["current"]["boundary"], "all")
        source = pairs[1]["measurement_event_sources"]
        self.assertTrue(source[0]["path"].endswith("lowlight_event.npz"))
        event = load_event_sources(source, self.data, self.settings, (times[0], times[1]))
        self.assertEqual(len(event), 14 * 6 * 8)
        self.assertEqual(int(event.t[-1]), times[1])
        self.assertTrue(np.all(event.t > times[0]))
        self.assertNotIn("normal_curr", pairs[1])
        self.assertEqual(len(frames), 6)

    def test_complete_cli_pipeline_calibrates_without_test_gt(self):
        self.make_scene("train", "i_a")
        self.make_scene("train", "i_b")
        self.make_scene("test", "i_test")
        config = self.root / "proxy.json"
        config.write_text(json.dumps(dict(linearization="identity", block_size=2)), encoding="utf-8")
        report, code = prepare(self.args("--validation-scenes", "i_b", "--proxy-config", str(config)))
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "ready")
        calibration = json.loads((self.output / "calibration.json").read_text(encoding="utf-8"))
        self.assertEqual(calibration["scenes"], dict(train=["i_a"], validation=["i_b"]))
        self.assertEqual(calibration["status"], "validated")
        pairs = json.loads((self.output / "pairs_test.json").read_text(encoding="utf-8"))
        EventObservationProxy(calibration).validate_manifest(pairs)
        self.assertNotIn("/normal/", json.dumps(pairs))
        gt = json.loads((self.output / "gt_trainval.json").read_text(encoding="utf-8"))
        self.assertNotIn("i_test", json.dumps(gt))
        sample = SDEEventPairDataset(self.output / "pairs_test.json", self.data)[1]
        self.assertEqual(float(sample["measurement"]["positive"].sum()), 14 * 6 * 6)
        shifted = copy.deepcopy(pairs)
        shifted["pairs"][1]["t_curr"] += 1
        self.assertFalse(check_test_support(calibration, shifted)["compatible"])

    def test_test_only_creates_no_calibration_and_separate_gt(self):
        self.make_scene("test", "i_test")
        with patch("scripts.prepare_sde_idea7.calibrate", side_effect=AssertionError("test GT must not fit")):
            report, code = prepare(self.args("--test-only"))
        self.assertEqual(code, 0)
        self.assertEqual(report["calibration_status"], "not_run")
        self.assertTrue((self.output / "evaluation_test.json").is_file())
        self.assertFalse((self.output / "calibration.json").exists())
        self.assertFalse((self.output / "gt_trainval.json").exists())

    def test_valid_calibration_with_unsupported_scene_keeps_full_and_supported_manifests(self):
        self.make_scene("train", "i_a")
        self.make_scene("train", "i_b")
        self.make_scene("test", "i_0_unsupported", dt=200)
        self.make_scene("test", "i_1_supported")
        config = self.root / "proxy.json"
        config.write_text(json.dumps(dict(linearization="identity", block_size=2)), encoding="utf-8")
        report, code = prepare(self.args("--validation-scenes", "i_b", "--proxy-config", str(config)))
        self.assertEqual(code, 3)
        self.assertEqual(report["calibration_status"], "validated")
        self.assertEqual(report["supported_test_scenes"], ["i_1_supported"])
        self.assertEqual(report["excluded_test_scenes"], ["i_0_unsupported"])
        full = json.loads((self.output / "pairs_test.json").read_text(encoding="utf-8"))
        subset = json.loads((self.output / "pairs_test_supported.json").read_text(encoding="utf-8"))
        self.assertEqual(len(full["pairs"]), 12)
        self.assertEqual(len(subset["pairs"]), 6)
        self.assertNotIn("i_0_unsupported", json.dumps(subset))
        self.assertEqual(subset["pairs"][0]["seed_indices"], full["pairs"][6]["seed_indices"])
        self.assertEqual(subset["pairs"][0]["seed_indices"]["current"], 6)
        calibration = json.loads((self.output / "calibration.json").read_text(encoding="utf-8"))
        EventObservationProxy(calibration).validate_manifest(subset)

    def test_scene_partition_is_deterministic_and_nonempty(self):
        scenes = [Path(f"i_{n}") for n in range(10)]
        split = partition_scenes(scenes)
        self.assertEqual(split, partition_scenes(list(reversed(scenes))))
        self.assertEqual(list(split.values()).count("validation"), 2)
        with self.assertRaisesRegex(ValueError, "two train scenes"):
            partition_scenes(scenes[:1])

    def test_scene_overlap_and_test_as_training_are_rejected(self):
        self.make_scene("test", "shared")
        self.make_scene("train", "shared")
        self.make_scene("train", "another")
        with self.assertRaisesRegex(ValueError, "Scene leakage"):
            prepare(self.args())
        with self.assertRaisesRegex(ValueError, "test directory"):
            prepare(self.args("--train-dir", "test"))

    def test_missing_or_partial_sources_fail_instead_of_silent_truncation(self):
        scene, times = self.make_scene("test", "i_1")
        (scene / "low" / f"{times[0]}.npz").unlink()
        with self.assertRaisesRegex(ValueError, "partial condition"):
            scan_scene(scene, "test", self.data, self.settings, self.group)
        (scene / "normal" / f"{times[0] + 5000}.png").unlink()
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            scan_scene(scene, "test", self.data, self.settings, self.group)

    def test_outside_stream_span_skips_incomplete_feedback(self):
        scene, _ = self.make_scene("test", "i_1", short_start=True)
        pairs, _, _, report = scan_scene(scene, "test", self.data, self.settings, self.group)
        self.assertEqual(pairs[1]["skip_reason"], "time_gap")
        self.assertEqual(pairs[1]["measurement_event_sources"], [])
        self.assertNotIn("skip_reason", pairs[2])
        self.assertEqual(report["skipped_intervals"][0]["reason"], "outside_observed_stream_span")

    def test_whole_stream_fallback_preserves_neighbor_condition_window(self):
        scene, times = self.make_scene("test", "i_1", fragments=False)
        pairs, _, _, _ = scan_scene(scene, "test", self.data, self.settings, self.group)
        spec = pairs[2]["condition_event_sources"]["current"]
        self.assertEqual(spec["boundary"], "open_open")
        self.assertEqual(spec["window"], [times[1], times[3]])

    def test_three_source_cache_retains_whole_stream_between_pairs(self):
        scene, times = self.make_scene("test", "i_1")
        whole = scene / "low" / "lowlight_event.npz"
        _read_raw_events_cached.cache_clear()
        original_load = np.load
        reads = []
        def tracked(path, *args, **kwargs):
            reads.append(Path(path).name)
            return original_load(path, *args, **kwargs)
        with patch("dataloaders.sde_event_pair_dataset.np.load", side_effect=tracked):
            for index in (1, 2, 3):
                read_raw_events(whole, "zero_one")
                read_raw_events(scene / "low" / f"{times[index - 1]}.npz", "zero_one")
                read_raw_events(scene / "low" / f"{times[index]}.npz", "zero_one")
        self.assertEqual(reads.count("lowlight_event.npz"), 1)

    def test_failed_calibration_is_preserved_with_nonzero_exit(self):
        self.make_scene("train", "i_a")
        self.make_scene("train", "i_b")
        self.make_scene("test", "i_test")
        with patch("scripts.prepare_sde_idea7.calibrate", return_value=dict(status="failed")):
            report, code = prepare(self.args())
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "calibration_failed")
        self.assertNotIn("observation_calibration", report["inference"])

    def test_main_writes_actionable_failure_report(self):
        code = main(["--data-root", str(self.data), "--output-dir", str(self.output), "--test-only"])
        self.assertEqual(code, 2)
        report = json.loads((self.output / "preparation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("error", report)


if __name__ == "__main__":
    unittest.main()
