"""标准库测试：不加载模型，核查范围选择、指标汇总和失败后的文件保留。"""

import csv
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "checkpoint_range", ROOT / "test_event_film_lora_checkpoint_range.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class CheckpointRangeTests(unittest.TestCase):
    def test_requested_range_and_endpoint(self):
        self.assertEqual(evaluation.checkpoint_steps(10, 100, 20),
                         [10000, 30000, 50000, 70000, 90000])
        self.assertEqual(evaluation.checkpoint_steps(10, 30, 10), [10000, 20000, 30000])
        for values in ((10, 15, 10), (20, 10, 10), (10, 20, 0), (-10, 20, 10)):
            with self.assertRaises(ValueError):
                evaluation.checkpoint_steps(*values)

    def test_missing_checkpoints_and_final_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "model_10000.pkl").touch()
            (checkpoint_dir / "model_final.pkl").touch()
            with self.assertRaisesRegex(ValueError, "model_20000.pkl"):
                evaluation.select_checkpoints(directory, [10000, 20000])
            selected = evaluation.select_checkpoints(directory, [10000, 20000], True)
            self.assertEqual([step for step, _ in selected], [10000])

    def test_single_mode_and_incomplete_metrics(self):
        args, forwarded, modes = evaluation.parse_args([
            "--training_result_dir", "run", "--start_k", "10", "--end_k", "20",
            "--output_dir", "metrics", "--ablation_modes", "correct",
            "--unet_lora_parts", "none",
        ])
        self.assertEqual(modes, ["correct"])
        self.assertEqual(forwarded, ["--unet_lora_parts", "none"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text("mode,count,psnr,ssim,lpips\ncorrect,2,25,0.8,0.2\n", encoding="utf-8")
            with self.assertRaises(KeyError):
                evaluation.read_metrics(path, "model.pkl", 10000, ["correct"])

    def test_incremental_summary_and_temporary_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            for step in (10000, 20000):
                (checkpoints / f"model_{step}.pkl").touch()
            output = root / "metrics"
            commands = []

            def fake_evaluator(command, check):
                commands.append(command)
                self.assertTrue(check)
                self.assertEqual(Path(command[1]).name, "test_event_film_lora_metrics.py")
                self.assertIn("--calculate_fid", command)
                self.assertNotIn("--save_debug_vis", command)
                destination = Path(command[command.index("--output_dir") + 1])
                with (destination / "metrics_event_film_ablation_overall.csv").open(
                        "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["mode", "count", *evaluation.METRICS])
                    writer.writeheader()
                    for mode in ("correct", "zero"):
                        writer.writerow(dict(mode=mode, count=3, psnr=25, ssim=.8, lpips=.2, fid=12))
                if len(commands) == 2:
                    raise subprocess.CalledProcessError(1, command)

            with patch.object(evaluation.subprocess, "run", side_effect=fake_evaluator):
                with self.assertRaises(subprocess.CalledProcessError):
                    evaluation.main([
                        "--training_result_dir", str(root), "--start_k", "10", "--end_k", "20",
                        "--output_dir", str(output), "--stage", "3",
                    ])
            self.assertEqual([path.name for path in output.iterdir()], ["checkpoint_metrics.csv"])
            with (output / "checkpoint_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["step"] for row in rows], ["10000", "10000"])
            self.assertEqual([row["mode"] for row in rows], ["correct", "zero"])
            self.assertEqual(set(rows[0]), set(evaluation.FIELDS))


if __name__ == "__main__":
    unittest.main()
