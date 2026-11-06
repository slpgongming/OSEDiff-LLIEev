"""用模拟 SDK 验证真实 Accelerate 调用，不连接云端或读取登录凭据。"""
import sys
import types
import unittest
from unittest.mock import Mock, patch

from accelerate import Accelerator, PartialState
from accelerate.utils import DistributedType
from my_utils.experiment_tracking import build_trackers, normalize_report_to


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.sdk = types.ModuleType("swanlab")
        self.sdk.init = Mock(return_value=object())
        self.sdk.log = Mock()
        self.sdk.finish = Mock()
        self.sdk.config = {}
        self.args = types.SimpleNamespace(report_to="swanlab", tracker_project_name="test-project",
                                         run_name="test-run", output_dir="output", logging_dir="logs")

    def test_accelerate_lifecycle(self):
        with patch.dict(sys.modules, {"swanlab": self.sdk}):
            trackers = build_trackers(self.args)
            self.sdk.init.assert_not_called()
            accelerator = Accelerator(cpu=True, log_with=trackers)
            accelerator.init_trackers("test-project", config={"lr": 0.001})
            self.sdk.init.assert_called_once()
            kwargs = self.sdk.init.call_args.kwargs
            self.assertEqual(kwargs["project"], "test-project")
            self.assertEqual(kwargs["experiment_name"], "test-run")
            self.assertEqual(kwargs["config"], {"lr": 0.001})
            accelerator.log({"loss/mod": 0.3}, step=12)
            self.sdk.log.assert_called_once_with({"loss/mod": 0.3}, step=12)
            accelerator.end_training()
            self.sdk.finish.assert_called_once()

    def test_non_main_rank_does_not_create_or_log(self):
        Accelerator(cpu=True)
        with patch.dict(sys.modules, {"swanlab": self.sdk}):
            tracker = build_trackers(self.args)[0]
            # 同时模拟分布式状态；仅修改 rank 时 Accelerate 仍视为单进程。
            with patch.multiple(PartialState(), process_index=1, num_processes=2,
                                distributed_type=DistributedType.MULTI_CPU):
                tracker.start()
                tracker.store_init_configuration({})
                tracker.log({"loss": 1}, step=1)
                tracker.finish()
            self.sdk.init.assert_not_called()
            self.sdk.log.assert_not_called()
            self.sdk.finish.assert_not_called()

    def test_backend_selection_and_legacy_alias(self):
        self.assertIsNone(normalize_report_to("none"))
        self.assertEqual(normalize_report_to("all"), "swanlab,tensorboard")
        with self.assertWarns(FutureWarning):
            self.assertEqual(normalize_report_to("wandb,swanlab"), "swanlab")
        self.args.report_to = "tensorboard"
        self.assertEqual(build_trackers(self.args), ["tensorboard"])
        self.args.report_to = None
        self.assertIsNone(build_trackers(self.args))

    def test_missing_sdk_error(self):
        with patch("my_utils.experiment_tracking.importlib.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(ImportError, "pip install swanlab"):
                build_trackers(self.args)


if __name__ == "__main__":
    unittest.main()
