"""兼容新旧 Accelerate 的 SwanLab 接入；实验只在主进程初始化。"""
import importlib
import warnings
from pathlib import Path

from accelerate.tracking import GeneralTracker, on_main_process


def normalize_report_to(value):
    if value is None or str(value).lower() in {"none", "null", ""}:
        return None
    names = str(value).lower().split(",")
    result = []
    for name in names:
        name = name.strip()
        if name == "wandb":
            warnings.warn("--report_to wandb now selects SwanLab; use --report_to swanlab.", FutureWarning)
            name = "swanlab"
        # 不使用 Accelerate 的 all 自动发现，避免环境中旧 W&B 被再次启用。
        for item in (["swanlab", "tensorboard"] if name == "all" else [name]):
            if item not in result:
                result.append(item)
    return ",".join(result)


class SwanLabTracker(GeneralTracker):
    name = "swanlab"
    requires_logging_directory = False
    main_process_only = True

    def __init__(self, project, experiment_name=None, logdir=None):
        super().__init__()
        # 导入 SDK 不创建实验；缺依赖时在加载大模型之前报错。
        try:
            self.sdk = importlib.import_module("swanlab")
        except ImportError as exc:
            raise ImportError("SwanLab requires: python -m pip install swanlab") from exc
        self.project = project
        self.experiment_name = experiment_name
        self.logdir = str(logdir) if logdir is not None else None
        self.run = None

    @property
    def tracker(self):
        return self.run

    @on_main_process
    def start(self):
        # 新版 Accelerate 调用 start，旧版直接调用 store_init_configuration。
        # 统一延迟到配置可用时初始化，避免创建两次实验。
        pass

    @on_main_process
    def store_init_configuration(self, values):
        if self.run is None:
            self.run = self.sdk.init(project=self.project, experiment_name=self.experiment_name,
                                     config=values, logdir=self.logdir)
        else:
            self.sdk.config.update(values)

    @on_main_process
    def log(self, values, step=None, **kwargs):
        if self.run is None:
            self.store_init_configuration({})
        self.sdk.log(values, step=step, **kwargs)

    @on_main_process
    def finish(self):
        if self.run is not None:
            self.sdk.finish()
            self.run = None


def build_trackers(args):
    """保留 TensorBoard 等后端；SwanLab 不依赖 Accelerate 原生版本支持。"""
    report = normalize_report_to(args.report_to)
    if not report:
        return None
    result = []
    for name in report.split(","):
        if name == "swanlab":
            result.append(SwanLabTracker(
                args.tracker_project_name, getattr(args, "run_name", None),
                Path(args.output_dir, getattr(args, "logging_dir", "logs"), "swanlab"),
            ))
        else:
            result.append(name)
    return result
