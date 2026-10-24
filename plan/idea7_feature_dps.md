# idea7：全通道 UNet 特征的类 DPS 推理纠错

实现基线：`idea4@3bcec8d4ee104d5655d83146de9735c4abd9a854`。实现位于开发分支 `idea7`。

## 方法与边界

保留已训练的 Event LoRA residual cross-attention。在倒数第二个完整 UNet up-block 输出处加入与特征同形状的修正量 `Δh=s0*u`，其中 `u` 为 FP32 `[1,C,H,W]` 临时叶变量，`s0` 是第一轮原始特征的固定 RMS。

固定模型权重、条件编码、前帧 baseline、原始区间事件、mask 和标定参数。候选图经 RGB→事件统计代理得到观测误差，梯度穿过 VAE、scheduler 和后段 UNet 更新 u；每次更新规范化梯度 RMS，再投影到相对 RMS 半径内。保留 baseline 和已评估候选中能量最低的有限结果；无改善或观测不足时回退。

这是 DPS-inspired feature refinement，不是原版 DPS 或后验采样。原 diffusion timestep 仍为一个，但 K 次修正需 K+1 次完整 UNet/VAE 前向及 K 次反传，另计条件编码和未缓存前帧。首版不缓存 UNet 前缀。允许 K=0/1/2、batch=1、256×256 方形输入、单注入点；默认 K=1、step_rel=0.01、radius_rel=0.05，这些是待验证的工程初值。

修正后的前帧不会递推：下一帧使用缓存的**未修正 baseline**。第一帧、声明的场景边界/时间缺口、合法无事件区间跳过纠错。模型参数、checkpoint 格式和原训练梯度语义保持不变。

## 代码入口

| 文件 | 责任 |
|---|---|
| `models/feature_dps_guidance.py` | 临时特征变量、更新/投影、候选选择、RNG 与清理 |
| `models/unet_2d_condition.py` | 完整 up-block 后的可选 `feature_guidance` |
| `osediff_event_film.py` | 单次预测、固定 context、可微 decode、缓存清理；已移除 ESG |
| `dataloaders/sde_event_pair_dataset.py` | 无 GT 的帧对输入、条件事件与原始测量两条通路 |
| `models/event_observation_proxy.py` | 固定 mask 的 processed RGB 净事件统计代理 |
| `scripts/build_event_pair_manifest.py` | 显式数据规格预检、规范化清单及文件指纹 |
| `calibrate_event_observation.py` | train 拟合、validation 验收、标定 JSON/报告 |
| `test_idea7_dps.py` | 预检、配置解析、顺序推理、baseline 缓存、统计和独立评估 |
| `test_scripts/test_idea7_dps.sh` | whb 启动脚本，保留环境变量覆盖与 dry-run |
| `test_event_baseline.py` | 从旧 ESG 入口迁出的单次 baseline 与历史 CSV 兼容 |

旧 `models/event_self_swap_guidance.py`、`test_idea4_esg.py`、对应 shell/测试已移除。7 个历史 wrapper 改调 baseline 入口，原账户路径、模型设置和图像 seed 不变。`legacy-baseline` 只保留旧 CSV 文件名/列，不保留 ESG 算法。新 `.gitattributes` 保证 shell 在 Windows checkout 后也使用 LF。

## 1. 准备显式配对清单

真实原始数据当前未在本机验证。工具不会从文件名大小猜时间单位，不会将“排序配对”当作已经证明事件覆盖。

```bash
python scripts/build_event_pair_manifest.py --write-example pair_spec.json
# 按实际数据修改 pair_spec.json 中的路径、时间、极性、覆盖和采集条件。
python scripts/build_event_pair_manifest.py \
  --spec pair_spec.json --data-root /实际数据根目录 \
  --output pairs.json --report preflight.json
```

主要字段如下；完整模板由上面命令生成。

```json
{
  "schema_version": 1,
  "settings": {
    "timestamp_unit": "us", "timestamp_anchor": "center", "clock_offset": 0,
    "polarity_encoding": "zero_one", "spatial_mapping": "identity", "image_size": 256,
    "event_bins": 5, "clora_event_active_bins": 5, "rgb_event_active_bins": 5
  },
  "pairs": [{
    "scene_id": "scene", "prev_id": "previous", "curr_id": "current",
    "low_prev": "scene/low/previous.png", "low_curr": "scene/low/current.png",
    "t_prev": 100, "t_curr": 200, "split": "test",
    "observation_group": "camera_isp_exposure_A",
    "observation_conditions": {"exposure": "填写实际曝光", "isp": "填写实际处理流程"},
    "condition_event_sources": {
      "previous": {"sources": [{"path": "scene/events.npz", "coverage": [0, 300]}], "window": [0, 200], "boundary": "open_open"},
      "current": {"sources": [{"path": "scene/events.npz", "coverage": [0, 300]}], "window": [100, 300], "boundary": "open_open"}
    },
    "measurement_event_sources": [{"path": "scene/events.npz", "coverage": [0, 300]}]
  }]
}
```

- 原始字段默认识别 `t/timestamp`、`x`、`y`、`p/polarity`；特殊名称通过 source 的 `keys` 显式映射。
- 测量严格取 `(t_prev+clock_offset, t_curr+clock_offset]`，保存 N+、N−；activity=N++N−，正负抵消不是无事件。绝对时间不能先转 float32。
- 来源 `coverage` 表示文件实际覆盖，必须真实填写。相接分片的边界事件保留；若两个相邻文件同时含共享边界时间，首版拒绝其所有权歧义，不自动去重或丢弃。
- 条件通路复用原 checkpoint 预处理，`boundary=all` 表示保留原分片全部事件；预体素只允许作为条件输入，需 `format=voxel`。反馈测量始终来自原始事件。
- 输入图像中心裁成正方形并缩放到256。测量计数留在原始裁剪网格；候选图可微插值回该网格后做粗块统计，不缩放/归一化真实计数。
- 第一帧将 `prev_id/low_prev/t_prev`、previous 条件设为 null，测量来源为空列表。首帧仍需当前条件事件。`time_gap/scene_boundary` 可显式作为 `skip_reason`。
- 每个场景只允许属于一个 split。同一帧跨 pair 的路径、条件和几何必须一致。清单中禁止正常光路径。
- 旧条件事件可能含当前帧之后的事件；本实现不据此声称严格因果/在线能力。

可选 `seed_indices: {"previous": 旧全局索引, "current": 旧全局索引}` 用于旧 baseline 对照。如果使用，所有有效帧都要填写，同一帧跨 pair 要一致。未提供时按 manifest 中帧首次出现顺序分配 `seed+index`；任意子集/重排 manifest 与旧 dataset 的 seed 不一定一致。

## 2. 标定观测代理

PNG/JPG 是处理后的 RGB。使用固定亮度变换后的逐像素 log 差，再聚合到同一 mask 的粗块：`prediction=a*mean(logY_current-logY_reference)`。目标是每像素每区间的净事件数，不是事件率。Huber 损失使用训练残差的固定稳健尺度 sigma；a>0 只是经验标定，不是精确相机阈值。

训练/验证使用另一个配对清单，仅含 train/validation 场景。GT 独立文件：

```json
{"schema_version": 1, "pairs": [
  {"scene_id": "scene", "curr_id": "current", "normal_prev": "scene/normal/previous.png", "normal_curr": "scene/normal/current.png"}
]}
```

```bash
python calibrate_event_observation.py \
  --pair-manifest pairs_trainval.json --gt-manifest gt_trainval.json \
  --data-root /实际标定数据根目录 \
  --output calibration.json --report calibration_report.md
```

默认代理明确采用 sRGB 逆变换近似、16像素粗块；可通过 `--proxy-config config.json` 声明 identity/sRGB/gamma、块大小、饱和阈值等设定。仅在 train/validation 选择这些配置，不使用 test GT 调参。

每个组至少需要两个有效 validation 帧对做错时对照。默认要求比零预测 MAE 改善至少5%，比错时反馈 MAE 改善至少1%；错时对照在双方有效块交集上比较，公共块少于当前有效块10%时失败。零权重块不进入 MAD 估计。

支持范围由实际有效 validation 记录确定：`validated_supports` 保存 `{sensor_height,sensor_width,dt}` 的明确组合。未验证尺寸/时距组合被推理拒绝，不把 train-only 或被跳过样本扩大成“已验证范围”。门槛按 observation_group 整体检验，支持清单表示验证样本覆盖，并非每个组合独立通过收益门槛。数据预检通过不等于代理标定通过，代理通过也不等于图像增强有效。

失败标定输出 `status=failed` 并返回退出码2；推理拒绝失败、缺字段或不匹配的配置。首版严格匹配时间间隔组合，存在真实采样抖动时需先按独立验证数据设计分组/容差，不能在测试时默许。

## 3. whb 推理

```bash
export CHECKPOINT_PATH=/实际训练权重/model.pkl
export PAIR_MANIFEST=/实际清单/pairs_test.json
export OBSERVATION_CALIBRATION=/实际标定/calibration.json
export OUTPUT_DIR=/实际输出/idea7_k1
# 若checkpoint训练为400~600区间，必须明确选择验证过的推理timestep。
export GENERATOR_TIMESTEP=500
# 旧checkpoint没有普通LoRA激活元数据时，也必须明确设置，例如none或down,mid,up。
# export UNET_LORA_PARTS=none

DRY_RUN=1 bash test_scripts/test_idea7_dps.sh
PREFLIGHT_ONLY=1 bash test_scripts/test_idea7_dps.sh
bash test_scripts/test_idea7_dps.sh
```

默认 whb SD 路径：`/public/home/whb/Chen_Project/PretrainedModel/stable-diffusion-2-1-base`；默认 indoor test 根：`/public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/test`。checkpoint、manifest、标定路径无虚构默认值。可用 `TEST_DATA_ROOT` 改为实际 outdoor 路径。

Stage2/3 自动从 checkpoint 推断是否有 decoder-refinement；Stage4 自动继承 variant、融合权重、CGDRS 配置。若 encoder 权重没有打包，需提供 `RGB_EVENT_PRETRAINED_ROOT`。Event LoRA 结构、普通 LoRA 激活组、active bins 和 timestep 做显式一致性检查；缺少唯一 timestep 时要求用户指定，不猜999。

```bash
# 同一模型配置的零修正基线；无需观测标定或反馈measurement文件。
DPS_STEPS=0 OUTPUT_DIR=/实际输出/idea7_k0 bash test_scripts/test_idea7_dps.sh
# 有明确预算的两次修正。
DPS_STEPS=2 OUTPUT_DIR=/实际输出/idea7_k2 bash test_scripts/test_idea7_dps.sh
# 只错置反馈计数；原条件事件保持正确。无可用donor时日志说明跳过。
FEEDBACK_MODE=shifted FEEDBACK_SHIFT=1 OUTPUT_DIR=/实际输出/idea7_shifted bash test_scripts/test_idea7_dps.sh
```

`DPS_STEP_REL` 默认正0.01、`DPS_RADIUS_REL` 默认正0.05、`DPS_UP_BLOCK=penultimate`、seed42、fp16。现有模型构造器使用 CUDA；CPU 支持预检/标定/单测，不支持正式 OSEDiff 权重推理。首版关闭 tiling 和 gradient checkpointing。多候选复用编码后的 RNG 状态，保持 VAE→scheduler 的随机序列，并恢复调用者 RNG。

## 输出与评估

- `images_baseline/`、`images_dps/`：零修正与最终候选 PNG。
- `resolved_config.json`：有效配置、源文件/checkpoint/manifest/标定指纹、seed规则、实际层索引。
- `per_image.csv`：选中轮次、观测能量、实际前向/反向数、参考缓存命中、修正范数、梯度、可靠性及耗时。
- `summary.json`、`scene_summary.json`：采用率、回退原因、平均/p95成本和显存；`last_sample.json`保留最后样本完整候选轨迹。
- `run.log`：shell捕获的运行日志。数据、条件编码、前帧生成与候选循环分别计时，端到端包含保存图像等开销。

正常光评估是生成结束后的独立步骤，通过 `--evaluation-manifest` 或 shell 的 `EVALUATION_MANIFEST` 指定：

```json
{"schema_version": 1, "frames": [
  {"scene_id": "scene", "frame_id": "current", "normal_path": "scene/normal/current.png"}
]}
```

输出 `evaluation.csv` 和 `evaluation_summary.json`，使用 Y 通道 PSNR/SSIM 和 LPIPS。评估对象是落盘8-bit PNG，指标值不参与候选选择。需单独配置好 pyiqa/指标权重；GT 清单缺失不影响纯生成。

## 验证与尚未验证的部分

针对性测试命令：

```bash
python -m pytest -q tests/test_idea7_feature_dps.py tests/test_denoising_context.py \
  tests/test_idea7_event_measurement.py tests/test_idea7_inference_entry.py
```

2026-10-24 最终完整 `python -m pytest -q tests` 结果：**86 passed，10 subtests passed**；仅1条 diffusers 使用旧 PyTorch pytree API 的弃用警告。没有失败或跳过的测试。

已实际运行 CPU 梯度测试，覆盖真实仓库 tiny UNet + residual cross-attention + DDPMScheduler + VAE 的2前向/1反传、零注入一致性、冻结权重、RNG恢复、投影与异常清理。入口集成使用真实dataset/proxy/runner和轻量生成器，验证无GT生成、前帧baseline缓存、K0、配置拒绝和VAE→scheduler随机序列。事件测试覆盖大时间戳、分片边界、正负抵消、原条件预处理回归、固定mask、离线标定和验证支持范围。

本机测试隔离环境是 Python3.12、PyTorch2.5.1 CPU、torchvision0.20.1 CPU、diffusers0.25.0、transformers4.38.2、peft0.9.0。仓库生产依赖声明未改变；这不等价于 whb 的 PyTorch2.0.1/GPU FP16 已通过。shell 完成参数逐项核对、独立落盘语法/LF/BOM及dry-run argv检查。

**尚未验证：真实 NPZ 采集语义、真实标定、正式 checkpoint、whb 文件路径存在性、GPU FP16显存/速度、PSNR/SSIM/LPIPS收益。** 观测误差降低只能作为算法自检，不能据此声称图像质量提升。
