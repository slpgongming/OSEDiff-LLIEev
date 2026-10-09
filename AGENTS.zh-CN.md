# AGENTS.zh-CN.md

这是给用户阅读的中文开发指引。请与英文版 `AGENTS.md` 保持同步更新；英文版供 agent 优先读取，中文版供用户阅读。

## 项目目标

本项目是在原始 OSEDiff 基础上的二次开发。原始 OSEDiff 面向 real-world image super-resolution，本项目目标是将其改造为一个基于事件相机 events 的低光增强模型，并形成具有清晰创新点和实验逻辑的高水平计算机视觉论文工作。

当前任务定义：

- 输入：低光 RGB 图像 `I_low` 与时间/空间对齐的事件数据 `E`。
- 输出：正常光/增强后的 RGB 图像 `I_gt`，输出空间分辨率与输入一致。
- 核心约束：保留 OSEDiff 单步扩散恢复的效率优势，围绕 event-guided low-light enhancement 做有研究价值的改造，而不是简单堆叠模块。

## 当前代码状态

代码仓库位于 `D:\CodexProject\develop_OSEDiff\OSEDiff`。外层目录还包含：

- `AGENTS.md.docx`：本文件的来源说明。
- `训练&测试结果.xlsx`：已有训练/测试结果记录。
- `Wu 等 - 2024 - One-Step Effective Diffusion Network for Real-World Image Super-Resolution.pdf`：原始 OSEDiff 论文。

仓库目前保留了原始 OSEDiff 结构，同时新增/改造了低光增强和事件引导分支：

- RGB-only LLE baseline：
  - `train_osediff_lle.py`
  - `test_osediff_lle.py`
  - `dataloaders/lle_dataset.py`
- 当前事件引导主线：Event-FiLM LoRA OSEDiff
  - `train_event_film_lora_osediff.py`
  - `test_event_film_lora_osediff.py`
  - `osediff_event_film.py`
  - `models/event_encoder.py`
  - `models/event_film_lora.py`
  - `models/event_lora_utils.py`
  - `dataloaders/sde_event_film_dataset.py`
  - `train_scripts/train_event_film_lora_stage2.sh`
  - `train_scripts/train_event_film_lora_stage3.sh`
  - `test_scripts/test_event_film_lora_stage2.sh`
  - `test_scripts/test_event_film_lora_stage3.sh`
- 方案和研究计划：
  - `plan/Event_FiLM_LoRA_OSEDiff_Baseline_Plan.md`
  - `plan/idea2-1.md`
- 数据预处理参考：
  - `references/eglol.py`

## 数据集

开发阶段使用 SDE dataset 的 indoor 子集，来自论文 `Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach`。

本机数据集根目录：

```text
D:\datasets\SDE_dataset
```

常用子目录：

```text
D:\datasets\SDE_dataset\SDE_indoor\sde_in_release\train
D:\datasets\SDE_dataset\SDE_indoor\sde_in_release\test
```

SDE indoor 的 scene 结构：

```text
train 或 test
  i_1/
    low/
      *.png
      *.npz
    normal/
      *.png
  i_2/
    low/
    normal/
```

重要配对规则：

- 每个 `i_n` 表示一个拍摄场景。
- `low/` 中包含低光图像 `.png` 和事件 `.npz`。
- `normal/` 中包含正常光 GT 图像 `.png`。
- 低光图像、正常光图像和事件文件通常按时间戳命名，但三者文件名不一定相同。
- 正确配对方式是分别在各自目录内按文件名排序，然后用相同 index 配对。
- 修改 dataset 时必须保持 RGB、GT、events 的时间和空间对齐。
- 几何增强时，RGB、GT、events 必须共享同一个 crop/flip/resize 参数。

事件预处理优先参考 `references/eglol.py`。当前 Event-FiLM 数据集实现位于 `dataloaders/sde_event_film_dataset.py`，其中包含按 scene/index 配对、事件切片/体素化、同步裁剪和归一化逻辑。

## 开发与训练平台

代码编辑主要在本机 Windows 11 上完成。本机有 RTX 5060，可用于轻量检查或部分推理，但正式训练通常不在本机进行。

正式训练在远程 Linux 计算中心进行，常用 2 张 40GB A100。编写 `.sh` 训练/测试脚本前，必须先确认用户将使用哪个账号，因为预训练权重和数据集路径随账号变化。

服务器区分：`whb`、`slpchen`、`zyx111`、`zwy` 使用的平台与八卡 `cjl` 服务器不是同一台服务器。不得假设它们共享文件系统、模型/数据集路径、SSH 凭据或运行环境。部署或启动前，必须同时确认目标平台与账号。

远程账号路径：

```text
slpchen
  SD-2-1-base: /public/home/slpchen/stable-diffusion-2-1-base
  train: /public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/train
  test:  /public/home/slpchen/EvLight--main/SDE_dataset/SDE_indoor/sde_in_release/test

hutao
  SD-2-1-base: /public/home/hutao/Chen_Project/PretrainedModel/stable-diffusion-2-1-base
  train: /public/home/hutao/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/train
  test:  /public/home/hutao/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/test

zyx111
  SD-2-1-base: /public/home/zyx111/Chen_Project/PretrainedModel/stable-diffusion-2-1-base
  train: /public/home/zyx111/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/train
  test:  /public/home/zyx111/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/test

whb
  SD-2-1-base: /public/home/whb/Chen_Project/PretrainedModel/stable-diffusion-2-1-base
  SDE_outdoor root: /public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_outdoor/sde_out_release/
  SDE_indoor root:  /public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/
  train/test 子目录在对应 release root 后追加 `train` 或 `test`。
  示例 indoor train:  /public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/train
  示例 outdoor train: /public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_outdoor/sde_out_release/train

zwy
  日志：该账号无法连接 W&B。今后所有 zwy 训练脚本均将指标保存在服务器本地，使用 TensorBoard（`--report_to tensorboard`，`--logging_dir` 设在输出目录内），不配置 W&B 在线日志。
  SD-2-1-base: /public/home/zhouweiyu/Chen_Project/PretrainedModel/stable-diffusion-2-1-base
  SDE_indoor root:  /public/home/zhouweiyu/Chen_Project/dataset/SDE/indoor/sde_in_release
  SDE_outdoor root: /public/home/zhouweiyu/Chen_Project/dataset/SDE/outdoor/sde_out_release
  train/test 子目录在对应 release root 后追加 `train` 或 `test`。
  预训练评估模型: /public/home/zhouweiyu/Chen_Project/PretrainedModel
  评估文件直接放在该目录：alexnet-owt-7be5be79.pth、LPIPS_v0.1_alex-df73285e.pth、pt_inception-2015-12-05-6726825d.pth

八卡
  SD-2-1-base: /home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base/
  SDE_indoor root:  /home/data/Chen_data/SDE_dataset/sde_in_release/
  SDE_outdoor root: /home/data/Chen_data/SDE_dataset/sde_out_release/sde_out_release/
  训练结果: /home/data1/cjl/OSEDiff_Project/TrainedWeights/
  测试/推理结果: /home/data1/cjl/OSEDiff_Project/Test_results/
  预训练评估模型: /home/data1/cjl/PretrainedModel/
  评估文件直接放在该目录：pt_inception-2015-12-05-6726825d.pth、LPIPS_v0.1_alex-df73285e.pth、alexnet-owt-7be5be79.pth
  train/test 子目录在对应 release root 后追加 `train` 或 `test`。
  示例 indoor train:  /home/data/Chen_data/SDE_dataset/sde_in_release/train
  示例 outdoor train: /home/data/Chen_data/SDE_dataset/sde_out_release/sde_out_release/train
```

训练权重通常保存到 `checkpoints/`，测试/推理结果通常保存到 `test_results/` 或脚本显式指定的 `output_dir`。

## 研究路线

### Stage 1: RGB-only LLE baseline

目标：验证 OSEDiff 在不使用事件信息时能否完成低光增强。

特点：

- 输入只有低光 RGB 图像。
- GT 为正常光 RGB 图像。
- 不引入 events。
- 用作所有 event-guided 方法的基本对照。

相关脚本：

```text
train_osediff_lle.py
test_osediff_lle.py
```

### Stage 2: 事件调制分支

当前主线在选中的 UNet self-attention 投影的 LoRA 低秩空间中引入事件调制。默认仍为 `attn1.to_v`；`--event_lora_targets` 可选择 `q,k,v,out`，分别对应 `attn1.to_q`、`attn1.to_k`、`attn1.to_v`、`attn1.to_out.0`。

现有 Stage 2 语义：

- 加载完整 Stage 1 权重。
- 冻结主干和 Event-LoRA A/B；可通过两个独立开关激活并训练选中的普通 LoRA 分组。
- 训练 `EventEncoder` 与 Event-FiLM modulation branch。
- 目标是让事件分支学习如何调制已经训练好的 RGB LoRA。

相关脚本：

```text
train_event_film_lora_osediff.py --stage 2
train_scripts/train_event_film_lora_stage2.sh
test_scripts/test_event_film_lora_stage2.sh
```

### Stage 3: 联合微调

现有 Stage 3 语义：

- 从 Stage 2 Event-FiLM checkpoint 初始化。
- 联合训练事件分支与选中投影的 Event-LoRA A/B。
- OSEDiff/Stable Diffusion 主干仍应保持冻结。
- 目标是让 LoRA 低秩空间进一步适配事件调制。

相关脚本：

```text
train_event_film_lora_osediff.py --stage 3
train_scripts/train_event_film_lora_stage3.sh
test_scripts/test_event_film_lora_stage3.sh
```

## 当前 Event-FiLM LoRA 设计要点

核心思想：

- 只替换选中的 UNet self-attention 投影，不替换 text cross-attention 的 `attn2`。
- 将普通 LoRA 的低秩特征 `z = A(x)` reshape 成空间特征图。
- 使用事件编码器输出的多尺度 event features 预测空间相关的 `dgamma/dbeta`。
- 在 LoRA 低秩空间中做 FiLM 调制：

```text
z_mod = (1 + s * tanh(dgamma)) * z + s * dbeta
```

- 再通过 LoRA up projection 得到 `lora_out`。
- modulation head 最后一层应保持 zero-init，使 Stage 2 初始化时 Event-FiLM 近似等价于 Stage 1 普通 LoRA。

实现注意：

- event features 只用于当前 UNet 调用；通常在 forward 后清空，启用 gradient checkpointing 时须保留到 backward 重算完成再清空。
- 当前实现默认 self-attention token 是正方形空间网格，通常通过 `sqrt(N)` 推断 `H=W`。支持非正方形输入前必须显式处理 `H,W`。
- 从 Stage 1 checkpoint 的 PEFT LoRA A/B 迁移到 Event-LoRA 时，要检查投影路径、adapter 名称、rank、scaling 和张量形状。

Event-LoRA 目标选择与 checkpoint 规则：

- `--event_lora_targets auto` 从 Stage 1 新建时选择 `v`，加载 Event checkpoint 时继承其中的目标集。显式选择可用 `q,k,v,out` 子集，也支持 `all` 和 `none`。
- `none` 保留全部原始 self-attention PEFT LoRA 结构；此时 EventEncoder 和事件头不训练，Stage 2/3 必须有其他可训练的生成器路径。
- Stage 2 冻结 Event-LoRA A/B，训练事件调制分支；Stage 3 同时训练两者。`--unet_lora_active_parts` 和 `--unet_lora_trainable_parts` 分别控制剩余普通 LoRA 的 down/mid/up 分组，解冻组必须属于激活组。
- 推理用 `--unet_lora_parts` 控制普通 LoRA。Stage 2/3 测试默认 `none`；idea3 测试入口保留原有默认值。
- 断点恢复和普通推理要求目标集与 checkpoint 一致。若要更换目标集，用 `--init_from_event_checkpoint` 配合显式 targets 仅迁移权重，不恢复原优化器和步数。
- 缺少 targets 元信息的旧 checkpoint 按 V-only 解释，并核对实际权重键；旧 V 共享头键名保持不变。
- Stage 2 刚构建完成后应做 zero-init consistency check：Stage 1 输出与 Stage 2 未训练输出应几乎一致。

## 代码开发要求

本项目后续开发优先级：

1. 代码能正确训练/测试。
2. 执行效率合理，避免明显低效的数据流或重复计算。
3. 可读性强，便于继续研究迭代。
4. 保持与现有 OSEDiff/Stage 1/Stage 2/Stage 3 结构一致，避免无意义的大范围重写。

注释要求：

- 新增或修改的关键代码必须添加简体中文注释。
- 注释应解释类/函数职责、张量含义、事件表示、shape 假设、关键分支逻辑和不明显的数值处理。
- 卷积块、注意力模块、事件编码器、LoRA 调制模块等核心结构，要尽量标明输入/输出 channel 和 shape。
- 大段复杂逻辑不要生成无注释代码。

实现要求：

- 训练脚本必须支持断点续训，优先沿用 `--resume_from_checkpoint` 或现有 checkpoint 恢复逻辑。
- `.sh` 脚本应保留环境变量覆盖能力，例如 `TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-...}"`。
- `.sh` 脚本中涉及远程路径时，先确认账号；不要把某一个账号路径硬编码为唯一选择。
- 对本地路径、预训练模型路径、checkpoint 路径、数据集路径增加必要的存在性检查，尽量早失败。
- 只有用户明确说运行成功并要求 `git commit` 时，才执行 git commit。
- 不要回退用户已有未提交代码；遇到无关 dirty change 时只绕开。

### 训练与推理 Bash 脚本必须二次检查

交付任何新增或修改的训练、测试、推理 `.sh` 脚本前，必须分别完成两轮检查：

1. **需求与参考脚本核对：** 逐项对照用户要求和参考脚本，检查账号、数据集划分与路径、Stage 1/2/3 初始化还是断点续训、输出与权重路径、模型和通道设置、所有损失权重及学习率、步数与 batch 设置、测试 timestep。必须核对 Bash 参数展开后的**实际默认值及正负号**：`${LAMBDA_VSD:-0.5}` 默认是正 `0.5`，`${LAMBDA_VSD:--0.5}` 默认是负 `-0.5`；不能只凭文本相似就认为参数一致。
2. **落盘文件独立复核：** 重新读取最终 `.sh` 文件；环境有 Bash 时运行 `bash -n` 检查语法；在不启动训练或推理的前提下核对实际命令行参数及默认值。将关键参数与 Python 参数解析器、损失或模型分支的启用条件交叉检查，特别确认预期为正的损失权重确实为正且会启用对应分支。检查 Linux 换行符及无 UTF-8 BOM。若远程路径、运行时环境变量覆盖或实际启动配置无法在本地核实，必须明确说明，不能宣称已验证运行配置。

两轮检查都通过后才能交付脚本；交付时简要列出关键参数的实际值和未核实的假设。

## 调试优先级

训练、推理或指标异常时，优先检查：

1. RGB/GT/events 是否按 scene 内排序 index 正确配对。
2. RGB、GT、events 的 crop/flip/resize 是否完全同步。
3. tensor shape 是否匹配，尤其是 `[B,C,H,W]` 与 `[B,N,C]` 的转换。
4. dtype 是否匹配，特别是 fp16 推理/训练中 base layer、LoRA layer、event branch 的 dtype。
5. device 是否匹配。
6. 图像归一化范围是否一致，RGB 通常为 `[-1,1]`。
7. event voxel 归一化是否稳定，避免 event 数量差异导致尺度爆炸。
8. VAE encode/decode latent scaling 是否一致。
9. diffusion timestep 是否满足当前 scheduler/API 假设。
10. LoRA 是否被正确注册、冻结/解冻、保存和加载。
11. Stage 1 到 Stage 2/3 的 checkpoint key 映射是否完整。
12. train/test preprocessing 是否一致。

如果训练崩溃或增强效果差，先分析数据流、latent 统计、亮度分布、归一化、一致性检查和监督信号，再提出结构性改造。

## 依赖问题排查

- `ModuleNotFoundError: No module named 'basicsr'` 表示当前启动训练的 Python 环境没有安装 BasicSR。`requirements.txt` 中已经包含 `basicsr`，所以先确认训练 shell 激活的是预期的 conda/virtualenv，并在同一个环境中运行 `python -m pip show basicsr`。
- 如果确实缺失，在当前环境中运行 `python -m pip install basicsr`，或重新执行 `python -m pip install -r requirements.txt`。如果服务器访问 PyPI 较慢，可使用站点推荐镜像，例如 `python -m pip install basicsr -i https://pypi.tuna.tsinghua.edu.cn/simple`。
- 优先使用 `python -m pip ...`，不要只用 `pip ...`，这样能确保包装进实际运行训练的 Python 解释器里。

## 实验与验证建议

事件引导方法至少要保留以下 sanity check：

- correct events / zero events / shuffled events 消融。
- Stage 2 初始化输出与 Stage 1 输出的一致性检查。
- `mean(abs(dgamma))`、`mean(abs(dbeta))`、`max(abs(dgamma))`、`max(abs(dbeta))`。
- LoRA output norm 与 base output norm。
- event activity map 与 `dgamma/dbeta` 可视化。
- scene-wise 指标，避免只看 overall average。

常规指标：

```text
PSNR
SSIM
LPIPS
NIQE / MUSIQ / CLIPIQA (可选)
```

事件相关指标或可视化可优先关注：

- event-active 区域。
- dark-region 区域。
- dark + event-active 区域。
- output 与 RGB-only output 的差异图。

## 修改原始 OSEDiff 逻辑时的要求

修改以下模块前，应先理解原始 OSEDiff 论文和现有实现：

- architecture
- training objectives
- LoRA injection/loading/saving
- latent processing
- diffusion pipeline
- VAE/UNet interaction
- scheduler timestep assumptions

如果改动偏离原始 OSEDiff 设计，必须在说明中解释：

- 为什么需要偏离。
- 预期收益是什么。
- 可能风险是什么。
- 对 Stage 1/Stage 2/Stage 3 对比是否会产生不公平影响。

## 常用文件定位

```text
README.md                         原始 OSEDiff 使用说明
PROJECT_PROGRESS.md               早期 LLE 进展记录，可能存在编码问题，仅作参考
osediff.py                        原始/基础 OSEDiff 模型路径
osediff_event_film.py             当前 Event-FiLM 模型路径
train_osediff_lle.py              RGB-only LLE 训练
test_osediff_lle.py               RGB-only LLE 测试
train_event_film_lora_osediff.py  Event-FiLM Stage 2/3 训练
test_event_film_lora_osediff.py   Event-FiLM 测试和消融
dataloaders/sde_event_film_dataset.py  SDE indoor + events 数据集
models/event_encoder.py           多尺度事件编码器
models/event_film_lora.py         Event-FiLM LoRA 模块
models/event_lora_utils.py        LoRA 替换、事件特征设置、保存/加载工具
train_scripts/                    Linux 训练脚本
test_scripts/                     Linux/远程测试脚本
plan/                             研究计划与实现方案
references/eglol.py               事件预处理参考
```

## 服务器上传打包流程

当用户要求制作一个精简、可上传服务器的仓库副本及 ZIP 压缩包时，使用以下流程。

用户必须提供：

- 新打包文件夹的名称。
- 目标服务器账号，用于选择 shell 脚本中的预训练模型和数据集路径。
- `OSEDiff/checkpoints/` 下需要打入压缩包的 checkpoint 文件夹或文件的准确名称。

用户可以选择提供：

- 要写入打包信息文件的自定义附加说明。

如果任一必填项缺失，先提醒用户补充，再开始打包。附加说明为可选项，可以为空。

打包步骤：

1. 以 `D:\CodexProject\develop_OSEDiff\OSEDiff` 为源代码仓库，在 `D:\CodexProject\develop_OSEDiff` 下创建用户指定名称的同级打包文件夹。
2. 将仓库代码和配套文件复制到新文件夹。主复制过程排除 `.git`、`experience`、`asserts`、所有 `__pycache__` 或 `pycache` 目录，以及源仓库的整个 `checkpoints` 目录。
3. 在打包目录中创建 `checkpoints/`，只复制用户明确指定的 checkpoint 文件夹或文件。这里采用白名单规则，不得混入其他权重。
4. 根据用户指定的服务器账号，修改源仓库中相关的训练或测试 `.sh` 文件，同时保留环境变量覆盖能力。账号路径使用“开发与训练平台”章节记录的路径；用户明确给出的本次运行参数也要一并写入，然后确保打包副本包含修改后的脚本。
5. 压缩前，对打包目录中所有可能在 Linux 执行的 `.sh` 文件做字节级检查：必须使用 LF 换行，不得包含 CRLF 或单独的回车符，也不得含 UTF-8 BOM。发现问题时先转换，再重新检查。Git 的换行警告不能代表磁盘文件的真实字节，应以字节级检查结果为准。
6. 在打包文件夹根目录新增 `PACKAGE_INFO.txt`，编码必须为不带 BOM 的 UTF-8。至少写入：
   - `打包时间（北京时间）`：使用 `Asia/Shanghai` 的实际打包日期和时间，推荐格式为 `YYYY-MM-DD HH:mm:ss +08:00`。
   - `Branch`：通过 `git branch --show-current` 获取源仓库当前分支名。
   - `Commit`：通过 `git rev-parse HEAD` 获取源仓库最近一次提交的完整哈希。
   - `附加说明`：写入用户提供的可选信息；未提供时明确写为 `无`。
7. 在打包文件夹旁生成 `<新文件夹名>.zip`。ZIP 内必须以该打包文件夹作为唯一的顶层根目录。
8. ZIP 生成后必须重新打开并直接检查压缩包内容，不能只凭压缩命令成功就认为完成。确认：
   - 所有排除目录均不存在。
   - `checkpoints` 中只有用户指定的白名单权重。
   - `PACKAGE_INFO.txt` 存在，且日期、branch、commit 和附加说明正确。
   - ZIP 内的 shell 脚本仍为纯 LF 且无 BOM。
   - 压缩包条目路径合法且格式一致。
9. 最终向用户报告打包文件夹路径、ZIP 路径、包含的 checkpoint、打包时间、branch、完整 commit 哈希、ZIP 大小和 SHA-256。打包过程不运行训练或推理。

## 协作偏好

- 如果用户要求先给修复计划，先给可审阅计划，得到确认后再改代码。
- 如果用户明确要求实现计划，直接按计划执行。
- 如果用户贴 traceback 或命令，优先解释具体失败层和最小可行修复。
- 如果用户要求解释参数，例如 `modulation_scale`、`lambda_edge`、`lambda_mod`、`lambda_smooth`、`control_*`，必须从当前实现路径解释，不要只按名字猜。
- 如果用户说不要运行代码，就只修改文件并给出可复制的命令，不主动训练/推理。
- 回答代码问题时优先引用具体文件和函数。
