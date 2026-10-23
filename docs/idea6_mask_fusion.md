# idea6：冻结 6 号 RGB LoRA 与 90 号事件参数，训练 mask 与注入强度

基于 `origin/idea4` 的 `3bcec8d4ee104d5655d83146de9735c4abd9a854`。
新入口独立于原 Stage 2/3，不改变旧实验行为。首次实验固定 256×256、5 个有效
事件 bin（0–4），默认 generator timestep=500，不启用 decoder refinement。

## 前向与冻结范围

每个 `attn1.to_v` 保留原 6 号 PEFT 层，包括它的 base、RGB A/B、scale。
其余普通 UNet LoRA 三组全部激活且冻结，`conv_in` 与 VAE encoder LoRA 也来自 6 号。
90 号仅迁移独立 A90/B90、EventEncoder、同尺度共享 residual cross-attention heads。
**不从 90 号加载普通 UNet LoRA、conv_in、VAE，也不加入无条件 B90*A90*h。**

```text
z90 = A90(h)
D90 = (alpha90 / rank90) * modulation_scale90 * B90(H(z90, f(E)) - H(z90, f(0)))
M   = sigmoid(mask_net(h, f(E) - f(0)))              # [B,1,H,W]
a   = max_strength * tanh(raw_strength)            # 每个插入层一个标量
out = RGB6_layer(h) + a * M * D90
```

mask 用 1×1 RGB/event 投影、3×3 融合卷积与 1×1 输出，每层独立；输入只有当前
UNet 特征和冻结事件特征，没有 GT。默认隐藏通道 16，`max_strength=2`，允许有界
正/负修正。mask 输出层初始全零，使 M=0.5；raw_strength 初始 0，使新模型初始
等于 RGB6。第一步主要学习强度，强度离开零后 mask 获得图像梯度。

H、A90/B90 参数冻结，但仍保留对 h 的梯度，保证早期 mask 可经后续冻结层反传。
VAE encoder 与纯事件编码可以 no_grad；UNet 与 VAE decoder 不可整体 no_grad。
全模型仅 mask 参数和 raw_strength 可训练，启动时严格检查。

一次输出只执行一次 UNet。零事件参考增加的是事件编码器和小型事件 head 的计算，
没有第二套 UNet，也没有两路整图结果混合。RGB/correct/zero/shuffled 是离线分别
运行的四种评估模式，每种模式仍各自单次 UNet。

### 旧 checkpoint 的缩放兼容

idea4 `replace_self_attn_to_v_with_event_lora` 从 PEFT 层复制实际 alpha，而源 checkpoint
的全局 `event_lora_config.alpha` 保存为 rank。当前 PEFT 初始化未显式设置 alpha，
其默认值为 8。因此旧 90 号 checkpoint 使用 `--legacy_event_lora_alpha 8`（默认），
不能直接拿全局 alpha=4 替代。若 checkpoint 含逐层 `event_lora_layer_configs`，优先
读取该层 alpha。配置与来源 SHA-256 一起保存；异构/不完整参数直接报错。
本次只支持 90 号的 `residual_cross_attn`，不静默转换其他变体。

## 训练

服务器路径尚未确定，以下所有 `/PATH/...` 都必须替换成该服务器实际路径。
此处未生成带账号默认值的 `.sh`，也未部署或启动远程训练。

使用能够运行 idea4 的 Linux CUDA 环境，另外确保 `accelerate`、`lpips` 与
`tensorboard` 可导入。推荐 Accelerate 0.27.2 或经过验证的兼容版本；项目原
`requirements.txt` 没有更改。首先用下面命令做 50 个 optimizer step 的显存检查：

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_idea6_mask_fusion.py \
  --pretrained_model_name_or_path /PATH/stable-diffusion-2-1-base \
  --train_data_root /PATH/SDE_indoor/train \
  --rgb_checkpoint /PATH/experiment6/model_94001.pkl \
  --event_checkpoint /PATH/experiment90/model_final.pkl \
  --output_dir /PATH/results/idea6_mask_pilot \
  --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_steps 50 --checkpointing_steps 50 \
  --generator_timestep 500 --mixed_precision fp16 \
  --learning_rate_mask 1e-4 --learning_rate_strength 1e-3 \
  --lambda_l2 1 --lambda_lpips 1 --lambda_edge 0.05 \
  --report_to tensorboard
```

有效 batch 为 1×2×4=8。首轮正式 pilot 默认 5000 个 optimizer step，500 step 保存
一次；50 步只是资源与数值检查，不能据此判断模型是否有效。若要继续这次运行，
保持同一输出目录及相同数据/优化设置，用 `--resume_from_checkpoint latest` 并提高
`--max_train_steps`。resume 不需要再传两个源 checkpoint，因为 fusion.pt 已包含
两套所需的冻结 adapter。完整 SD 基础模型仍从 pretrained 路径读取。

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_idea6_mask_fusion.py \
  --pretrained_model_name_or_path /PATH/stable-diffusion-2-1-base \
  --train_data_root /PATH/SDE_indoor/train \
  --output_dir /PATH/results/idea6_mask_pilot \
  --resume_from_checkpoint latest --max_train_steps 5000 \
  --train_batch_size 1 --gradient_accumulation_steps 4 --mixed_precision fp16
```

损失为 `1*MSE + 1*LPIPS(VGG) + 0.05*event-weighted GT gradient L1`，沿用 idea4 的
edge 定义。没有 VSD、regularizer UNet、第二个优化器或旧 event-head 参数正则。
LPIPS 权重为 0 时不会创建 VGG。冻结主干显式使用半精度，mask/强度保留 FP32 主参数。
固定 prompt 的文本编码器只在初始化使用一次，之后释放。

可选 `--enable_xformers_memory_efficient_attention`（仅已正确安装 xformers 时）。
显存仍不足时加入 `--gradient_checkpointing`：新实现对 UNet 与 VAE decoder 使用
非重入 checkpoint，并由闭包捕获该 microbatch 的事件特征，在重算时重新设置/清理。
没有调用旧入口的 `unet.enable_gradient_checkpointing()`。
这会增加训练反向重算；评估/推理仍只调用一次 UNet。

输出包括：

- `run.json`：参数、可训练参数总数、源 checkpoint 路径与 SHA-256。
- `metrics.jsonl` 与 `logs/`：训练损失、各层 mask/强度/注入幅度、每卡峰值
  allocated/reserved GiB。损失日志是该更新最后一个 microbatch 的跨卡平均。
- `checkpoint-N/fusion.pt`：两套源 adapter、mask/强度、结构配置与来源。
- 同目录 Accelerate optimizer/scaler/RNG 状态和 `progress.json`：断点恢复。
  不重复保存整套 SD 主干。只在 optimizer step 边界保存；`latest` 跳过未完成目录。
  另外显式保存 `idea6_rng_RANK.pt`，避免旧 Accelerate 配合新 PyTorch 时静默跳过 RNG 恢复。

恢复要求相同 world size、数据、batch/累积、loss 与学习率；数据 crop/flip 按
epoch/index 确定，样本顺序使用显式 epoch seed，并跳过已完成批次。
不支持把训练中途的源 RGB/event checkpoint 换成其他文件。

## 配对推理与有效性判断

```bash
python test_idea6_mask_fusion.py \
  --pretrained_model_name_or_path /PATH/stable-diffusion-2-1-base \
  --checkpoint /PATH/results/idea6_mask_pilot/checkpoint-5000 \
  --test_data_root /PATH/SDE_indoor/test \
  --output_dir /PATH/results/idea6_mask_eval \
  --ablation_modes rgb,correct,zero,shuffled --mixed_precision fp16 --save_masks
```

测试复用 idea4 的 scene/index 配对与中心裁剪流程；每张低光 RGB **只采样一次 VAE
latent，四种模式共用该 latent**。`rgb` 是同一生成器关闭事件注入的完整 6 号路径，
不是重新加载历史输出图片；不同历史裁剪或随机 latent 的图片不可直接逐像素比较。
`shuffled` 使用数据集确定性的异帧事件，batch=1 时也有效。

每模式保存 `scene/0001_result.png`、GT、低光输入、事件活动图；`--save_masks` 保存
各插入层的原生分辨率 mask。输出 `per_image.csv` 和 `summary.json`，PSNR/SSIM 为
与旧入口相同的 Y-channel/YCbCr 口径，LPIPS 使用 PyIQA 默认实现。FID 可后续通过
已有离线工具计算，避免这次推理额外常驻 Inception。`--skip_metrics` 可只生成图像。

检查顺序：

1. `zero` 与 `rgb` 的最大绝对输出差 ≤1e-5；否则脚本报错，不把失败当有效消融。
2. 比较 correct 对 rgb/zero 的逐场景指标和局部细节；亮度/颜色与形状都要看。
3. 比较 correct 与 shuffled，确认改进需要正确事件内容。
4. 查看强度是否全部接近 0，以及 mask 是否退化为常数。只有相对 RGB6 改善才表明
   固定事件增量能被利用；若要进一步证明“空间 mask 优于标量混合”，还需独立的
   仅训练各层 alpha 对照，当前 pilot 不声称已经证明这一点。

本实验不训练 90 号的事件特征提取能力，也不新增高分辨率细节生成分支。
结果不好只能说明这套固定增量与调节机制的组合尚未奏效，不能直接证明事件无用。

## 本地验证

```bash
python -m unittest discover -s tests -p 'test_idea6_mask_fusion.py' -v
```

测试覆盖初始 RGB 等价、任意 gate 后零事件等价、真实旧事件头差分迁移、冻结范围、
alpha 启动和 mask 梯度、checkpoint 前后梯度一致、不同 batch 的条件隔离、保存加载
与不完整权重拒绝。安装完整依赖时，还测试真实 PEFT/自定义 UNet、带 VAE 的小型
完整生成器图像反传及权重往返、Accelerate 优化器与 RNG 恢复。

本次本地检查：17 项测试全部通过（包含后续 epoch 中途恢复的样本顺序）；
CPU PyTorch 2.14.1、Diffusers 0.25.0、PEFT 0.9.0、
Accelerate 0.27.2、Transformers 4.38.2。隔离测试环境位于任务工作目录，未更改项目的
依赖锁定或用户现有 Python 环境。Transformers 测试版本与 requirements 的 4.28.1
不同（旧 tokenizer 在本机 Python 3.12 无预编译包）。

CPU 小模型测试不替代服务器上真实 6/90 权重、双卡 DDP、FP16
与显存峰值验证；路径确定后应先跑上述 50 步资源检查。
