# AGENTS.md

This is the agent-facing English guide for future code development. Keep it synchronized with `AGENTS.zh-CN.md`, which is the Chinese human-facing version for the user.

## Project Goal

This project is a secondary development fork of the original OSEDiff project. The original OSEDiff targets real-world image super-resolution. This project adapts it into an event-guided low-light image enhancement model and aims to support a high-quality computer-vision paper with clear novelty and experimental logic.

Task definition:

- Input: a low-light RGB image `I_low` and temporally/spatially aligned event data `E`.
- Output: an enhanced normal-light RGB image `I_gt` with the same spatial resolution as the input.
- Core constraint: preserve OSEDiff's one-step diffusion efficiency while making a meaningful event-guided low-light enhancement contribution, not just stacking modules.

## Current Repository State

The code repository is located at `D:\CodexProject\develop_OSEDiff\OSEDiff`. The outer workspace also contains:

- `AGENTS.md.docx`: source instructions for this guide.
- `训练&测试结果.xlsx`: existing training/testing result records.
- `Wu 等 - 2024 - One-Step Effective Diffusion Network for Real-World Image Super-Resolution.pdf`: the original OSEDiff paper.

The repository keeps the original OSEDiff structure and adds low-light enhancement plus event-guided branches:

- RGB-only LLE baseline:
  - `train_osediff_lle.py`
  - `test_osediff_lle.py`
  - `dataloaders/lle_dataset.py`
- Current event-guided mainline: Event-FiLM LoRA OSEDiff
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
- Plans and research notes:
  - `plan/Event_FiLM_LoRA_OSEDiff_Baseline_Plan.md`
  - `plan/idea2-1.md`
- Event preprocessing reference:
  - `references/eglol.py`

## Dataset

Development currently uses the indoor subset of the SDE dataset from `Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach`.

Local dataset root:

```text
D:\datasets\SDE_dataset
```

Common subdirectories:

```text
D:\datasets\SDE_dataset\SDE_indoor\sde_in_release\train
D:\datasets\SDE_dataset\SDE_indoor\sde_in_release\test
```

SDE indoor scene layout:

```text
train or test
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

Important pairing rules:

- Each `i_n` folder is one scene.
- `low/` contains low-light `.png` images and event `.npz` files.
- `normal/` contains normal-light GT `.png` images.
- Low-light images, normal-light images, and event files are usually timestamp-named, but matching files do not necessarily share the same filename.
- Correct pairing is by sorting each relevant folder by filename and matching by the same sorted index.
- Dataset changes must preserve temporal and spatial alignment among RGB, GT, and events.
- Geometric augmentation must share the same crop/flip/resize parameters for RGB, GT, and events.

Event preprocessing should be grounded in `references/eglol.py`. The current Event-FiLM dataset implementation is `dataloaders/sde_event_film_dataset.py`; it handles scene/index pairing, event slicing or voxelization, synchronized cropping, and normalization.

## Development and Training Platforms

Code is edited primarily on the local Windows 11 machine. The local machine has an RTX 5060 and can be used for light checks or some inference, but formal training is usually not done locally.

Formal training runs on a remote Linux computing platform, usually with two 40GB A100 GPUs. Before writing `.sh` training or testing scripts, ask which account will be used because pretrained-model and dataset paths vary by account.

Server separation: the platforms used by `whb`, `slpchen`, `zyx111`, and `zwy` are not the eight-GPU `cjl` server. Do not assume shared filesystems, model/dataset paths, SSH credentials, or runtime environments between these platforms and the eight-GPU server. Confirm the target platform as well as the account before deployment or launch.

Remote account paths:

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
  For train/test subdirectories, append `train` or `test` to the corresponding release root.
  Example indoor train:  /public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_indoor/sde_in_release/train
  Example outdoor train: /public/home/whb/Chen_Project/dataset/SDE_dataset/SDE_outdoor/sde_out_release/train

zwy
  Logging: This account cannot reach W&B. For all future zwy training scripts, save metrics locally on the server with TensorBoard (`--report_to tensorboard` and an output-local `--logging_dir`); do not configure W&B online logging.
  SD-2-1-base: /public/home/zhouweiyu/Chen_Project/PretrainedModel/stable-diffusion-2-1-base
  SDE_indoor root:  /public/home/zhouweiyu/Chen_Project/dataset/SDE/indoor/sde_in_release
  SDE_outdoor root: /public/home/zhouweiyu/Chen_Project/dataset/SDE/outdoor/sde_out_release
  For train/test subdirectories, append `train` or `test` to the corresponding release root.
  Pretrained evaluation models: /public/home/zhouweiyu/Chen_Project/PretrainedModel
  Evaluation files kept there: alexnet-owt-7be5be79.pth, LPIPS_v0.1_alex-df73285e.pth, pt_inception-2015-12-05-6726825d.pth

八卡
  SD-2-1-base: /home/data1/cjl/PretrainedModel/stable-diffusion-2-1-base/
  SDE_indoor root:  /home/data/Chen_data/SDE_dataset/sde_in_release/
  SDE_outdoor root: /home/data/Chen_data/SDE_dataset/sde_out_release/sde_out_release/
  Training results: /home/data1/cjl/OSEDiff_Project/TrainedWeights/
  Test/inference results: /home/data1/cjl/OSEDiff_Project/Test_results/
  Pretrained evaluation models: /home/data1/cjl/PretrainedModel/
  Evaluation files kept there: pt_inception-2015-12-05-6726825d.pth, LPIPS_v0.1_alex-df73285e.pth, alexnet-owt-7be5be79.pth
  For train/test subdirectories, append `train` or `test` to the corresponding release root.
  Example indoor train:  /home/data/Chen_data/SDE_dataset/sde_in_release/train
  Example outdoor train: /home/data/Chen_data/SDE_dataset/sde_out_release/sde_out_release/train
```

Training checkpoints are usually saved under `checkpoints/`. Test/inference results are usually saved under `test_results/` or the explicit `output_dir` passed to the script.

## Research Stages

### Stage 1: RGB-only LLE baseline

Goal: verify whether OSEDiff can perform low-light enhancement without event information.

Characteristics:

- Input is only the low-light RGB image.
- GT is the normal-light RGB image.
- No events are used.
- This is the baseline for all event-guided methods.

Scripts:

```text
train_osediff_lle.py
test_osediff_lle.py
```

### Stage 2: Event modulation branch

The current mainline injects event modulation into selected UNet self-attention projections. The default remains `attn1.to_v`; `--event_lora_targets` can select `q,k,v,out` (`attn1.to_q`, `attn1.to_k`, `attn1.to_v`, `attn1.to_out.0`).

Current Stage 2 semantics:

- Load the full Stage 1 checkpoint.
- Freeze the backbone and Event-LoRA A/B; selected ordinary LoRA groups may be activated and trained through the two independent switches.
- Train `EventEncoder` and the Event-FiLM modulation branch.
- The goal is to let the event branch learn how to modulate an already trained RGB LoRA.

Scripts:

```text
train_event_film_lora_osediff.py --stage 2
train_scripts/train_event_film_lora_stage2.sh
test_scripts/test_event_film_lora_stage2.sh
```

### Stage 3: Joint fine-tuning

Current Stage 3 semantics:

- Initialize from a Stage 2 Event-FiLM checkpoint.
- Jointly train the event branch and the selected Event-LoRA A/B.
- The OSEDiff/Stable Diffusion backbone should remain frozen.
- The goal is to further adapt the LoRA low-rank space to event modulation.

Scripts:

```text
train_event_film_lora_osediff.py --stage 3
train_scripts/train_event_film_lora_stage3.sh
test_scripts/test_event_film_lora_stage3.sh
```

## Current Event-FiLM LoRA Design

Core idea:

- Replace only selected UNet self-attention projections; never replace text cross-attention `attn2`.
- Reshape the ordinary LoRA low-rank feature `z = A(x)` into a spatial feature map.
- Use multi-scale event features from the event encoder to predict spatially varying `dgamma/dbeta`.
- Apply FiLM modulation in the LoRA low-rank space:

```text
z_mod = (1 + s * tanh(dgamma)) * z + s * dbeta
```

- Pass the modulated low-rank feature through LoRA up projection to obtain `lora_out`.
- The last layer of the modulation head should be zero-initialized so Stage 2 starts nearly equivalent to the Stage 1 ordinary LoRA.

Implementation notes:

- Event features are scoped to the current UNet pass. Clear them after forward normally, or after backward when gradient checkpointing needs them for recomputation.
- The current implementation assumes self-attention tokens form a square spatial grid and usually infers `H=W` from `sqrt(N)`. Explicitly handle `H,W` before supporting non-square inputs.
- When migrating Stage 1 PEFT LoRA A/B into Event-LoRA, check projection paths, adapter names, ranks, scaling, and tensor shapes carefully.

Event-LoRA target selection and checkpoint rules:

- `--event_lora_targets auto` uses `v` for a new Stage 1 initialization and inherits the target set from an Event checkpoint. Explicit subsets use `q,k,v,out`; `all` and `none` are also supported.
- `none` leaves all self-attention PEFT LoRA modules unchanged. EventEncoder and event heads do not train in this mode; a Stage 2/3 run must have another trainable generator path.
- Stage 2 freezes Event-LoRA A/B and trains event modulation. Stage 3 trains both. `--unet_lora_active_parts` and `--unet_lora_trainable_parts` independently control the remaining ordinary LoRA by down/mid/up group; trainable groups must be active.
- Inference uses `--unet_lora_parts` for ordinary LoRA. Stage 2/3 inference defaults to `none`; the idea3 inference parser keeps its historical default.
- Resume and ordinary inference require the checkpoint's Event-LoRA target set. Use `--init_from_event_checkpoint` with explicit targets for weights-only migration to a new target set; do not restore an old optimizer or step count across target changes.
- V-only checkpoints without target metadata are interpreted as legacy `v`. Key evidence must agree with the declared target set. The historical V shared-head state keys remain unchanged.
- After building Stage 2, run a zero-init consistency check: Stage 1 output and untrained Stage 2 output should be nearly identical.

## Code Development Requirements

Priority order for future development:

1. The code must train and test correctly.
2. Runtime efficiency should be reasonable; avoid clearly inefficient data flow or repeated computation.
3. Readability should support continued research iteration.
4. Stay consistent with the existing OSEDiff/Stage 1/Stage 2/Stage 3 structure and avoid unnecessary large rewrites.

Comment requirements:

- Key new or modified code must include Simplified Chinese comments.
- Comments should explain class/function responsibilities, tensor meaning, event representation, shape assumptions, key branches, and non-obvious numeric handling.
- For convolution blocks, attention modules, event encoders, and LoRA modulation modules, annotate input/output channels and shapes where practical.
- Do not generate large uncommented blocks of complex code.

Implementation requirements:

- Training scripts must support breakpoint resume, preferably through `--resume_from_checkpoint` or existing checkpoint recovery logic.
- `.sh` scripts should keep environment-variable overrides, for example `TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-...}"`.
- Before writing remote paths into `.sh` scripts, ask which account will be used; do not hard-code one account path as the only option.
- Add early existence checks for local paths, pretrained-model paths, checkpoint paths, and dataset paths where appropriate.
- Only run `git commit` after the user says the run succeeded and explicitly asks for a commit.
- Do not revert user changes. If unrelated dirty changes exist, leave them alone.

### Mandatory two-pass review for training and inference Bash scripts

Before delivering any new or modified training, testing, or inference `.sh` script, complete two separate checks:

1. **Requirements and reference check:** Compare the final script against the user's request and any reference script, parameter by parameter. Verify the account, dataset split and paths, Stage 1/2/3 initialization versus resume, output/checkpoint paths, model/channel settings, every loss weight and learning rate, step/batch settings, and test timestep. Evaluate Bash parameter-expansion defaults as actual values, including their signs: `${LAMBDA_VSD:-0.5}` defaults to positive `0.5`, whereas `${LAMBDA_VSD:--0.5}` defaults to negative `-0.5`. Do not infer equivalence from similar-looking text.
2. **Independent final-file check:** Re-read the saved `.sh` file, check Bash syntax with `bash -n` when Bash is available, and inspect the effective command-line arguments/default values without starting training or inference. Cross-check critical arguments against the Python parser and the conditions that enable losses or model branches; in particular, confirm that intended positive loss weights are positive and actually activate their branches. Check Linux line endings and absence of a UTF-8 BOM. If a remote path, runtime override, or actual launch configuration cannot be verified locally, state that limitation explicitly rather than claiming the run is validated.

Do not hand off a script until both checks pass. In the handoff, briefly report the critical resolved values and any unverified assumptions.

## Debugging Priorities

For training, inference, or metric anomalies, check these first:

1. Whether RGB/GT/events are correctly paired by sorted index within each scene.
2. Whether RGB, GT, and events share exactly the same crop/flip/resize.
3. Tensor shape mismatches, especially `[B,C,H,W]` versus `[B,N,C]` conversions.
4. dtype mismatches, especially under fp16 training/inference across base layers, LoRA layers, and the event branch.
5. Device mismatches.
6. Image normalization range consistency, usually `[-1,1]` for RGB.
7. Event voxel normalization stability, avoiding scale explosions from event-count differences.
8. VAE encode/decode latent scaling consistency.
9. Diffusion timestep assumptions required by the current scheduler/API.
10. Whether LoRA modules are correctly registered, frozen/unfrozen, saved, and loaded.
11. Whether Stage 1 to Stage 2/3 checkpoint key mapping is complete.
12. Whether train/test preprocessing matches.

If training collapses or enhancement quality is poor, analyze data flow, latent statistics, brightness distribution, normalization, consistency checks, and supervision signals before proposing architectural redesigns.

## Dependency Troubleshooting

- `ModuleNotFoundError: No module named 'basicsr'` means the active Python environment used to launch training does not have BasicSR installed. `requirements.txt` includes `basicsr`, so first verify that the training shell activates the intended conda/virtualenv and run `python -m pip show basicsr` inside that same environment.
- If BasicSR is missing, install it in the active environment with either `python -m pip install basicsr` or `python -m pip install -r requirements.txt`. On servers with slow PyPI access, use the site's preferred mirror, for example `python -m pip install basicsr -i https://pypi.tuna.tsinghua.edu.cn/simple`.
- Prefer `python -m pip ...` over plain `pip ...` so the package is installed into the exact Python interpreter that will run training.

## Experiment and Validation Suggestions

Event-guided methods should keep at least these sanity checks:

- correct events / zero events / shuffled events ablation.
- Stage 2 initialization output versus Stage 1 output consistency.
- `mean(abs(dgamma))`, `mean(abs(dbeta))`, `max(abs(dgamma))`, and `max(abs(dbeta))`.
- LoRA output norm and base output norm.
- event activity map and `dgamma/dbeta` visualization.
- scene-wise metrics, not only overall averages.

Common metrics:

```text
PSNR
SSIM
LPIPS
NIQE / MUSIQ / CLIPIQA (optional)
```

Event-related analysis can focus on:

- event-active regions.
- dark regions.
- dark + event-active regions.
- difference maps between the event-guided output and the RGB-only output.

## Requirements Before Modifying Original OSEDiff Logic

Before changing these areas, understand the original OSEDiff paper and the current implementation:

- architecture
- training objectives
- LoRA injection/loading/saving
- latent processing
- diffusion pipeline
- VAE/UNet interaction
- scheduler timestep assumptions

If a change deviates from original OSEDiff design, explain:

- why the deviation is needed.
- the expected benefit.
- the possible risks.
- whether it affects fairness among Stage 1/Stage 2/Stage 3 comparisons.

## Common File Map

```text
README.md                         Original OSEDiff usage notes
PROJECT_PROGRESS.md               Early LLE progress notes; may have encoding issues, reference only
osediff.py                        Original/basic OSEDiff model path
osediff_event_film.py             Current Event-FiLM model path
train_osediff_lle.py              RGB-only LLE training
test_osediff_lle.py               RGB-only LLE testing
train_event_film_lora_osediff.py  Event-FiLM Stage 2/3 training
test_event_film_lora_osediff.py   Event-FiLM testing and ablation
dataloaders/sde_event_film_dataset.py  SDE indoor + events dataset
models/event_encoder.py           Multi-scale event encoder
models/event_film_lora.py         Event-FiLM LoRA module
models/event_lora_utils.py        LoRA replacement, event feature setting, save/load helpers
train_scripts/                    Linux training scripts
test_scripts/                     Linux/remote testing scripts
plan/                             Research plans and implementation notes
references/eglol.py               Event preprocessing reference
```

## Server Upload Packaging Workflow

Use this workflow when the user asks for a reduced, server-ready copy of the repository and a ZIP archive for upload.

Required information from the user:

- New package folder name.
- Target server account, used to select the correct pretrained-model and dataset paths in the shell scripts.
- The exact checkpoint folders or files under `OSEDiff/checkpoints/` that must be included.

Optional information:

- Additional free-form text to record in the package metadata file.

If any required item is missing, remind the user and obtain it before packaging. The additional text is optional and may be left empty.

Packaging procedure:

1. Treat `D:\CodexProject\develop_OSEDiff\OSEDiff` as the source repository. Create the requested package folder as a sibling under `D:\CodexProject\develop_OSEDiff`.
2. Copy the repository code and supporting files into the new folder. Exclude `.git`, `experience`, `asserts`, all `__pycache__` or `pycache` directories, and the source `checkpoints` directory during the main copy.
3. Create `checkpoints/` in the package and copy only the checkpoint folders or files explicitly named by the user. Treat this as a whitelist: do not include any other weights.
4. Update the relevant training or testing `.sh` files inside the source repository for the requested server account while preserving environment-variable overrides. Use the account paths documented in the Development and Training Platforms section. Apply any run settings explicitly supplied by the user, then ensure the packaged copy contains the updated scripts.
5. Before archiving, inspect every packaged `.sh` file that may be executed on Linux at the byte level. It must use LF line endings, contain no CRLF or bare carriage returns, and have no UTF-8 BOM. Convert and re-check any failing file. A Git line-ending warning is not proof of the bytes on disk; trust the byte-level check.
6. Add `PACKAGE_INFO.txt` at the package root, encoded as UTF-8 without BOM. Record at least:
   - `Package date (Beijing time)`: the actual packaging date and time in `Asia/Shanghai`, preferably `YYYY-MM-DD HH:mm:ss +08:00`.
   - `Branch`: the source repository branch from `git branch --show-current`.
   - `Commit`: the full hash of the latest source commit from `git rev-parse HEAD`.
   - `Additional information`: the user's optional text, or a clear empty value such as `None`.
7. Create `<package-folder-name>.zip` beside the package folder. The ZIP must contain the package folder as its single top-level root.
8. Reopen and inspect the completed ZIP rather than trusting archive creation alone. Verify:
   - excluded directories are absent.
   - only the user-approved checkpoint whitelist is present.
   - `PACKAGE_INFO.txt` is present and contains the correct date, branch, commit, and optional text.
   - packaged shell scripts are still LF-only and BOM-free.
   - archive entry paths are valid and consistently represented.
9. Report the package folder path, ZIP path, included checkpoints, package date, branch, full commit hash, ZIP size, and SHA-256 checksum. Do not run training or inference as part of packaging.

## Collaboration Preferences

- If the user asks for a fix plan first, provide a reviewable plan and wait for approval before editing.
- If the user explicitly asks to implement a plan, execute it directly.
- If the user provides a traceback or command, prioritize the exact failing layer and the smallest viable fix.
- If explaining parameters such as `modulation_scale`, `lambda_edge`, `lambda_mod`, `lambda_smooth`, or `control_*`, explain from the current implementation path rather than guessing from the name.
- If the user says not to run code, only edit files and provide runnable commands; do not start training or inference.
- When answering code questions, prefer references to specific files and functions.
