#!/usr/bin/env bash
set -euo pipefail

# idea4 ESG 仅做推理；所有机器相关路径必须由调用者显式提供。
: "${PRETRAINED_MODEL_NAME_OR_PATH:?Set PRETRAINED_MODEL_NAME_OR_PATH to the SD2.1 root}"
: "${TEST_DATA_ROOT:?Set TEST_DATA_ROOT to the SDE test root}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an idea3 Stage4/Stage5 checkpoint}"
: "${RGB_EVENT_PRETRAINED_ROOT:?Set RGB_EVENT_PRETRAINED_ROOT to Encoder_I/E/Merge weights}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for idea4 ESG results}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/archive_run_context.sh"

EVENT_LORA_VARIANT="${EVENT_LORA_VARIANT:-residual_cross_attn}"
DECODER_REFINEMENT_VARIANT="${DECODER_REFINEMENT_VARIANT:-auto}"
if [[ "$EVENT_LORA_VARIANT" == residual_cross_attn* ]]; then
  EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-64}"
else
  EVENT_LORA_MOD_CHANNELS="${EVENT_LORA_MOD_CHANNELS:-32}"
fi

for required_dir in \
  "$PRETRAINED_MODEL_NAME_OR_PATH" \
  "$TEST_DATA_ROOT" \
  "$RGB_EVENT_PRETRAINED_ROOT"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory does not exist: $required_dir" >&2
    exit 1
  fi
done
for required_file in \
  "$CHECKPOINT_PATH" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_I.pth" \
  "$RGB_EVENT_PRETRAINED_ROOT/Encoder_E.pth"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[ERROR] Required file does not exist: $required_file" >&2
    exit 1
  fi
done

OPTIONAL_ARGS=()
if [[ "${CALCULATE_FID:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--calculate_fid)
fi
if [[ "${SAVE_IMAGES:-1}" == "0" ]]; then
  OPTIONAL_ARGS+=(--no-save_images)
fi
if [[ "${RESUME_IGNORE_MISMATCH:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--resume_ignore_mismatch)
fi
if [[ "${MAX_SAMPLES:-0}" != "0" ]]; then
  OPTIONAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

mkdir -p "$OUTPUT_DIR"
archive_run_context "$OUTPUT_DIR" "${BASH_SOURCE[0]}" idea4 esg_test \
  "PRETRAINED_MODEL_NAME_OR_PATH=$PRETRAINED_MODEL_NAME_OR_PATH" \
  "TEST_DATA_ROOT=$TEST_DATA_ROOT" \
  "CHECKPOINT_PATH=$CHECKPOINT_PATH" \
  "RGB_EVENT_PRETRAINED_ROOT=$RGB_EVENT_PRETRAINED_ROOT" \
  "EVENT_LORA_VARIANT=$EVENT_LORA_VARIANT" \
  "DECODER_REFINEMENT_VARIANT=$DECODER_REFINEMENT_VARIANT" \
  "ESG_GUIDANCE_SCALES=${ESG_GUIDANCE_SCALES:-0,0.1,0.25,0.5,0.75}" \
  "ESG_SWAP_RATIOS=${ESG_SWAP_RATIOS:-0.05,0.10}" \
  "ESG_SWAP_MODES=${ESG_SWAP_MODES:-spatial}" \
  "ESG_PAIR_POLICIES=${ESG_PAIR_POLICIES:-dissimilar}" \
  "ESG_FEATURE_SCALES=${ESG_FEATURE_SCALES:-32,16,8}" \
  "ESG_EXECUTION_MODE=${ESG_EXECUTION_MODE:-batched}"

echo "[Idea4 ESG] checkpoint=$CHECKPOINT_PATH"
echo "[Idea4 ESG] output=$OUTPUT_DIR"
echo "[Idea4 ESG] guidance=${ESG_GUIDANCE_SCALES:-0,0.1,0.25,0.5,0.75} ratio=${ESG_SWAP_RATIOS:-0.05,0.10}"
echo "[Idea4 ESG] mode=${ESG_SWAP_MODES:-spatial} policy=${ESG_PAIR_POLICIES:-dissimilar} scales=${ESG_FEATURE_SCALES:-32,16,8}"

python "$REPO_ROOT/test_idea4_esg.py" \
  --input_image "$TEST_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
  --event_lora_checkpoint "$CHECKPOINT_PATH" \
  --rgb_event_pretrained_root "$RGB_EVENT_PRETRAINED_ROOT" \
  --event_lora_variant "$EVENT_LORA_VARIANT" \
  --decoder_refinement_variant "$DECODER_REFINEMENT_VARIANT" \
  --cgdrs_event_levels "${CGDRS_EVENT_LEVELS:-1,2,3}" \
  --cgdrs_gate_bias "${CGDRS_GATE_BIAS:--2.0}" \
  --cgdrs_activity_mode "${CGDRS_ACTIVITY_MODE:-abs_mean}" \
  --lora_rank "${LORA_RANK:-4}" \
  --mod_channels "$EVENT_LORA_MOD_CHANNELS" \
  --modulation_scale "${EVENT_LORA_MODULATION_SCALE:-0.1}" \
  --mod_num_heads "${MOD_NUM_HEADS:-4}" \
  --mod_window_size "${MOD_WINDOW_SIZE:-8}" \
  --event_lora_cbam_reduction "${EVENT_LORA_CBAM_REDUCTION:-8}" \
  --event_lora_cbam_spatial_kernel "${EVENT_LORA_CBAM_SPATIAL_KERNEL:-7}" \
  --generator_timestep "${GENERATOR_TIMESTEP:-500}" \
  --image_size "${IMAGE_SIZE:-256}" \
  --mixed_precision "${MIXED_PRECISION:-fp16}" \
  --esg_guidance_scales "${ESG_GUIDANCE_SCALES:-0,0.1,0.25,0.5,0.75}" \
  --esg_swap_ratios "${ESG_SWAP_RATIOS:-0.05,0.10}" \
  --esg_swap_modes "${ESG_SWAP_MODES:-spatial}" \
  --esg_pair_policies "${ESG_PAIR_POLICIES:-dissimilar}" \
  --esg_feature_scales "${ESG_FEATURE_SCALES:-32,16,8}" \
  --esg_execution_mode "${ESG_EXECUTION_MODE:-batched}" \
  --esg_seed "${ESG_SEED:-42}" \
  "${OPTIONAL_ARGS[@]}"
