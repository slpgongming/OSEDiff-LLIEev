#!/usr/bin/env bash

# 训练/推理入口脚本通过 source 调用此函数；每次运行都保留不可覆盖的副本。
archive_run_context() {
  local output_dir="$1"
  local script_path="$2"
  local stage="$3"
  local kind="$4"
  shift 4

  mkdir -p "$output_dir/run_configs"
  local resolved_script
  resolved_script="$(readlink -f -- "$script_path")"
  local timestamp
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  local pid="${BASHPID:-$$}"
  local basename_script
  basename_script="$(basename -- "$resolved_script")"
  local destination="$output_dir/run_configs/${timestamp}_pid${pid}_${kind}_${stage}_${basename_script}"

  cp -- "$resolved_script" "$destination"
  {
    printf 'stage=%q\n' "$stage"
    printf 'kind=%q\n' "$kind"
    printf 'timestamp=%q\n' "$timestamp"
    printf 'pid=%q\n' "$pid"
    printf 'script=%q\n' "$resolved_script"
    printf 'git_branch=%q\n' "$(git -C "$(dirname -- "$resolved_script")" branch --show-current 2>/dev/null || true)"
    printf 'git_commit=%q\n' "$(git -C "$(dirname -- "$resolved_script")" rev-parse HEAD 2>/dev/null || true)"
    local item key value
    for item in "$@"; do
      key="${item%%=*}"
      value="${item#*=}"
      printf '%s=%q\n' "$key" "$value"
    done
  } > "${destination}.resolved_args.txt"
  echo "[Run Archive] saved script and resolved args under ${output_dir}/run_configs"
}
