#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=${1:?Usage: run_libero_suites.sh CHECKPOINT NEW_OUTPUT [GPUS=0,1,2,3] [TRIALS=50]}
checkpoint=$(realpath "$checkpoint")
output=$(realpath -m "${2:?Specify new output directory}")
IFS=, read -ra gpus <<< "${3:-0,1,2,3}"
trials=${4:-50}; base_port=${BASE_PORT:-18930}
[[ ${#gpus[@]} == 4 ]] || { echo 'Supply four distinct physical GPU IDs'; exit 2; }
[[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" == 4 ]] || { echo 'GPU IDs must be distinct'; exit 2; }
[[ ! -e "$output" ]] || { echo "Output already exists: $output"; exit 2; }
mkdir -p "$output"
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$repo_dir/.libero}
suite_pids=()
cleanup() { for pid in "${suite_pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
suites=(libero_spatial libero_object libero_goal libero_10)
for i in 0 1 2 3; do
  setsid bash "$repo_dir/scripts/run_one_libero_suite.sh" "${gpus[$i]}" "$((base_port+i))" "${suites[$i]}" "$trials" "$checkpoint" "$output/${suites[$i]}" &
  suite_pids+=("$!")
done
rc=0
for pid in "${suite_pids[@]}"; do wait "$pid" || rc=1; done
echo "$rc" > "$output/exit_code.txt"
exit "$rc"
