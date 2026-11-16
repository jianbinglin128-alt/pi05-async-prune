#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-help}
if [[ "$mode" == help ]]; then
    printf '%s\n' 'Usage: bash scripts/run_prune.sh MODE GPU [script arguments]' \
      'MODE: server | eval | calibrate | calibrate-cpu | export | benchmark | summary | test' \
      'Uses the existing Python environment; does not copy or modify it.'
    exit 0
fi
gpu=${2:?Specify GPU index (ignored for CPU export/summary/test)}
shift 2
python_bin=${PRUNE_PYTHON:-$repo_dir/.venv/bin/python}
[[ -x "$python_bin" ]] || { printf 'Missing Python: %s\n' "$python_bin"; exit 2; }
export PYTHONPATH="$repo_dir:$repo_dir/src:$repo_dir/packages/openpi-client/src:${LIBERO_SOURCE:-$repo_dir/third_party/libero}${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$HOME/.cache/openpi}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.60}
export CUDA_VISIBLE_DEVICES="$gpu"
cd "$repo_dir"
case "$mode" in
    server) export JAX_PLATFORMS=cuda; script=scripts/serve_pruned_future.py ;;
    calibrate) export JAX_PLATFORMS=cuda; script=scripts/calibrate_future_channels.py ;;
    calibrate-cpu) export JAX_PLATFORMS=cpu; script=scripts/calibrate_future_channels.py ;;
    benchmark) export JAX_PLATFORMS=cuda; script=scripts/benchmark_future_channels.py ;;
    export) export JAX_PLATFORMS=cpu; script=scripts/export_future_channels.py ;;
    eval)
        export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
        export MUJOCO_EGL_DEVICE_ID="$gpu"
        export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero}
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        script=examples/libero/eval_pruned_future.py ;;
    summary) export JAX_PLATFORMS=cpu; script=scripts/summarize_pruned_future.py ;;
    test) export JAX_PLATFORMS=cpu; exec "$python_bin" -m unittest discover -s tests -p 'test_prune_*.py' -v ;;
    *) printf 'Unknown mode: %s\n' "$mode"; exit 2 ;;
esac
exec "$python_bin" "$script" "$@"
