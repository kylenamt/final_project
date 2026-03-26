#!/usr/bin/env bash
set -euo pipefail

# Ensure TensorFlow can find CUDA/cuDNN libs from the conda environment
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

MODE="${MODE:-train}"
TFRECORD_PATH="${TFRECORD_PATH:-data/ddsp_data/vocalset.tfrecord}"
SAVE_DIR="${SAVE_DIR:-artifacts/ddsp_runs/vocalset}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GIN_MODEL="${GIN_MODEL:-models/solo_instrument.gin}"
GIN_DATASET="${GIN_DATASET:-datasets/tfrecord.gin}"
GIN_EVAL="${GIN_EVAL:-eval/basic_f0_ld.gin}"
GIN_SEARCH_PATH="${GIN_SEARCH_PATH:-configs/ddsp_gin}"
EXTRA_GIN_PARAMS="${EXTRA_GIN_PARAMS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v ddsp_run >/dev/null 2>&1; then
  echo "ddsp_run not found in PATH. Install ddsp (pip install ddsp) and try again." >&2
  exit 1
fi

if [[ ! -f "${TFRECORD_PATH}" && "${TFRECORD_PATH}" != *"*"* ]]; then
  echo "TFRecord not found at ${TFRECORD_PATH}. Set TFRECORD_PATH or run 'make prepare'." >&2
  exit 1
fi

GIN_SEARCH_FLAGS=()
if [[ -z "${GIN_SEARCH_PATH}" ]]; then
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    DETECTED_GIN_PATH=$(${PYTHON_BIN} - <<'PY'
import os
try:
    import ddsp
except Exception:
    raise SystemExit(0)
gin_dir = os.path.join(os.path.dirname(ddsp.__file__), "training", "gin")
print(gin_dir if os.path.isdir(gin_dir) else "")
PY
    )
    if [[ -n "${DETECTED_GIN_PATH}" ]]; then
      GIN_SEARCH_PATH="${DETECTED_GIN_PATH}"
    fi
  fi
fi

if [[ -n "${GIN_SEARCH_PATH}" ]]; then
  IFS=":" read -r -a _paths <<< "${GIN_SEARCH_PATH}"
  for p in "${_paths[@]}"; do
    GIN_SEARCH_FLAGS+=("--gin_search_path=${p}")
  done
else
  echo "GIN_SEARCH_PATH is empty and DDSP gin directory could not be detected." >&2
  echo "Set GIN_SEARCH_PATH to configs/ddsp_gin or ddsp/training/gin in your env." >&2
  exit 1
fi

# Add gin search path and project root to PYTHONPATH so gin can resolve
# custom modules (e.g. src.safe_trainer).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_EXTRA_PATHS="${GIN_SEARCH_PATH}:${PROJECT_ROOT}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${_EXTRA_PATHS}:${PYTHONPATH}"
else
  export PYTHONPATH="${_EXTRA_PATHS}"
fi

GIN_PARAM_FLAGS=("--gin_param=TFRecordProvider.file_pattern='${TFRECORD_PATH}*'" "--gin_param=batch_size=${BATCH_SIZE}")
if [[ -n "${EXTRA_GIN_PARAMS}" ]]; then
  IFS="," read -r -a _params <<< "${EXTRA_GIN_PARAMS}"
  for p in "${_params[@]}"; do
    GIN_PARAM_FLAGS+=("--gin_param=${p}")
  done
fi

mkdir -p "${SAVE_DIR}"

ddsp_run \
  --mode="${MODE}" \
  --save_dir="${SAVE_DIR}" \
  --gin_file="${GIN_MODEL}" \
  --gin_file="${GIN_DATASET}" \
  --gin_file="${GIN_EVAL}" \
  "${GIN_SEARCH_FLAGS[@]}" \
  "${GIN_PARAM_FLAGS[@]}" \
  --alsologtostderr
