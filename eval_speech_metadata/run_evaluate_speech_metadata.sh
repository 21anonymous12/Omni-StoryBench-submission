#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


export CUDA_VISIBLE_DEVICES=6

TORCH_LIB="$(env -u PYTHONPATH PYTHONNOUSERSITE=1 python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

CONDA_ENV_LIB=""
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CONDA_ENV_LIB="${CONDA_PREFIX}/lib"
fi

export LD_LIBRARY_PATH="${TORCH_LIB}${CONDA_ENV_LIB:+:${CONDA_ENV_LIB}}:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64"

env -u PYTHONPATH -u PIP_CONSTRAINT PYTHONNOUSERSITE=1 python evaluate_speech_metadata.py "$@" 2>&1 | tee log.txt
