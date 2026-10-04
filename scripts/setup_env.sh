#!/bin/bash
# =====================================================================================
# setup_env.sh -- Step 0: create the virtual environment and configure the HF cache.
#
#   bash scripts/setup_env.sh            CPU-only install (default)
#   bash scripts/setup_env.sh --cuda     CUDA build of PyTorch (cluster nodes)
#
# The CUDA wheel index defaults to cu126; override with TORCH_CUDA_INDEX=cu124 etc.
# to match the driver on the target node (`nvidia-smi` prints the supported CUDA).
# Safe to re-run: pip skips anything already satisfied.
# =====================================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TORCH_INDEX="https://download.pytorch.org/whl/cpu"
[[ "${1:-}" == "--cuda" ]] && TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_INDEX:-cu126}"

# ---- Persistent Hugging Face cache (see run_eval.sh for the rationale) --------------
if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -n "${NLP_PROJECT_CACHE:-}" ]]; then
        export HF_HOME="${NLP_PROJECT_CACHE}/huggingface"
    elif [[ -d "/home/morg/NLP_2526b/${USER}" ]]; then
        export HF_HOME="/home/morg/NLP_2526b/${USER}/.cache/huggingface"
    else
        export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
    fi
fi
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}"

echo "Project root : ${PROJECT_ROOT}"
echo "HF_HOME      : ${HF_HOME}"
echo "Torch index  : ${TORCH_INDEX}"

# ---- Virtual environment ------------------------------------------------------------
if [[ ! -d .venv ]]; then
    echo ">>> Creating .venv"
    python3 -m venv .venv
fi
PY="${PROJECT_ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="${PROJECT_ROOT}/.venv/Scripts/python.exe"

echo ">>> Upgrading pip"
"${PY}" -m pip install --quiet --upgrade pip setuptools wheel

echo ">>> Installing PyTorch"
"${PY}" -m pip install --index-url "${TORCH_INDEX}" torch

echo ">>> Installing project requirements"
"${PY}" -m pip install -r requirements.txt

# ---- Persist the cache configuration for interactive shells -------------------------
cat > .env <<EOF
# Sourced by 'source .env' before interactive work.
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src:\${PYTHONPATH:-}"
EOF

echo ">>> Running the unit tests"
PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" -m pytest tests -q

echo ">>> Verifying the Hugging Face environment"
PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" src/setup_hf.py

echo
echo "Environment ready.  Next:  bash scripts/run_eval.sh"
