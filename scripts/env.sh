#!/usr/bin/env bash
# Shared, relocation-safe environment for every experiment command.
# Source this file; do not execute it in a subshell when invoking Python tools.

set -euo pipefail

_sr_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export SENTINEL_REPAIR_ROOT="$(cd -- "${_sr_script_dir}/.." && pwd -P)"
export SENTINEL_REPAIR_UPSTREAM="${SENTINEL_REPAIR_ROOT}/third_party/llm-verifier-noise"
export SENTINEL_REPAIR_VENV="${SENTINEL_REPAIR_ROOT}/.venv"
export SENTINEL_REPAIR_TMP="${SENTINEL_REPAIR_ROOT}/tmp"

# Keep every cache, temporary file, downloaded model, and tracker artifact inside
# the isolated experiment repository. HOME is deliberately left untouched.
export XDG_CACHE_HOME="${SENTINEL_REPAIR_ROOT}/.cache/xdg"
export XDG_CONFIG_HOME="${SENTINEL_REPAIR_ROOT}/.cache/xdg-config"
export XDG_DATA_HOME="${SENTINEL_REPAIR_ROOT}/.cache/xdg-data"
export UV_CACHE_DIR="${SENTINEL_REPAIR_ROOT}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${SENTINEL_REPAIR_ROOT}/.cache/uv/python"
export UV_TOOL_DIR="${SENTINEL_REPAIR_ROOT}/.cache/uv/tools"
export UV_TOOL_BIN_DIR="${SENTINEL_REPAIR_ROOT}/.local/bin"
export UV_NO_PROGRESS=1
export PIP_CACHE_DIR="${SENTINEL_REPAIR_ROOT}/.cache/pip"

export HF_HOME="${SENTINEL_REPAIR_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

export TORCH_HOME="${SENTINEL_REPAIR_ROOT}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${SENTINEL_REPAIR_ROOT}/.cache/torch/extensions"
export TORCHINDUCTOR_CACHE_DIR="${SENTINEL_REPAIR_ROOT}/.cache/torch/inductor"
export TRITON_CACHE_DIR="${SENTINEL_REPAIR_ROOT}/.cache/triton"
export CUDA_CACHE_PATH="${SENTINEL_REPAIR_ROOT}/.cache/cuda"
export NUMBA_CACHE_DIR="${SENTINEL_REPAIR_ROOT}/.cache/numba"
export VLLM_CACHE_ROOT="${SENTINEL_REPAIR_ROOT}/.cache/vllm"
export VLLM_CONFIG_ROOT="${SENTINEL_REPAIR_ROOT}/.cache/vllm-config"
export RAY_TMPDIR="${SENTINEL_REPAIR_TMP}/ray"
export MPLCONFIGDIR="${SENTINEL_REPAIR_ROOT}/.cache/matplotlib"
export NLTK_DATA="${SENTINEL_REPAIR_ROOT}/.cache/nltk"
export IPYTHONDIR="${SENTINEL_REPAIR_ROOT}/.cache/ipython"
export JUPYTER_CONFIG_DIR="${SENTINEL_REPAIR_ROOT}/.cache/jupyter"

export WANDB_MODE=disabled
export WANDB_DIR="${SENTINEL_REPAIR_ROOT}/runs/wandb"
export WANDB_CACHE_DIR="${SENTINEL_REPAIR_ROOT}/.cache/wandb"
export WANDB_CONFIG_DIR="${SENTINEL_REPAIR_ROOT}/.cache/wandb-config"
export WANDB_DATA_DIR="${SENTINEL_REPAIR_ROOT}/.cache/wandb-data"
export WANDB_ARTIFACT_DIR="${SENTINEL_REPAIR_ROOT}/runs/wandb-artifacts"

export TMPDIR="${SENTINEL_REPAIR_TMP}"
export TMP="${SENTINEL_REPAIR_TMP}"
export TEMP="${SENTINEL_REPAIR_TMP}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUSERBASE="${SENTINEL_REPAIR_ROOT}/.local/python"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_TERMINAL_PROMPT=0

export PYTHONPATH="${SENTINEL_REPAIR_ROOT}/src"
export PATH="${SENTINEL_REPAIR_VENV}/bin:${SENTINEL_REPAIR_ROOT}/.local/bin:${PATH}"

mkdir -p \
  "${SENTINEL_REPAIR_TMP}" \
  "${RAY_TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" \
  "${UV_CACHE_DIR}" \
  "${UV_PYTHON_INSTALL_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_ASSETS_CACHE}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${NUMBA_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_CONFIG_ROOT}" \
  "${MPLCONFIGDIR}" \
  "${NLTK_DATA}" \
  "${IPYTHONDIR}" \
  "${JUPYTER_CONFIG_DIR}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${WANDB_DATA_DIR}" \
  "${WANDB_ARTIFACT_DIR}" \
  "${SENTINEL_REPAIR_ROOT}/.local/bin" \
  "${SENTINEL_REPAIR_ROOT}/logs/slurm" \
  "${SENTINEL_REPAIR_ROOT}/runs/bootstrap" \
  "${SENTINEL_REPAIR_ROOT}/runs/slurm"

unset _sr_script_dir
