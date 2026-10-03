#!/usr/bin/env bash
# Create the fully local Python environment and exact shallow upstream checkout.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${script_dir}/env.sh"

readonly upstream_url="https://github.com/eth-sri/llm-verifier-noise.git"
readonly upstream_commit="913e3bf642747d9e178775a8a84f0e59fc9567f0"
readonly python_version="3.12.12"
readonly uv_version="0.12.9"
readonly uv_bin="${SENTINEL_REPAIR_ROOT}/.local/bin/uv"

if [[ ! -x "${uv_bin}" ]]; then
  installer="$(mktemp "${SENTINEL_REPAIR_TMP}/uv-install.XXXXXX")"
  cleanup_installer() {
    rm -f -- "${installer}"
  }
  trap cleanup_installer EXIT
  curl --fail --location --silent --show-error \
    "https://astral.sh/uv/${uv_version}/install.sh" \
    --output "${installer}"
  UV_UNMANAGED_INSTALL="${SENTINEL_REPAIR_ROOT}/.local/bin" sh "${installer}"
  cleanup_installer
  trap - EXIT
fi

actual_uv_version="$("${uv_bin}" --version)"
if [[ "${actual_uv_version}" != "uv ${uv_version}"* ]]; then
  echo "Unexpected local uv version: ${actual_uv_version}; expected uv ${uv_version}." >&2
  exit 1
fi

mkdir -p -- "$(dirname -- "${SENTINEL_REPAIR_UPSTREAM}")"
if [[ ! -d "${SENTINEL_REPAIR_UPSTREAM}/.git" ]]; then
  if [[ -e "${SENTINEL_REPAIR_UPSTREAM}" ]]; then
    echo "Refusing to replace non-Git path: ${SENTINEL_REPAIR_UPSTREAM}" >&2
    exit 1
  fi
  git init --quiet "${SENTINEL_REPAIR_UPSTREAM}"
  git -C "${SENTINEL_REPAIR_UPSTREAM}" remote add origin "${upstream_url}"
  git -C "${SENTINEL_REPAIR_UPSTREAM}" fetch --depth 1 origin "${upstream_commit}"
  git -C "${SENTINEL_REPAIR_UPSTREAM}" -c advice.detachedHead=false \
    checkout --quiet --detach FETCH_HEAD
fi

actual_commit="$(git -C "${SENTINEL_REPAIR_UPSTREAM}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${upstream_commit}" ]]; then
  echo "Unexpected upstream commit: ${actual_commit}" >&2
  echo "Expected: ${upstream_commit}" >&2
  exit 1
fi
actual_remote="$(git -C "${SENTINEL_REPAIR_UPSTREAM}" remote get-url origin)"
if [[ "${actual_remote}" != "${upstream_url}" ]]; then
  echo "Unexpected upstream remote: ${actual_remote}" >&2
  exit 1
fi

"${script_dir}/apply_upstream_patch.sh"

"${uv_bin}" python install "${python_version}"
if [[ ! -x "${SENTINEL_REPAIR_VENV}/bin/python" ]]; then
  "${uv_bin}" venv --python "${python_version}" --seed "${SENTINEL_REPAIR_VENV}"
fi

if ! actual_python="$("${SENTINEL_REPAIR_VENV}/bin/python" -c \
  'import platform; print(platform.python_version())' 2>/dev/null)"; then
  echo "Existing environment has no runnable Python: ${SENTINEL_REPAIR_VENV}" >&2
  exit 1
fi
if [[ "${actual_python}" != "${python_version}" ]]; then
  echo "Existing environment uses Python ${actual_python}; expected ${python_version}." >&2
  echo "Move ${SENTINEL_REPAIR_VENV} aside and rerun; this script will not overwrite it." >&2
  exit 1
fi

cd -- "${SENTINEL_REPAIR_ROOT}"
"${uv_bin}" sync \
  --locked \
  --python "${SENTINEL_REPAIR_VENV}/bin/python" \
  --extra test \
  --extra train

"${uv_bin}" --version > "${SENTINEL_REPAIR_ROOT}/runs/bootstrap/uv-version.txt"
"${SENTINEL_REPAIR_VENV}/bin/python" --version \
  > "${SENTINEL_REPAIR_ROOT}/runs/bootstrap/python-version.txt" 2>&1
printf '%s\n' "${actual_commit}" \
  > "${SENTINEL_REPAIR_ROOT}/runs/bootstrap/upstream-commit.txt"
"${uv_bin}" pip freeze --python "${SENTINEL_REPAIR_VENV}/bin/python" \
  > "${SENTINEL_REPAIR_ROOT}/runs/bootstrap/requirements.freeze.txt"

echo "Environment ready: ${SENTINEL_REPAIR_VENV}"
echo "Upstream pinned: ${actual_commit}"
