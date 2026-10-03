#!/usr/bin/env bash
# Apply the ordered local-only patches, idempotently.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${script_dir}/env.sh"

readonly expected_commit="913e3bf642747d9e178775a8a84f0e59fc9567f0"
readonly -a patch_files=(
  "${SENTINEL_REPAIR_ROOT}/patches/0001-local-training.patch"
)

if [[ ! -d "${SENTINEL_REPAIR_UPSTREAM}/.git" ]]; then
  echo "Missing upstream checkout: ${SENTINEL_REPAIR_UPSTREAM}" >&2
  exit 1
fi
if [[ "$(git -C "${SENTINEL_REPAIR_UPSTREAM}" rev-parse HEAD)" != "${expected_commit}" ]]; then
  echo "Refusing to patch an unpinned upstream checkout." >&2
  exit 1
fi

for patch_file in "${patch_files[@]}"; do
  if git -C "${SENTINEL_REPAIR_UPSTREAM}" apply --check "${patch_file}" 2>/dev/null; then
    git -C "${SENTINEL_REPAIR_UPSTREAM}" apply "${patch_file}"
    echo "Applied $(basename -- "${patch_file}")."
  elif git -C "${SENTINEL_REPAIR_UPSTREAM}" apply --reverse --check "${patch_file}" 2>/dev/null; then
    echo "Already applied: $(basename -- "${patch_file}")."
  else
    echo "Patch neither applies nor reverses cleanly: ${patch_file}" >&2
    echo "The pinned upstream worktree has drifted." >&2
    exit 1
  fi
done
