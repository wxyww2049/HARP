#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_KERNELS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)/kernels"

cd "${SCRIPT_DIR}"
taskset -c 10-17 python setup.py build_ext --inplace

mkdir -p "${ROOT_KERNELS_DIR}"

shopt -s nullglob
so_candidates=(
    "${SCRIPT_DIR}"/kernels/my_allocator*.so
    "${SCRIPT_DIR}"/kernels/my_allocator/*.so
    "${SCRIPT_DIR}"/my_allocator*.so
)
shopt -u nullglob

if [ "${#so_candidates[@]}" -eq 0 ]; then
    echo "No my_allocator .so found after build." >&2
    exit 1
fi

latest_so="$(ls -t -- "${so_candidates[@]}" | head -n 1)"
cp -f -- "${latest_so}" "${ROOT_KERNELS_DIR}/"
echo "Copied $(basename "${latest_so}") to ${ROOT_KERNELS_DIR}"
