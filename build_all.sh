#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAKE_BIN="${MAKE_BIN:-make}"

run_in_dir() {
    local rel_dir="$1"
    local step_name="$2"
    shift 2

    echo
    echo "==> ${step_name}"
    (
        cd "${SCRIPT_DIR}/${rel_dir}"
        "$@"
    )
}

mkdir -p "${SCRIPT_DIR}/kernels"

run_in_dir "accumulate_sp" "Compiling accumulate_sp" bash ./compile.sh
run_in_dir "hmpi" "Compiling hmpi" bash ./compile.sh
run_in_dir "mem" "Compiling mem" bash ./compile.sh
run_in_dir "gemm/bgemm" "Building bgemm" "${MAKE_BIN}"
run_in_dir "gemm/ring_attention" "Building ring_attention" env PYTHON_BIN="${PYTHON_BIN}" "${MAKE_BIN}"
run_in_dir "." "Building async runtime" "${PYTHON_BIN}" build.py

echo
echo "Build completed successfully."
