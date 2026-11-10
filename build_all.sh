#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run/env.sh"
# The site PyTorch was linked against the previous KBLAS SONAME.
legacy_blas="${TORCH_KBLAS_LIB:-/work_ssd/software/HPCKit/25.1.0.SPC001/kml/bisheng/lib/sme/kblas/multi/libkblas.so}"
if [[ -f "$legacy_blas" ]]; then
    mkdir -p "$SCRIPT_DIR/kernels/compat"
    ln -sfn "$legacy_blas" "$SCRIPT_DIR/kernels/compat/libkblas_armv9p_v25.0.0.so"
fi
export MAX_JOBS="${MAX_JOBS:-4}"
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
