# SZSC AArch64 compute-node environment. Source from bash.
harp_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export HPCKIT_ROOT=${HPCKIT_ROOT:-/work_ssd/software/HPCKit/25.2.1}
export HMPI_ROOT=${HMPI_ROOT:-$HPCKIT_ROOT/hmpi/bisheng/release/hmpi}
export KBLAS_INCLUDE=${KBLAS_INCLUDE:-$HPCKIT_ROOT/kml/bisheng/include}
export KBLAS_LIBDIR=${KBLAS_LIBDIR:-$HPCKIT_ROOT/kml/bisheng/lib/sme/kblas/multi}
export HARP_PYTHON_ENV=${HARP_PYTHON_ENV:-$HOME/miniconda3/envs/DiT_DNN}
export PATH="$HPCKIT_ROOT/compiler/bisheng/bin:$HMPI_ROOT/bin:$HARP_PYTHON_ENV/bin:$PATH"
export OPAL_PREFIX="$HMPI_ROOT"
export LD_LIBRARY_PATH="$harp_root/kernels:$harp_root/kernels/compat:$KBLAS_LIBDIR:$HPCKIT_ROOT/compiler/bisheng/lib:$HMPI_ROOT/lib:$HMPI_ROOT/../hucx/lib:$HMPI_ROOT/../xucg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC=${CC:-clang} CXX=${CXX:-clang++}
export PYTHON_BIN=${PYTHON_BIN:-python}
