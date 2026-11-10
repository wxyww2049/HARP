#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source ../run/env.sh
"$PYTHON_BIN" setup.py build_ext --build-lib ../kernels
