#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
nodes=${1:?Usage: bash run/run.sh N [train.py options]}
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo "N must be a positive integer" >&2; exit 2; }
shift
mkdir -p e2e-logs
printf -v command '%q ' bash "$PWD/run/sub.sh" "$nodes" "$@"
dsub -nn "$nodes" -n "HARP-$nodes" -rpn 4 -R "cpu=152" \
    -oo "$PWD/e2e-logs/out-%J" -eo "$PWD/e2e-logs/err-%J" \
    --mpi hmpi -q "${HARP_QUEUE:-q_grapes}" "$command"
