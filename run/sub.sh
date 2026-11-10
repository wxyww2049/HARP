#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
nodes=${1:?Usage: bash run/sub.sh N [train.py options]}
shift
source run/env.sh
mkdir -p rankfiles
rankfile=$(mktemp "$PWD/rankfiles/run.XXXXXXXX")
trap 'rm -f "$rankfile"' EXIT
"$PYTHON_BIN" gen_rankfile/gen_rankfile.py "${CCS_ALLOC_FILE:?Set CCS_ALLOC_FILE to a Donau allocation file}" "$nodes" > "$rankfile"
numrank=$(wc -l < "$rankfile")
export TORCH_SHM_ID=$(cat /proc/sys/kernel/random/uuid)
export MASTER_ADDR=$(awk -F '[= ]' 'NR == 1 {print $3}' "$rankfile")
export MASTER_PORT=${MASTER_PORT:-39503}
export HARP_USE_HBM=${HARP_USE_HBM:-1}
mpirun -np "$numrank" --rankfile "$rankfile" --wdir "$PWD" \
    --mca rmaps_rank_file_physical true --mca coll ^ucg \
    --mca coll_tuned_use_dynamic_rules true --mca plm_rsh_agent /usr/bin/ssh \
    -x HARP_USE_HBM -x PATH -x LD_LIBRARY_PATH -x PYTHON_BIN -x OPAL_PREFIX -x TORCH_SHM_ID \
    -x MASTER_ADDR -x MASTER_PORT -x UCX_TLS=rc -x UCX_RNDV_THRESH=512K \
    -x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y -x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
    bash run/train.sh "$@"
