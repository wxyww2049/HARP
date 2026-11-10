#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
job=${1:?Usage: bash run/watch.sh JOB_ID}
tail -F "e2e-logs/out-$job" "e2e-logs/err-$job"
