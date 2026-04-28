N=$1
watch -n 1 "tail e2e-logs/err-$N e2e-logs/out-$N && cat e2e-logs/err-$N