N=$1
dsub -nn $N -n D2AR-$N -rpn 4 -R "cpu=152" -oo e2e-logs/out-$N -eo e2e-logs/err-$N  --mpi hmpi -q q_queue "bash run/sub.sh $N"
bash run/watch.sh $N