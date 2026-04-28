#!/bin/bash
N=$1
set --

source ~/.bashrc_0409
source ~/miniconda3/bin/activate
conda activate DiT_DNN_6
perf_file=./sen/node_scores.txt
hostfile=./rankfiles/run-rankfile-$N
python gen_rankfile/gen_rankfile.py $CCS_ALLOC_FILE $N > $hostfile
numrank=$(wc -l < $hostfile)
masteraddr=$(grep '^rank 0=' $hostfile | awk -F'[= ]' '{print $3}')
echo $masteraddr

cleaned_up=0
export TORCH_SHM_ID=$(uuidgen)
mpirun 	--allow-run-as-root -np $numrank \
	--rankfile $hostfile \
	--mca coll ^ucg \
	--mca rmaps_rank_file_physical true \
	--mca coll_tuned_use_dynamic_rules true \
	--mca plm_rsh_agent /usr/bin/ssh \
	--mca grpcomm_direct_priority 100 \
	--mca coll_tuned_dynamic_rules_filename ~/shell/coll_tuned_default_decision_allreduce.conf \
    --mca io romio321 \
    -x UCX_TLS=rc \
	-x UCX_RNDV_THRESH=512K \
	-x PATH \
	-x LD_LIBRARY_PATH \
	-x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y \
	-x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
	-x MASTER_ADDR=$masteraddr \
   	-x MASTER_PORT=39503 \
	./run/train.sh 
