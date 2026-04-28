#!/bin/bash
export CORES_PER_PROCESS=38
SP_SIZE=4

BATCH_SIZE_PER_PROCESS=1
export MPI_CORE_OFFSET=32
export TOKENIZERS_PARALLELISM=false
rank=$((OMPI_COMM_WORLD_RANK))

t_rank=$((rank%16/4))
LIB_PATH=~/envs/die_env/0409/die_$t_rank/lib
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/envs/die_env/0409/die_${t_rank}/bgemm:$LD_LIBRARY_PATH
export ASYNC_GRAD_MAX_PENDING_BYTES=$((1024 * 1024 * 1024))
export ASYNC_DP_MAX_PENDING_BYTES=$((1024 * 1024 * 1024))
export COPY_ENGINE_MAX_PENDING_BYTES=$((1024 * 1024 * 1024))

export ASYNC_RUNTIME_ENABLE=1
export COMM_MODE=0

export LD_PRELOAD=~/envs/die_env/0409/die_${t_rank}/libkm.so:~/envs/die_env/0409/die_${t_rank}/lib/libm.so.6:$LD_PRELOAD

be=$((rank%16* CORES_PER_PROCESS))
en=$((be+ CORES_PER_PROCESS - 1))


export OMP_NUM_THREADS=32
START_CORE=$((rank* CORES_PER_PROCESS % 608))
OMP_PLACES_LIST=""
for ((i=0; i<CORES_PER_PROCESS; i++)); do
    CORE_ID=$((START_CORE + i))
    if ((CORE_ID % 38 >= 32)); then
        continue
    fi
    
    if [ -z "$OMP_PLACES_LIST" ]; then
        OMP_PLACES_LIST="{${CORE_ID}}"
    else
        OMP_PLACES_LIST="${OMP_PLACES_LIST},{${CORE_ID}}"
    fi
done
benuma=$(((rank % 16) / 4 * 4))
ennuma=$((benuma + 3))
hbm_numa=$((rank%16+16))
if ((rank < 16)); then
    echo $be-$en
    echo $OMP_PLACES_LIST
    echo "numactl -m $benuma-$ennuma,$hbm_numa"
fi

OMP_PLACES="$OMP_PLACES_LIST"
export OMP_PLACES
export KML_BLAS_THREAD_TYPE=OMP
export OMP_PROC_BIND=spread
export OMP_MAX_ACTIVE_LEVELS=3
export BLAS_NUM_THREADS=1
export OMP_WAIT_POLICY=passive
numactl -m $benuma-$ennuma,$hbm_numa \
    taskset -c $be-$en \
    python train.py \
    --pretrained_model_name_or_path ./z_image \
    --work_dir ./results \
    --seed 3407 \
    --train_batch_size $BATCH_SIZE_PER_PROCESS \
    --dataloader_num_workers 0 \
    --max_train_steps 1000  \
    --data_root HARP/data/index.json \
    --sp_size $SP_SIZE \
    --data_root  ~/datasets/s2_test
