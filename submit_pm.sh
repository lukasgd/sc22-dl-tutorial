#!/bin/bash 
#SBATCH -p nvgpu 
#SBATCH -A csstaff
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH --time=0:15:00
#SBATCH -J pm-crop64
#SBATCH -o logs/%x-%j.out

environment=$(realpath env/ngc-sc22-dl-tutorial-24.01.toml)

DATADIR=/iopsstor/scratch/cscs/lukasd/ds/tutorials/sc22_data
LOGDIR=logs
mkdir -p ${LOGDIR}
args="--expdir ${LOGDIR} --datadir ${DATADIR} ${@}"

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

BENCHY_CONFIG=benchy-conf.yaml
BENCHY_OUTPUT=${BENCHY_OUTPUT:-"benchy_output"}
sed "s/.*output_filename.*/        output_filename: ${BENCHY_OUTPUT}.json/" ${BENCHY_CONFIG} > benchy-run-${SLURM_JOBID}.yaml
export BENCHY_CONFIG_FILE=benchy-run-${SLURM_JOBID}.yaml
export MASTER_ADDR=$(hostname)

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
    echo "Enabling debugging..."
    ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
    if [ "${DEBUG_RANK:-0}" -ge "$SLURM_NTASKS" ]; then
        echo "DEBUG_RANK = ${DEBUG_RANK:-0} is not a valid rank (#ranks = $SLURM_NTASKS), exiting..."
        exit 1
    fi
else
    ENROOT_ENTRYPOINT=""
fi

set -x
srun -ul --environment=${environment} ${ENROOT_ENTRYPOINT} \
    bash -c "
    source export_DDP_vars.sh
    if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ] && [ \"\${SLURM_PROCID:-0}\" -eq ${DEBUG_RANK:-0} ]; then
        echo \"Running training script with debugpy on \$(hostname)\"
        DEBUG_CMD=\"-m debugpy --listen 5678 --wait-for-client\"
    else
        DEBUG_CMD=\"\"
    fi
    ${PROFILE_CMD} python \${DEBUG_CMD} train.py ${args}
    "
rm benchy-run-${SLURM_JOBID}.yaml
