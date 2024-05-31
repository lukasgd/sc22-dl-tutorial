#!/bin/bash
##SBATCH -p nvgpu
#SBATCH -A csstaff
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J pm-crop64-tensorboard
#SBATCH -o logs/%x-%j.out

environment=$(realpath env/ngc-sc22-dl-tutorial-24.05.toml)

LOGDIR=logs

set -x
srun -ul --environment=${environment} \
    bash -c "
    if [ \"\${SLURM_PROCID:-0}\" -eq 0 ]; then
        echo \"Launching tensorboard...
Use port-forwarding on your client machine with
  ssh -N -L 6006:localhost:6006 \$(hostname)
and open the browser at http://localhost:6006/.\"
        tensorboard --logdir ${LOGDIR} --port 6006
    fi
    "
