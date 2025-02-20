#!/bin/bash --login

#SBATCH --time 5:00:00 #Was 14:30
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=49152M   # memory per CPU core
# #sbatchh --array 1-264

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

mamba activate PLACEHOLDER_ENVIRONMENT

python3 -u new_cleaning_training.py