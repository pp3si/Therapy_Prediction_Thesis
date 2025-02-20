#!/bin/bash --login

#SBATCH --time 17:00:00
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=57344M   # memory per CPU core
#SBATCH --array 1-264

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

infile="config_${SLURM_ARRAY_TASK_ID}.in"

mamba activate PLACEHOLDER_ENVIRONMENT

python3 -u LSTM_config_runner.py --infile "$infile" --datafile "OQ_lists_new.pkl"