#!/bin/bash
#SBATCH -p gpu-v100-preempt
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH -o "logs/slurm-%N.out"
#SBATCH --array=0-65

srun ./main.sh "V100" "${SLURM_ARRAY_TASK_ID}" 
