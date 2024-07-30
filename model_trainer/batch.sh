#!/bin/bash
#
#SBATCH --qos=<optional>
#SBATCH --nodes=1
#SBATCH --constraint=<gpu-name>
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --account=<optional>
#SBATCH --time=24:00:00
#SBATCH --array=1-<kfold>

srun python main.py --config <path-config>

