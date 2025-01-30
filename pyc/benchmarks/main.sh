#!/bin/bash

source ~/.bashrc
module load CUDA/12.4.1

if [ "$1" == "H100" ]
then
    conda activate gnn-h100
    python h100_main.py $2
elif [ "$1" == "V100" ]
then
    conda activate gnn-v100
    python v100_main.py $2
elif [ "$1" == "A100" ]
then
    conda activate gnn-a100
    python a100_main.py $2
fi
