#!/bin/bash
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate GNN
cd ../
python main_cluster.py --Mode DataLoader --DataLoaderTruthLevel TruthTopChildren --DataLoaderAddSamples [$outDir/tttt_1500GeV] --DataLoaderName tttt_1500GeV
