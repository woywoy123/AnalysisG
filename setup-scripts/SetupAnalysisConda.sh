#!/bin/bash 

source ~/.bashrc

cd ~
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
conda env remove -n GNN
conda create --name GNN python=3.9
conda activate GNN
conda install pip

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install pyg -c pyg
conda install -c conda-forge torchmetrics
conda install matplotlib
pip install networkx[all]
pip install uproot awkward
pip install h5py
pip install mplhep
pip install Cython


