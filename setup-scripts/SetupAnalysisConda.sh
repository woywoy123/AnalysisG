#!/bin/bash 

source ~/.bashrc

cd ~
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
conda env remove -n GNN
conda create --name GNN python=3.10
conda activate GNN
conda install --yes -c conda-forge root
conda install --yes pip

#conda install --yes pytorch==1.13 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
#conda install --yes pyg -c pyg
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install --yes -c conda-forge torchmetrics
conda install --yes matplotlib
conda install pyg -c pyg

pip install networkx[all]
pip install uproot awkward
pip install h5py
pip install mplhep
pip install Cython
