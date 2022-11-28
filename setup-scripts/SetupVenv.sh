#!/bin/bash 

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "gcc gcc620_x86_64_slc6"
lsetup "python 3.9.14-x86_64-centos7"
python3 -m venv PythonGNN
source ./PythonGNN/bin/activate
 
# Packages to be installed.
pip3 install matplotlib  
pip3 install networkx[all]
pip3 install uproot awkward
pip3 install h5py
pip3 install mplhep
pip3 install -U scikit-learn

pip3 install torch torchvision torchaudio #torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
ver=$(python -c "import torch; print(torch.__version__)")
pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-$ver.html
pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-$ver.html
pip3 install torch-geometric
pip3 install torchmetrics

ver=$(echo "$ver" | cut -d'+' -f1)
pip3 install torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$ver+cpu.html

cd ../
python setup.py install

echo "export PythonGNN=$PWD/setup-scripts/PythonGNN/bin/activate" >> ~/.bashrc

