#!/bin/bash 

source ~/.bashrc
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "gcc gcc620_x86_64_slc6"
lsetup "python 3.9.14-x86_64-centos7"

python3 -m venv PythonGNN
export PATH=/usr/bin/gcc-11
source ./PythonGNN/bin/activate
 
# Packages to be installed.
pip install --upgrade pip
pip install setuptools -U
pip install Cython
pip install matplotlib  
pip install networkx
pip install uproot awkward
pip install h5py
pip install mplhep
pip install -U scikit-learn
pip install Cython
pip install vector
pip install tqdm
pip install torch torchvision torchaudio 

ver=$(python3 -c "import torch; print(torch.__version__)")
pip install pyg_lib -f https://data.pyg.org/whl/torch-$ver+cu117.html
pip install torch_scatter -f https://data.pyg.org/whl/torch-$ver+cu117.html
pip install torch_sparse -f https://data.pyg.org/whl/torch-$ver+cu117.html
pip install torch_cluster -f https://data.pyg.org/whl/torch-$ver+cu117.html
pip install torch_spline_conv -f https://data.pyg.org/whl/torch-$ver+cu117.html
pip install torch_geometric -f https://data.pyg.org/whl/torch-$ver+cu117.html

cd ../
pip install .
#
#cd torch-extensions
#pip install .
#
#echo "export PythonGNN=$PWD/setup-scripts/PythonGNN/bin/activate" >> ~/.bashrc

