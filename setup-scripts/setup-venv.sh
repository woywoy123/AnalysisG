#!/bin/bash 

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "gcc gcc620_x86_64_slc6"
lsetup "python 3.9.14-x86_64-centos7"
python3 -m venv PythonGNN
source ./PythonGNN/bin/activate
echo "export PythonGNN=$PWD/PythonGNN/bin/activate" >> ~/.bashrc
echo "alias GNN='source $PWD/PythonGNN/bin/activate'" >> ~/.bashrc

#export CUDA_PATH=/usr/local/cuda-11.8
export VERSION=cpu
export TORCH=1.13.0
export MAX_JOBS=12
#export CC=gcc-620
#export CXX=g++-620

cd ../
bash setup.sh


