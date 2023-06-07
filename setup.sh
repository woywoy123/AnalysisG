#!/bin/bash 

#export CUDA_PATH=/usr/local/cuda-11.8
#export VERSION=cpu
#export TORCH=1.13.0
#export MAX_JOBS=12
#export CC=gcc-6.20
#export CXX=gcc-6.20

echo "y" | ami_atlas_post_install
pip install torch==${TORCH} --extra-index-url https://download.pytorch.org/whl/${VERSION}
pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH}+${VERSION}.html
pip install torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${VERSION}.html
pip install torch_cluster -f https://data.pyg.org/whl/torch-${TORCH}+${VERSION}.html

pip install .
cd torch-extensions 
pip install . 
