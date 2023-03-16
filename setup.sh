#!/bin/bash 

export CUDA_PATH=/usr/local/cuda-11.7
export MAX_JOBS=12
export CC=gcc-11
export CXX=gcc-11
pip install .
cd torch-extensions 
pip install . 
