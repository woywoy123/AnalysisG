#!/bin/bash

cd ..
source scripts/gnn-analysis/bin/activate
mkdir -p build
cd build 

#pip install Cython
#pip install boost-histogram
#pip install mplhep
#pip install pyyaml
#pip install tqdm
#pip install pwinput
#pip install scipy
#pip install h5py
#pip install scikit-learn

cmake .. 
make -j4
cmake ..
