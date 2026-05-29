#!/bin/bash

python3 -m venv gnn-analysis
source $PWD/gnn-analysis/bin/activate
cd pyAMI_atlas
pip install .
cd ..

pip install boost_histogram
pip install mplhep
pip install pwinput
pip install tqdm
pip install scipy
pip install h5py
pip install scikit-learn
pip install pyyaml
pip install cython
pip install pyAMI-core


