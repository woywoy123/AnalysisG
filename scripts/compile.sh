#!/bin/bash

cd ..
source scripts/gnn-analysis/bin/activate
cd build 
cmake .. 
make -j12 
cmake ..
