#!/bin/bash

cd ..
source scripts/gnn-analysis/bin/activate
cd build 
make -j12  && cmake .. > log.txt
