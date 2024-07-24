#!/bin/bash

cd <path>
source .bashrc
conda activate analysis
cd <res-path>
python <src>main.py --config <config-file>
