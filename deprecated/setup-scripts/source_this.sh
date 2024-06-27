#!/bin/bash
setupATLAS
lsetup 'gcc gcc620_x86_64_slc6'
export PythonGNN=/home/tnom6927/Documents/Project/Analysis/bsm4tops-gnn-analysis/AnalysisG/setup-scripts/PythonGNN/bin/activate
alias GNN='source /home/tnom6927/Documents/Project/Analysis/bsm4tops-gnn-analysis/AnalysisG/setup-scripts/PythonGNN/bin/activate'
source $PythonGNN
