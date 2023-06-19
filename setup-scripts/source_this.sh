#!/bin/bash
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "gcc gcc620_x86_64_slc6"
export PythonGNN=/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/AnalysisG/setup-scripts/PythonGNN/bin/activate
alias GNN='source /home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/AnalysisG/setup-scripts/PythonGNN/bin/activate'
source $PythonGNN
