#!/bin/bash 

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
Version="3.9.18-x86_64-centos7"
lsetup "python $Version"
python3.9 -m venv PythonGNN
source ./PythonGNN/bin/activate

echo "#!/bin/bash" > source_this.sh
echo "setupATLAS" >> source_this.sh
echo "lsetup 'gcc gcc620_x86_64_slc6'" >> source_this.sh
echo "export PythonGNN=$PWD/PythonGNN/bin/activate" >> source_this.sh
echo "alias GNN='source $PWD/PythonGNN/bin/activate'" >> source_this.sh
echo 'source $PythonGNN' >> source_this.sh

