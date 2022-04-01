#!/bin/bash

function CreateSubmit ()
{
  echo "#!/bin/bash" > Spawn.sh
  echo "source ~/.bashrc" >> Spawn.sh
  echo 'eval "$(conda shell.bash hook)"' >> Spawn.sh
  echo "conda activate GNN" >> Spawn.sh
  echo "cd ../" >> Spawn.sh
  echo "python main.py >> ClusterCode/log.txt" >> Spawn.sh

  echo "executable = Spawn.sh" > GNN.submit
  echo "error = results.error.$""(ClusterID)" >> GNN.submit
  echo 'Requirements = OpSysAndVer == "CentOS7"' >> GNN.submit
  echo "Request_GPUs = 1" >> GNN.submit
  echo "+RequestRuntime = $((60*60*$1))" >> GNN.submit
  echo "queue 1" >> GNN.submit
}

CreateSubmit 2 
chmod +x Spawn.sh
condor_submit GNN.submit
