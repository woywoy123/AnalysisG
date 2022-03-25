#!/bin/bash

function CreateSubmit ()
{
  echo "#!/bin/bash" >> Spawn.sh
  echo 'eval "$(conda shell.bash hook)"' >> Spawn.sh
  echo "conda init bash" >> Spawn.sh
  echo "conda activate GNN" >> Spawn.sh
  echo "cd ../" >> Spawn.sh
  echo 'echo $PWD' >> Spawn.sh
  echo "python main.py" >> Spawn.sh

  echo "executable = Spawn.sh" >> GNN.submit
  echo "error = results.error.$""(ClusterID)" >> GNN.submit
  echo 'Requirements = OpSysAndVer == "CentOS7"' >> GNN.submit
  echo "Request_GPU = 1" >> GNN.submit
  echo "+RequestRuntime = $((60*$1))" >> GNN.submit
  echo "queue 1" >> GNN.submit
}

rm Spawn.sh
rm GNN.submit
CreateSubmit 10

condor_submit GNN.submit
