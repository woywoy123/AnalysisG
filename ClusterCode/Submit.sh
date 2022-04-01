#!/bin/bash

function CreateShellScript ()
{
  local str=$1
  echo "#!/bin/bash" > Spawn.sh
  echo "source ~/.bashrc" >> Spawn.sh
  echo 'eval "$(conda shell.bash hook)"' >> Spawn.sh
  echo "conda activate GNN" >> Spawn.sh
  echo "cd ../" >> Spawn.sh
  echo "$1 >> ClusterCode/log.txt" >> Spawn.sh
}

function CreateSubmitGPU()
{
  local str=$1
  echo "executable = $1.sh" > $1.submit
  echo "error = results.error.$""(ClusterID)" >> $1.submit
  echo 'Requirements = OpSysAndVer == "CentOS7"' >> $1.submit
  echo "Request_GPUs = 1" >> $1.submit
  echo "+RequestRuntime = $((60*60*$1))" >> $1.submit
  echo "+Request_Memory = 1024" >> $1.submit
  echo "queue" >> $1.submit
}

function CreateSubmitCPU()
{
  local str=$1
  echo "executable = $1.sh" > $1.submit
  echo "error = results.error.$""(ClusterID)" >> $1.submit
  echo 'Requirements = OpSysAndVer == "CentOS7"' >> $1.submit
  echo "Request_Cpus = 1" >> $1.submit
  echo "+RequestRuntime = $((60*60*$1))" >> $1.submit
  echo "+Request_Memory = 4096" >> $1.submit
  echo "queue" >> $1.submit
}




CreateShellScript "python main_cluster.py --Mode Cache "
chmod +x Spawn.sh
condor_submit GNN.submit
