#!/bin/bash

function CreateShellScript ()
{
  local str=$1
  local name=$2
  mkdir $2
  echo "#!/bin/bash" > $name/Spawn.sh
  echo "source ~/.bashrc" >> $name/Spawn.sh
  echo 'eval "$(conda shell.bash hook)"' >> $name/Spawn.sh
  echo "conda activate GNN" >> $name/Spawn.sh
  echo "cd ../" >> $name/Spawn.sh
  echo "$str" >> $name/Spawn.sh
  chmod +x $name/Spawn.sh
  echo "JOB $name $name/$name.submit" >> SubmissionDAG.submit
}

function CreateSubmitGPU()
{
  local hours=$1
  local name=$2
  echo "executable = $name/Spawn.sh" > $name/$name.submit
  echo "error = results.error.$""(ClusterID)" >> $name/$name.submit
  echo 'Requirements = OpSysAndVer == "CentOS7"' >> $name/$name.submit
  echo "Request_GPUs = 4" >> $name/$name.submit
  echo "+RequestRuntime = $((60*60*$hours))" >> $name/$name.submit
  echo "+Request_Memory = 1024" >> $name/$name.submit
  echo "queue" >> $name/$name.submit
}

function CreateSubmitCPU()
{
  local hours=$1
  local name=$2
  local GB=4
  echo "executable = $name/Spawn.sh" > $name/$name.submit
  echo "error = results.error.$""(ClusterID)" >> $name/$name.submit
  echo 'Requirements = OpSysAndVer == "CentOS7"' >> $name/$name.submit
  echo "Request_Cpus = 1" >> $name/$name.submit
  echo "+RequestRuntime = $((60*60*$hours))" >> $name/$name.submit
  echo "+Request_Memory = $((1024*$GB))" >> $name/$name.submit
  echo "queue" >> $name/$name.submit
}

function CreateEdge()
{
  echo "PARENT $1 CHILD $2" >> SubmissionDAG.submit
}
echo "" > SubmissionDAG.submit


sampleDir="/nfs/dust/atlas/user/woywoy12/BSM4tops-GNN-Samples"
outDir="/nfs/dust/atlas/user/woywoy12/BSM4tops-GNN-Output"

## ============ Cache Builder ============ ##
#CreateShellScript "python main_cluster.py --Mode Cache --SampleDir $sampleDir/tttt/OldSample/1500_GeV/MCe --CompilerName tttt_1500GeV --OutputDir $outDir/Cache" 'tttt_1500GeV_Cache'
#CreateSubmitCPU 6 'tttt_1500GeV_Cache'
#
#CreateShellScript "python main_cluster.py --Mode Cache --SampleDir $sampleDir/ttbar/MCa --CompilerName ttbar --OutputDir $outDir/Cache" 'ttbar_Cache'
#CreateSubmitCPU 6 'ttbar_Cache'
#
#CreateShellScript "python main_cluster.py --Mode Cache --SampleDir $sampleDir/t/MCa --CompilerName t --OutputDir $outDir/Cache" 't_Cache'
#CreateSubmitCPU 6 't_Cache'
#
CreateShellScript "python main_cluster.py --Mode Cache --SampleDir $sampleDir/Zmumu/MCd --CompilerName Zmumu --OutputDir $outDir/Cache" 'Zmumu_Cache'
CreateSubmitCPU 6 'Zmumu_Cache'

## ============ DataLoader Builder ============ ##
# +++++++> Truth Top Children 
#CreateShellScript "python main_cluster.py --Mode DataLoader --DataLoaderTruthLevel TruthTopChildren --OutputDir $outDir --DataLoaderAddSamples $outDir/Cache/tttt_1500GeV_Cache --DataLoaderName TTC_tttt_1500GeV" 'TTC_tttt_1500GeV_Data'
#CreateSubmitCPU 12 'TTC_tttt_1500GeV_Data'
#CreateEdge "tttt_1500GeV_Cache" "TTC_tttt_1500GeV_Data"
#CreateEdge "ttbar_Cache" "TTC_tttt_1500GeV_Data"
#CreateEdge "t_Cache" "TTC_tttt_1500GeV_Data"
#
#CreateShellScript "python main_cluster.py --Mode DataLoader --DataLoaderTruthLevel TruthTopChildren --OutputDir $outDir --DataLoaderAddSamples $outDir/Cache/tttt_1500GeV_Cache $outDir/Cache/ttbar_Cache $outDir/Cache/t_Cache --DataLoaderName TTC_Mixed_1500GeV" 'TTC_Mixed_1500GeV_Data'
#CreateSubmitCPU 12 'TTC_Mixed_1500GeV_Data'
#CreateEdge "tttt_1500GeV_Cache" "TTC_Mixed_1500GeV_Data"
#CreateEdge "ttbar_Cache" "TTC_Mixed_1500GeV_Data"
#CreateEdge "t_Cache" "TTC_Mixed_1500GeV_Data"

## +++++++> Truth Jets 
#CreateShellScript "python main_cluster.py --Mode DataLoader --DataLoaderTruthLevel TruthJetsLep --OutputDir $outDir --DataLoaderAddSamples $outDir/Cache/tttt_1500GeV_Cache --DataLoaderName TJL_tttt_1500GeV" 'TJL_tttt_1500GeV_Data'
#CreateSubmitCPU 12 'TJL_tttt_1500GeV_Data'
##
#CreateShellScript "python main_cluster.py --Mode DataLoader --DataLoaderTruthLevel TruthJetsLep --OutputDir $outDir --DataLoaderAddSamples $outDir/Cache/tttt_1500GeV_Cache $outDir/Cache/ttbar_Cache $outDir/Cache/t_Cache --DataLoaderName TJL_Mixed_1500GeV" 'TJL_Mixed_1500GeV_Data'
#CreateSubmitCPU 12 'TJL_Mixed_1500GeV_Data'
##
### +++++++> Jets 
#CreateShellScript "python main_cluster.py --Mode DataLoader --DataLoaderTruthLevel JetsLep --OutputDir $outDir --DataLoaderAddSamples $outDir/Cache/tttt_1500GeV_Cache --DataLoaderName JL_tttt_1500GeV" 'JL_tttt_1500GeV_Data'
#CreateSubmitCPU 12 'JL_tttt_1500GeV_Data'
##
#CreateShellScript "python main_cluster.py --Mode DataLoader --DataLoaderTruthLevel JetsLep --OutputDir $outDir --DataLoaderAddSamples $outDir/Cache/tttt_1500GeV_Cache $outDir/Cache/ttbar_Cache $outDir/Cache/t_Cache --DataLoaderName JL_Mixed_1500GeV" 'JL_Mixed_1500GeV_Data'
#CreateSubmitCPU 12 'JL_Mixed_1500GeV_Data'

### =============== Model Runners ================ ##
## ===> InvMass
## +++++++++++++> InvMassNode - 1500 GeV
#CreateShellScript "python main_cluster.py --Mode Train --Model InvMassNode --ModelName TTC_tttt_1500GeV_InvMassNode --ModelDataLoaderInput $outDir/DataLoaders/TTC_tttt_1500GeV.pkl --ModelOutputDir $outDir/TrainedModels" "TTC_tttt_1500GeV_InvMassNode"
#CreateSubmitGPU 12 "TTC_tttt_1500GeV_InvMassNode"
#
## +++++++++++++> InvMassEdge - 1500 GeV
#CreateShellScript "python main_cluster.py --Mode Train --Model InvMassEdge --ModelName TTC_tttt_1500GeV_InvMassEdge --ModelDataLoaderInput $outDir/DataLoaders/TTC_tttt_1500GeV.pkl --ModelOutputDir $outDir/TrainedModels" "TTC_tttt_1500GeV_InvMassEdge"
#CreateSubmitGPU 12 "TTC_tttt_1500GeV_InvMassEdge"
#
## +++++++++++++> InvMassNode - Mixed + 1500 GeV
#CreateShellScript "python main_cluster.py --Mode Train --Model InvMassNode --ModelName TTC_Mixed_1500GeV_InvMassNode --ModelDataLoaderInput $outDir/DataLoaders/TTC_Mixed_1500GeV.pkl --ModelOutputDir $outDir/TrainedModels" "TTC_Mixed_1500GeV_InvMassNode"
#CreateSubmitGPU 12 "TTC_Mixed_1500GeV_InvMassNode"
#
## +++++++++++++> InvMassEdge - Mixed + 1500 GeV
#CreateShellScript "python main_cluster.py --Mode Train --Model InvMassEdge --ModelName TTC_Mixed_1500GeV_InvMassEdge --ModelDataLoaderInput $outDir/DataLoaders/TTC_Mixed_1500GeV.pkl --ModelOutputDir $outDir/TrainedModels" "TTC_Mixed_1500GeV_InvMassEdge"
#CreateSubmitGPU 12 "TTC_Mixed_1500GeV_InvMassEdge"
#
## ===> PathNets
## +++++++++++++> PathNetNode
#CreateShellScript "python main_cluster.py --Mode Train --Model PathNetNode --ModelName TTC_tttt_1500GeV_PathNetNode --ModelDataLoaderInput $outDir/DataLoaders/TTC_tttt_1500GeV.pkl --ModelOutputDir $outDir/TrainedModels" "TTC_tttt_1500GeV_PathNetNode"
#CreateSubmitGPU 12 "TTC_tttt_1500GeV_PathNetNode"
#
## +++++++++++++> PathNetEdge
#CreateShellScript "python main_cluster.py --Mode Train --Model PathNetEdge --ModelName TTC_tttt_1500GeV_PathNetEdge --ModelDataLoaderInput $outDir/DataLoaders/TTC_tttt_1500GeV.pkl --ModelOutputDir $outDir/TrainedModels" "TTC_tttt_1500GeV_PathNetEdge"
#CreateSubmitGPU 12 "TTC_tttt_1500GeV_PathNetEdge"
#
## +++++++++++++> PathNetNode - Mixed + 1500 GeV
#CreateShellScript "python main_cluster.py --Mode Train --Model PathNetNode --ModelName TTC_Mixed_1500GeV_PathNetNode --ModelDataLoaderInput $outDir/DataLoaders/TTC_Mixed_1500GeV.pkl --ModelOutputDir $outDir/TrainedModels" "TTC_Mixed_1500GeV_PathNetNode"
#CreateSubmitGPU 12 "TTC_Mixed_1500GeV_PathNetNode"
#
## +++++++++++++> PathNetEdge - Mixed + 1500 GeV
#CreateShellScript "python main_cluster.py --Mode Train --Model PathNetEdge --ModelName TTC_Mixed_1500GeV_PathNetEdge --ModelDataLoaderInput $outDir/DataLoaders/TTC_Mixed_1500GeV.pkl --ModelOutputDir $outDir/TrainedModels" "TTC_Mixed_1500GeV_PathNetEdge"
#CreateSubmitGPU 12 "TTC_Mixed_1500GeV_PathNetEdge"
#
#
#CreateEdge "TTC_tttt_1500GeV_Data" "TTC_tttt_1500GeV_InvMassNode"
#CreateEdge "TTC_tttt_1500GeV_Data" "TTC_tttt_1500GeV_InvMassEdge"
#CreateEdge "TTC_Mixed_1500GeV_Data" "TTC_Mixed_1500GeV_InvMassNode"
#CreateEdge "TTC_Mixed_1500GeV_Data" "TTC_Mixed_1500GeV_InvMassEdge"
#CreateEdge "TTC_tttt_1500GeV_Data" "TTC_tttt_1500GeV_PathNetNode"
#CreateEdge "TTC_tttt_1500GeV_Data" "TTC_tttt_1500GeV_PathNetEdge"
#CreateEdge "TTC_Mixed_1500GeV_Data" "TTC_Mixed_1500GeV_PathNetNode"
#CreateEdge "TTC_Mixed_1500GeV_Data" "TTC_Mixed_1500GeV_PathNetEdge"
