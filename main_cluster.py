from main import Generate_Cache_Batches
from Closure.GNN import TrainEvaluate, GenerateTemplate
import os
import argparse

def GenerateTemplate_Batches(TruthLevel, Direct, OutputDir, OutputName):
    from Functions.IO.IO import PickleObject
    from Functions.IO.Files import WriteDirectory, Directories 
    
    File_Dumps = []
    for i in Direct:
        d = Directories(i)
        Files = d.ListFilesInDir(i)
        for f in Files:
            File_Dumps.append(i + "/" + f)
    
    try:
        os.mkdir(OutputDir)
    except FileExistsError:
        pass

    GenerateTemplate("", Tree = TruthLevel, Additional_Samples = File_Dumps, OutputName = OutputDir + "/" + OutputName, device = "cpu")

def MakeDirectory(directory):
    s = ""
    for i in directory.split("/"):
        s += "/" + i 
        try: 
            os.mkdir(s)
        except FileExistsError:
            pass

parse = argparse.ArgumentParser()
parse.add_argument("--Mode", type = str, help = "Choose between 'Cache'/'Train'/'DataLoader'")
parse.add_argument("--Model", type = str, help = "Specify the model you want to train. See Models.py")

parse.add_argument("--SampleDir", type = str, help = "Directory where Analysis .root files are stored.")
parse.add_argument("--CompilerName", type = str, help = "Specify the name of the cache output.")
parse.add_argument("--OutputDir", type = str, help = "Specify the Analysis output root directory. This is where all files are going to be dumped.")

parse.add_argument("--DataLoaderTruthLevel", type = str, help = "This is used in conjunction with the --mode DataLoader option. This will dump event particles at the specified TruthLevel. See Graph.py")
parse.add_argument("--DataLoaderAddSamples", type = list, help = "Used in conjunction with --mode DataLoader option. Specify all the files from the Cache to be included.")
parse.add_argument("--DataLoaderName", type = str, help = "Used in conjunction with --mode DataLoader option. Name of the DataLoader directory being created.")

args = parse.parse_args()

if args.Mode == "Cache":
    Generate_Cache_Batches(args.SampleDir, SingleThread = True, Compiler = args.Compiler, Custom = True, CustomDirectory = args.OutputDir) 

elif args.Mode == "DataLoader":
    GenerateTemplate_Batches(args.DataLoaderTruthLevel, args.DataLoaderAddSamples, args.OutputDir + "/DataLoaders", args.DataLoaderName) 




    #Generate_Cache_Batches("/CERN/CustomAnalysisTopOutput/SingleTop/Merger", Compiler = "t", Custom = True, CustomDirectory = SampleRootDir)
    #Generate_Cache_Batches("/CERN/CustomAnalysisTopOutput/ttbar/Merger", Compiler = "tt", Custom = True, CustomDirectory = SampleRootDir)
    #Generate_Cache_Batches("/CERN/CustomAnalysisTopOutput/tttt_1500GeV/Merger_A", Compiler = "tttt_1500", Custom = True, CustomDirectory = "/CERN/BSM4tops-GNN-Samples")
    #GenerateTemplate_Batches("TruthTopChildren", [Cached[0]], "/CERN/BSM4tops-GNN-Samples/DataLoaders", "tttt_1500_GeV") 
    #GenerateTemplate_Batches("TruthTopChildren", Cached, "/CERN/BSM4tops-GNN-Samples/DataLoaders", "Mixed_with_1500_GeV") 
 





exit()


if __name__ == '__main__':

    # ====== Evaluation of Models on Clustered environment ======== #
    # Parameters 
    SampleRootDir = "/CERN/BSM4tops-GNN-Samples"
    Cached = [SampleRootDir+"/tttt_1500_Cache", SampleRootDir+"/t_Cache", SampleRootDir + "/tt_Cache"]
    OutputModels = "/CERN/BSM4tops-GNN-Samples/TrainedModels"
    name = "TruthTopChildrenInvMassNode-ONLY_tttt_1500GeV"

   
    MakeDirectory(OutputModels + "/" + name)
    TrainEvaluate("InvMassNode", name, "/CERN/BSM4tops-GNN-Samples/DataLoaders/tttt_1500_GeV", ClusterOutputDir = "/CERN/BSM4tops-GNN-Samples/TrainedModels")

