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
    except PermissionError:
        pass

    GenerateTemplate("", Level = TruthLevel, Additional_Samples = File_Dumps, OutputName = OutputDir + "/" + OutputName + ".pkl", device = "cpu")

def MakeDirectory(directory):
    s = ""
    for i in directory.split("/"):
        s += "/" + i 
        try: 
            os.mkdir(s)
        except FileExistsError:
            pass
        except PermissionError:
            pass

parse = argparse.ArgumentParser()
parse.add_argument("--Mode", type = str, help = "Choose between 'Cache'/'Train'/'DataLoader'")

parse.add_argument("--SampleDir", type = str, help = "Directory where Analysis .root files are stored.")
parse.add_argument("--CompilerName", type = str, help = "Specify the name of the cache output.")
parse.add_argument("--OutputDir", type = str, help = "Specify the Analysis output root directory. This is where all files are going to be dumped.")

parse.add_argument("--DataLoaderTruthLevel", type = str, help = "This is used in conjunction with the --mode DataLoader option. This will dump event particles at the specified TruthLevel. See Graph.py")
parse.add_argument("--DataLoaderAddSamples", type = str, nargs="+", help = "Used in conjunction with --mode DataLoader option. Specify all the files from the Cache to be included.")
parse.add_argument("--DataLoaderName", type = str, help = "Used in conjunction with --mode DataLoader option. Name of the DataLoader directory being created.")

parse.add_argument("--Model", type = str, help = "Specify the model you want to train. See Models.py")
parse.add_argument("--ModelOutputDir", type = str, help = "Specify the output directory that the model dumps files into.")
parse.add_argument("--ModelName", type = str, help = "Specify the name of model output (anything).")
parse.add_argument("--ModelDataLoaderInput", type = str, help = "Specify the DataLoader input file which should be loaded into the model.")

args = parse.parse_args()

if args.Mode == "Cache":
    Generate_Cache_Batches(args.SampleDir, SingleThread = False, Compiler = args.CompilerName, Custom = True, CustomDirectory = args.OutputDir) 

elif args.Mode == "DataLoader":
    GenerateTemplate_Batches(args.DataLoaderTruthLevel, args.DataLoaderAddSamples, args.OutputDir + "/DataLoaders", args.DataLoaderName) 

elif args.Mode == "Train":
    MakeDirectory(args.ModelOutputDir + "/" + args.ModelName)
    TrainEvaluate(args.Model, args.ModelName, args.ModelDataLoaderInput, ClusterOutputDir = args.ModelOutputDir)

exit()
