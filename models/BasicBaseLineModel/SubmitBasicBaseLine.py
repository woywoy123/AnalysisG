from AnalysisTopGNN.Generators import Analysis 
from AnalysisTopGNN.Submission import Condor
from AnalysisTopGNN.Events import Event, EventGraphTruthJetLepton, EventGraphTruthTopChildren
from EventFeatureTemplate import ApplyFeatures
from V3.BasicBaseLine import BasicBaseLineRecursion
from V2.BasicBaseLine import BasicBaseLine

Submission = Condor()
Submission.ProjectName = "BasicBaseLineChildren"
GeneralDir = "/CERN/CustomAnalysisTopOutputTest/"

#GeneralDir = "/nfs/dust/atlas/user/<...>/SamplesGNN/SmallSample/"
#GeneralDir = "/nfs/dust/atlas/user/<...>/SamplesGNN/CustomAnalysisTopOutput/"

def EventLoaderConfig(Name, Dir):
    Ana = Analysis()
    Ana.Event = Event
    Ana.Threads = 12
    #Ana.NEvents = 10
    Ana.chnk = 1000
    Ana.EventCache = True 
    Ana.Tree = "nominal"
    Ana.InputSample(Name, GeneralDir + Dir)
    Submission.AddJob(Name, Ana, "64GB", "24h")

def DataLoaderConfig(Name):
    Ana = Analysis()
    Ana.EventGraph = EventGraphTruthTopChildren
    Ana.DataCache = True
    Ana.FullyConnect = True
    Ana.SelfLoop = True
    Ana.DumpHDF5 = True
    Ana.Threads = 12
    #Ana.NEvents = 10
    Ana.InputSample(Name)
    ApplyFeatures(Ana, "TruthChildren")
    Ana.DataCacheOnlyCompile = [Name]
    Submission.AddJob("Data_" + Name, Ana, "64GB", "24h", [Name])

def ModelConfig(Name):
    TM = Analysis()
    TM.RunName = Name
    TM.ONNX_Export = False
    TM.TorchScript_Export = True
    TM.kFold = 10
    TM.Threads = 12
    TM.Device = "cuda"
    TM.Epochs = 10
    TM.BatchSize = 20
    TM.chnk = 1000
    return TM

def ModelConfigRecursion(Name):
    TM = ModelConfig(Name)
    TM.VerboseLevel = 3
    TM.Model = BasicBaseLineRecursion()
    return TM

def ModelConfigNominal(Name):
    TM = ModelConfig(Name)
    TM.Model = BasicBaseLine()
    return TM

Submission.SkipEventCache = True
Submission.SkipDataCache = True

# ====== Event Loader ======== #
#EventLoaderConfig("ttbar", "ttbar")
#EventLoaderConfig("SingleTop", "t")
EventLoaderConfig("BSM4Top", "tttt")
#EventLoaderConfig("Zmumu", "Zmumu")


# ====== Data Loader ======== #
#DataLoaderConfig("ttbar")
#DataLoaderConfig("SingleTop")
DataLoaderConfig("BSM4Top")
#DataLoaderConfig("Zmumu")

# ====== Merge ======= #
Smpl = ["Data_BSM4Top"] #, "Data_ttbar", "Data_SingleTop"] #, "Data_Zmumu"]
Loader = Analysis()
#Loader.InputSample("ttbar")
#Loader.InputSample("SingleTop")
Loader.InputSample("BSM4Top")
#Loader.InputSample("Zmumu")
Loader.MergeSamples = False
Loader.GenerateTrainingSample = False
Loader.ValidationSize = 90
Loader.chnk = 100
Submission.AddJob("Sample", Loader, "64GB", "96h", Smpl)

# ======= Model to Train ======== #
inpt = ["Sample"] # <-- Wait for to finish
BaseName = "BasicBaseLineRecursion_MRK"
#
i = 1
TM1 = ModelConfigRecursion(BaseName + str(i))
#i += 1
#TM2 = ModelConfigRecursion(BaseName + str(i))
#i += 1
#TM3 = ModelConfigRecursion(BaseName + str(i))
#i += 1
#TM4 = ModelConfigRecursion(BaseName + str(i))
#i += 1
#TM5 = ModelConfigRecursion(BaseName + str(i))
#i += 1
#TM6 = ModelConfigRecursion(BaseName + str(i))
#i += 1
#TM7 = ModelConfigRecursion(BaseName + str(i))
#
TM1.LearningRate = 0.01
#TM2.LearningRate = 0.001
#TM3.LearningRate = 0.001
#TM4.LearningRate = 0.01
#TM5.LearningRate = 0.001
#TM6.LearningRate = 0.001
#TM7.LearningRate = 0.001
#
TM1.WeightDecay = 0.01
#TM2.WeightDecay = 0.01
#TM3.WeightDecay = 0.001
#TM4.WeightDecay = 0.001
#TM5.WeightDecay = 0.001
#TM6.WeightDecay = 0.001
#TM7.WeightDecay = 0.001
#
#
TM1.SchedulerParams = {"gamma" : 0.5}
TM1.DefaultScheduler = "ExponentialR"
#
#TM2.SchedulerParams = {"gamma" : 1.0}
#TM2.DefaultScheduler = "ExponentialR"
#
#TM3.SchedulerParams = {"gamma" : 1.5}
#TM3.DefaultScheduler = "ExponentialR"
#
#TM4.SchedulerParams = {"gamma" : 2.0}
#TM4.DefaultScheduler = "ExponentialR"
#
#TM5.SchedulerParams = {"base_lr" : 0.000001, "max_lr" : 0.1}
#TM5.DefaultScheduler = "CyclicLR"
#
#TM6.DefaultScheduler = None
#
#TM7.DefaultScheduler = None
#TM7.DefaultOptimizer = "SGD"
#
i = 1
Submission.AddJob(BaseName + str(i), TM1, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM2, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM3, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM4, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM5, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM6, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM7, "12GB", "48h", inpt)
#
#
#
BaseName = "BasicBaseLineNominal_MRK"
#
i = 1
TM1 = ModelConfigNominal(BaseName + str(i))
#i += 1
#TM2 = ModelConfigNominal(BaseName + str(i))
#i += 1
#TM3 = ModelConfigNominal(BaseName + str(i))
#i += 1
#TM4 = ModelConfigNominal(BaseName + str(i))
#i += 1
#TM5 = ModelConfigNominal(BaseName + str(i))
#i += 1
#TM6 = ModelConfigNominal(BaseName + str(i))
#i += 1
#TM7 = ModelConfigNominal(BaseName + str(i))
#
#TM1.LearningRate = 0.01
#TM2.LearningRate = 0.001
#TM3.LearningRate = 0.001
#TM4.LearningRate = 0.01
#TM5.LearningRate = 0.001
#TM6.LearningRate = 0.001
#TM7.LearningRate = 0.001
#
#TM1.WeightDecay = 0.01
#TM2.WeightDecay = 0.01
#TM3.WeightDecay = 0.001
#TM4.WeightDecay = 0.001
#TM5.WeightDecay = 0.001
#TM6.WeightDecay = 0.001
#TM7.WeightDecay = 0.001
#
#
#TM1.SchedulerParams = {"gamma" : 0.5}
#TM1.DefaultScheduler = "ExponentialR"
#
#TM2.SchedulerParams = {"gamma" : 1.0}
#TM2.DefaultScheduler = "ExponentialR"
#
#TM3.SchedulerParams = {"gamma" : 1.5}
#TM3.DefaultScheduler = "ExponentialR"
#
#TM4.SchedulerParams = {"gamma" : 2.0}
#TM4.DefaultScheduler = "ExponentialR"
#
#TM5.SchedulerParams = {"base_lr" : 0.000001, "max_lr" : 0.1}
#TM5.DefaultScheduler = "CyclicLR"
#
#TM6.DefaultScheduler = None
#
#TM7.DefaultScheduler = None
#TM7.DefaultOptimizer = "SGD"
#
i = 1
Submission.AddJob(BaseName + str(i), TM1, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM2, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM3, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM4, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM5, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM6, "12GB", "48h", inpt)
#i += 1
#Submission.AddJob(BaseName + str(i), TM7, "12GB", "48h", inpt)

Submission.LocalDryRun()
#Submission.DumpCondorJobs()
