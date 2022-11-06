from AnalysisTopGNN.Generators import Analysis 
from AnalysisTopGNN.Submission import Condor
from AnalysisTopGNN.Events import Event, EventGraphTruthJetLepton
from EventFeatureTemplate import ApplyFeatures
#from BasicBaseLine import BasicBaseLineTruthJet


Submission = Condor()
Submission.ProjectName = "PathNetsTest"
GeneralDir = "/CERN/CustomAnalysisTopOutputTest/"

def EventLoaderConfig(Name, Dir):
    Ana = Analysis()
    Ana.EventImplementation = Event
    Ana.CompileSingleThread = False
    Ana.CPUThreads = 4
    Ana.EventCache = True 
    Ana.InputSample(Name, GeneralDir + Dir)
    Submission.AddJob(Name, Ana, "64GB", "24h")

def DataLoaderConfig(Name):
    Ana = Analysis()
    Ana.EventGraph = EventGraphTruthJetLepton
    Ana.DataCache = True
    Ana.FullyConnect = True
    Ana.SelfLoop = True
    ApplyFeatures(Ana)
    Ana.DataCacheOnlyCompile = [Name]
    Submission.AddJob("Data_" + Name, Ana, "4GB", "24h", [Name])



# ====== Event Loader ======== #
EventLoaderConfig("ttbar", "ttbar")
EventLoaderConfig("SingleTop", "t")
EventLoaderConfig("BSM4Top", "tttt")
EventLoaderConfig("Zmumu", "Zmumu")


# ====== Data Loader ======== #
DataLoaderConfig("ttbar")
DataLoaderConfig("SingleTop")
DataLoaderConfig("BSM4Top")
DataLoaderConfig("Zmumu")

# ====== Merge ======= #
Smpl = ["Data_SingleTop", "Data_BSM4Top", "Data_ttbar", "Data_Zmumu"]
Loader = Analysis()
Loader.Device = "cpu"
Loader.GenerateTrainingSample = True
Loader.RebuildTrainingSample = True 
Loader.TrainingSampleSize = 20
Submission.AddJob("Sample", Loader, "8GB", "24h", Smpl)


## ======= Model to Train ======== #
#TM1 = Analysis()
#TM1.RunName = "BaseLineTruthJet_MRK1"
#TM1.ONNX_Export = True
#TM1.TorchScript_Export = True
#TM1.kFold = 10
#TM1.CPUThreads = 2
#TM1.Device = "cuda"
#TM1.Epochs = 100
#TM1.Model = BasicBaseLineTruthJet()
#TM1.LearningRate = 0.01
#TM1.WeightDecay = 0.01
#TM1.BatchSize = 50
#TM1.SchedulerParams = {"gamma" : 0.9}
#TM1.DefaultScheduler = "ExponentialR"
#Submission.AddJob("BasicBaseLineTruthJet_MRK1", TM1, "12GB", "48h", ["Sample"])
#
#TM2 = Analysis()
#TM2.RunName = "BaseLineTruthJet_MRK2"
#TM2.ONNX_Export = True
#TM2.TorchScript_Export = True
#TM2.kFold = 10
#TM2.CPUThreads = 2
#TM2.Device = "cuda"
#TM2.Epochs = 100
#TM2.Model = BasicBaseLineTruthJet()
#TM2.LearningRate = 0.001
#TM2.WeightDecay = 0.01
#TM2.BatchSize = 50
#TM2.SchedulerParams = {"gamma" : 0.9}
#TM2.DefaultScheduler = "ExponentialR"
#Submission.AddJob("BasicBaseLineTruthJet_MRK2", TM2, "12GB", "48h", ["Sample"])
#
#TM3 = Analysis()
#TM3.RunName = "BaseLineTruthJet_MRK3"
#TM3.ONNX_Export = True
#TM3.TorchScript_Export = True
#TM3.kFold = 10
#TM3.CPUThreads = 2
#TM3.Device = "cuda"
#TM3.Epochs = 100
#TM3.Model = BasicBaseLineTruthJet()
#TM3.LearningRate = 0.001
#TM3.WeightDecay = 0.001
#TM3.BatchSize = 50
#TM3.SchedulerParams = {"gamma" : 0.9}
#TM3.DefaultScheduler = "ExponentialR"
#Submission.AddJob("BasicBaseLineTruthJet_MRK3", TM3, "12GB", "48h", ["Sample"])
#
#TM4 = Analysis()
#TM4.RunName = "BaseLineTruthJet_MRK4"
#TM4.ONNX_Export = True
#TM4.TorchScript_Export = True
#TM4.kFold = 10
#TM4.CPUThreads = 2
#TM4.Device = "cuda"
#TM4.Epochs = 100
#TM4.Model = BasicBaseLineTruthJet()
#TM4.LearningRate = 0.01
#TM4.WeightDecay = 0.001
#TM4.BatchSize = 50
#TM4.SchedulerParams = {"gamma" : 0.9}
#TM4.DefaultScheduler = "ExponentialR"
#Submission.AddJob("BasicBaseLineTruthJet_MRK4", TM4, "12GB", "48h", ["Sample"])
#
#TM5 = Analysis()
#TM5.RunName = "BaseLineTruthJet_MRK5"
#TM5.ONNX_Export = True
#TM5.TorchScript_Export = True
#TM5.kFold = 10
#TM5.CPUThreads = 2
#TM5.Device = "cuda"
#TM5.Epochs = 100
#TM5.Model = BasicBaseLineTruthJet()
#TM5.LearningRate = 0.001
#TM5.WeightDecay = 0.001
#TM5.BatchSize = 50
#TM5.SchedulerParams = {"base_lr" : 0.000001, "max_lr" : 0.1}
#TM5.DefaultScheduler = "CyclicLR"
#Submission.AddJob("BasicBaseLineTruthJet_MRK5", TM5, "12GB", "48h", ["Sample"])
#
#TM6 = Analysis()
#TM6.RunName = "BaseLineTruthJet_MRK6"
#TM6.ONNX_Export = True
#TM6.TorchScript_Export = True
#TM6.kFold = 10
#TM6.CPUThreads = 2
#TM6.Device = "cuda"
#TM6.Epochs = 100
#TM6.Model = BasicBaseLineTruthJet()
#TM6.LearningRate = 0.001
#TM6.WeightDecay = 0.001
#TM6.BatchSize = 50
#TM6.DefaultScheduler = None
#Submission.AddJob("BasicBaseLineTruthJet_MRK6", TM6, "12GB", "48h", ["Sample"])
#
#
#TM7 = Analysis()
#TM7.RunName = "BaseLineTruthJet_MRK7"
#TM7.ONNX_Export = True
#TM7.TorchScript_Export = True
#TM7.kFold = 10
#TM7.CPUThreads = 2
#TM7.Device = "cuda"
#TM7.Epochs = 100
#TM7.Model = BasicBaseLineTruthJet()
#TM7.LearningRate = 0.001
#TM7.WeightDecay = 0.001
#TM7.BatchSize = 50
#TM7.DefaultOptimizer = "SGD"
#TM7.DefaultScheduler = None
#Submission.AddJob("BasicBaseLineTruthJet_MRK7", TM7, "12GB", "48h", ["Sample"])

Submission.LocalDryRun()
#Submission.DisableRebuildTrainingSample = False
#Submission.DisableDataCache = False 
#Submission.DisableEventCache = False
#Submission.DumpCondorJobs()
