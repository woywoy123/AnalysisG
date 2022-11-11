from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.Events import EventGraphTruthTopChildren
from AnalysisTopGNN.Submission import Condor
from AnalysisTopGNN.Generators import Analysis
from Templates.EventFeatureTemplate import ApplyFeatures
from BasicBaseLineV3.BasicBaseLine import BasicBaseLineRecursion

def EventGen():
    Ana = Analysis()
    Ana.Event = Event
    Ana.Threads = 4
    Ana.chnk = 100
    Ana.EventStop = 100
    Ana.EventCache = True
    Ana.DumpHDF5 = False
    Ana.DumpPickle = True
    return Ana

def DataGen():
    Ana = Analysis()
    Ana.Threads = 4
    Ana.chnk = 100
    Ana.EventGraph = EventGraphTruthTopChildren
    Ana.EventStop = 100
    Ana.DataCache = True
    Ana.DumpHDF5 = True
    Ana.DumpPickle = False
    return Ana

def Optimization():
    Ana = Analysis()
    Ana.Threads = 2
    Ana.chnk = 10
    Ana.Epochs = 4
    Ana.kFolds = 10
    Ana.Device = "cuda"
    return Ana




#smplDir = "/home/<....>/Downloads/CustomAnalysisTopOutputTest/"
smplDir = "/nfs/dust/atlas/user/woywoy12/SmallSample/"

Sub = Condor()
Sub.EventCache = True 
Sub.DataCache = True 
Sub.OutputDirectory = "./Results/"
Sub.ProjectName = "TopTruthChildrenReconstruction"
Sub.Tree = "nominal"
Sub.VerboseLevel = 1

# ====== Event Generator ======= #
A1 = EventGen()
A1.InputSample("tttt", smplDir + "tttt")

A2 = EventGen()
A2.InputSample("t", smplDir + "t")

A3 = EventGen()
A3.InputSample("ttbar", smplDir + "ttbar")

#A4 = EventGen()
#A4.InputSample("Zmumu", smplDir + "Zmumu")

# ====== Graph Generator ======= #
D1 = DataGen()
D1.InputSample("tttt")
ApplyFeatures(D1, "TruthChildren")

D2 = DataGen()
D2.InputSample("t")
ApplyFeatures(D2, "TruthChildren")

D3 = DataGen()
D3.InputSample("ttbar")
ApplyFeatures(D3, "TruthChildren")
#
#D4 = DataGen()
#D4.InputSample("Zmumu")

# ======= Merge and Training Sample ======= #
TrSmpl = DataGen()
TrSmpl.InputSample("tttt")
TrSmpl.InputSample("t")
TrSmpl.InputSample("ttbar")
TrSmpl.DataCache = True 
TrSmpl.TrainingSampleName = "topsChildren"
TrSmpl.TrainingPercentage = 90

# ======= Change Batch Size ====== #
i = 1
Tr1 = Optimization()
Tr1.TrainingSampleName = "topsChildren"
Tr1.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr1.BatchSize = 10
Tr1.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}
Tr1.Model = BasicBaseLineRecursion()

i += 1
Tr2 = Optimization()
Tr2.TrainingSampleName = "topsChildren"
Tr2.BatchSize = 50
Tr2.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr2.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}
Tr2.Model = BasicBaseLineRecursion()

i += 1
Tr3 = Optimization()
Tr3.TrainingSampleName = "topsChildren"
Tr3.BatchSize = 100
Tr3.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr3.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}
Tr3.Model = BasicBaseLineRecursion()

i += 1
Tr4 = Optimization()
Tr4.TrainingSampleName = "topsChildren"
Tr4.BatchSize = 200
Tr4.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr4.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}
Tr4.Model = BasicBaseLineRecursion()

# ======= Change Learning And Decay ====== #
i += 1
Tr5 = Optimization()
Tr5.TrainingSampleName = "topsChildren"
Tr5.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr5.BatchSize = 10
Tr5.Optimizer = {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.001}}
Tr5.Model = BasicBaseLineRecursion()

i += 1
Tr6 = Optimization()
Tr6.TrainingSampleName = "topsChildren"
Tr6.BatchSize = 50
Tr6.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr6.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.0001}}
Tr6.Model = BasicBaseLineRecursion()

i += 1
Tr7 = Optimization()
Tr7.TrainingSampleName = "topsChildren"
Tr7.BatchSize = 100
Tr7.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr7.Optimizer = {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}
Tr7.Model = BasicBaseLineRecursion()

i += 1
Tr8 = Optimization()
Tr8.TrainingSampleName = "topsChildren"
Tr8.BatchSize = 200
Tr8.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr8.Optimizer = {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}
Tr8.Model = BasicBaseLineRecursion()

# ======= Change Scheduler ExponentialLR ====== #
i += 1
Tr9 = Optimization()
Tr9.TrainingSampleName = "topsChildren"
Tr9.BatchSize = 10
Tr9.Scheduler = {"ExponentialLR" : {"gamma" : 0.5}}
Tr9.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr9.Optimizer = {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.001}}
Tr9.Model = BasicBaseLineRecursion()

i += 1
Tr10 = Optimization()
Tr10.TrainingSampleName = "topsChildren"
Tr10.BatchSize = 50
Tr10.Scheduler = {"ExponentialLR" : {"gamma" : 1.0}}
Tr10.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr10.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.0001}}
Tr10.Model = BasicBaseLineRecursion()

i += 1
Tr11 = Optimization()
Tr11.TrainingSampleName = "topsChildren"
Tr11.BatchSize = 100
Tr11.Scheduler = {"ExponentialLR" : {"gamma" : 2.0}}
Tr11.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr11.Optimizer = {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}
Tr11.Model = BasicBaseLineRecursion()

i += 1
Tr12 = Optimization()
Tr12.TrainingSampleName = "topsChildren"
Tr12.BatchSize = 200
Tr12.Scheduler = {"ExponentialLR" : {"gamma" : 4.0}}
Tr12.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr12.Optimizer = {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}
Tr12.Model = BasicBaseLineRecursion()

# ======= Change Scheduler CyclicLR ====== #
i += 1
Tr13 = Optimization()
Tr13.TrainingSampleName = "topsChildren"
Tr13.BatchSize = 10
Tr13.Scheduler = {"CyclicLR" : {"base_lr" : 0.00001, "max_lr" : 0.0001}}
Tr13.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr13.Optimizer = {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.001}}
Tr13.Model = BasicBaseLineRecursion()

i += 1
Tr14 = Optimization()
Tr14.TrainingSampleName = "topsChildren"
Tr14.BatchSize = 50
Tr14.Scheduler = {"CyclicLR" : {"base_lr" : 0.00001, "max_lr" : 0.001}}
Tr14.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr14.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.0001}}
Tr14.Model = BasicBaseLineRecursion()

i += 1
Tr15 = Optimization()
Tr15.TrainingSampleName = "topsChildren"
Tr15.BatchSize = 100
Tr15.Scheduler = {"CyclicLR" : {"base_lr" : 0.00001, "max_lr" : 0.01}}
Tr15.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr15.Optimizer = {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}
Tr15.Model = BasicBaseLineRecursion()

i += 1
Tr16 = Optimization()
Tr16.TrainingSampleName = "topsChildren"
Tr16.BatchSize = 200
Tr16.Scheduler = {"CyclicLR" : {"base_lr" : 0.00001, "max_lr" : 0.1}}
Tr16.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr16.Optimizer = {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}
Tr16.Model = BasicBaseLineRecursion()

# ======= Change Optimizer ====== #
i += 1
Tr17 = Optimization()
Tr17.TrainingSampleName = "topsChildren"
Tr17.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr17.BatchSize = 10
Tr17.Optimizer = {"SGD" : {"lr" : 0.001, "weight_decay" : 0.001, "momentum" : 0.0001}}
Tr17.Model = BasicBaseLineRecursion()

i += 1
Tr18 = Optimization()
Tr18.TrainingSampleName = "topsChildren"
Tr18.BatchSize = 50
Tr18.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr18.Optimizer = {"SGD" : {"lr" : 0.001, "weight_decay" : 0.001, "momentum" : 0.0005}}
Tr18.Model = BasicBaseLineRecursion()

i += 1
Tr19 = Optimization()
Tr19.TrainingSampleName = "topsChildren"
Tr19.BatchSize = 100
Tr19.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr19.Optimizer = {"SGD" : {"lr" : 0.001, "weight_decay" : 0.001, "momentum" : 0.001}}
Tr19.Model = BasicBaseLineRecursion()

i += 1
Tr20 = Optimization()
Tr20.TrainingSampleName = "topsChildren"
Tr20.BatchSize = 200
Tr20.RunName = "BasicBaseLineRecursion_MRK" + str(i)
Tr20.Optimizer = {"SGD" : {"lr" : 0.001, "weight_decay" : 0.001, "momentum" : 0.0015}}
Tr20.Model = BasicBaseLineRecursion()

SampleJobsEvent = {"tttt" : A1, "t" : A2, "ttbar" : A3}
for i in SampleJobsEvent:
    Sub.AddJob(i, SampleJobsEvent[i], "12GB", "1h")


SampleJobsEvent = {"tttt" : A1, "t" : A2, "ttbar" : A3}
for i in SampleJobsEvent:
    Sub.AddJob(i, SampleJobsEvent[i], "12GB", "1h")

SampleJobsData = {"tttt" : D1, "t" : D2, "ttbar" : D3}
for i in SampleJobsData:
    Sub.AddJob(i + "_Data", SampleJobsData[i], "12GB", "1h", [i])

Sub.AddJob("Training", TrSmpl, "12GB", "48h", [i + "_Data" for i in SampleJobsData])

TrainJobsData = {
    "MRK1" : Tr1, "MRK2" : Tr2, "MRK3" : Tr3, 
    "MRK4" : Tr4, "MRK5" : Tr5, "MRK6" : Tr6,
    "MRK7" : Tr7, "MRK8" : Tr8, "MRK9" : Tr9,
    "MRK10" : Tr10, "MRK11" : Tr11, "MRK12" : Tr12,
    "MRK13" : Tr13, "MRK14" : Tr14, "MRK15" : Tr15,
    "MRK16" : Tr16, "MRK17" : Tr17, "MRK18" : Tr18,
    "MRK19" : Tr19, "MRK20" : Tr20
}

for i in TrainJobsData:
    Sub.AddJob(i, TrainJobsData[i], "12GB", "1h", ["Training"])
Sub.DumpCondorJobs()
#Sub.LocalDryRun()
