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

smplDir = "/CERN/Samples/Processed/bsm4tops/"
#smplDir = "/nfs/dust/atlas/user/<...>/SmallSample/"

Sub = Condor()
Sub.EventCache = True 
Sub.DataCache = True 
Sub.OutputDirectory = "./Results/"
Sub.ProjectName = "TopTruthChildrenReconstruction"
Sub.Tree = "nominal"
Sub.VerboseLevel = 3

## ====== Event Generator ======= #
#Evnt = ["bsm-4-tops-mc16a", "bsm-4-tops-mc16d", "bsm-4-tops-mc16e"]
#for i in Evnt:
#    A = EventGen()
#    A.InputSample(i, smplDir + i.split("-")[-1])
#    Sub.AddJob(i, A, "12GB" , "1h")
#
#    D = DataGen()
#    D.InputSample(i)
#    ApplyFeatures(D, "TruthChildren") 
#    Sub.AddJob(i + "_Data", D, "12GB", "1h", [i])
#
## ======= Merge and Training Sample ======= #
#TrSmpl = DataGen()
#for i in Evnt:
#    TrSmpl.InputSample(i)
#TrSmpl.DataCache = True 
#TrSmpl.TrainingSampleName = "topsChildren"
#TrSmpl.TrainingPercentage = 90
#Sub.AddJob("Training", TrSmpl, "12GB", "48h", [i + "_Data" for i in Evnt])

# ======= Model Training ====== #

Opt = {
            "Optimizer1" : {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}, 
            "Optimizer2" : {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}, 
            "Optimizer3" : {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}, 
            "Optimizer4" : {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}, 

            "Optimizer5" : {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.001}}, 
            "Optimizer6" : {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.0001}}, 
            "Optimizer7" : {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}, 
            "Optimizer8" : {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}, 

            "Optimizer9" : {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.001}},
            "Optimizer10" : {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.0001}}, 
            "Optimizer11" : {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}, 
            "Optimizer12" : {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}},

            "Optimizer13" : {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.001}}, 
            "Optimizer14" : {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.0001}}, 
            "Optimizer15" : {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}, 
            "Optimizer16" : {"ADAM" : {"lr" : 0.0001, "weight_decay" : 0.0001}}, 

            "Optimizer17" : {"SGD" : {"lr" : 0.001, "weight_decay" : 0.001, "momentum" : 0.0001}}, 
            "Optimizer18" : {"SGD" : {"lr" : 0.001, "weight_decay" : 0.001, "momentum" : 0.0005}}, 
            "Optimizer19" : {"SGD" : {"lr" : 0.001, "weight_decay" : 0.001, "momentum" : 0.001}}, 
            "Optimizer20" : {"SGD" : {"lr" : 0.001, "weight_decay" : 0.001, "momentum" : 0.0015}}, 
        }

sched = {
            "Sched1" : None, 
            "Sched2" : None, 
            "Sched3" : None, 
            "Sched4" : None, 

            "Sched5" : None, 
            "Sched6" : None, 
            "Sched7" : None,  
            "Sched8" : None,  

            "Sched9" : {"ExponentialLR" : {"gamma" : 0.5}},
            "Sched10" : {"ExponentialLR" : {"gamma" : 1.0}},
            "Sched11" : {"ExponentialLR" : {"gamma" : 2.0}},
            "Sched12" : {"ExponentialLR" : {"gamma" : 4.0}},

            "Sched13" : {"CyclicLR" : {"base_lr" : 0.00001, "max_lr" : 0.0001}},
            "Sched14" : {"CyclicLR" : {"base_lr" : 0.00001, "max_lr" : 0.001}},
            "Sched15" : {"CyclicLR" : {"base_lr" : 0.00001, "max_lr" : 0.01}},
            "Sched16" : {"CyclicLR" : {"base_lr" : 0.00001, "max_lr" : 0.1}},

            "Sched17" : None, 
            "Sched18" : None, 
            "Sched19" : None, 
            "Sched20" : None, 
        }

btch = {
            "BATCH1" : 10, 
            "BATCH2" : 50, 
            "BATCH3" : 100, 
            "BATCH4" : 200, 

            "BATCH5" : 10, 
            "BATCH6" : 50,  
            "BATCH7" : 100,  
            "BATCH8" : 200,  

            "BATCH9" : 10,
            "BATCH10" : 50,
            "BATCH11" : 100,
            "BATCH12" : 200,

            "BATCH13" : 10,
            "BATCH14" : 50,
            "BATCH15" : 100,
            "BATCH16" : 200,

            "BATCH17" : 10,
            "BATCH18" : 50,
            "BATCH19" : 100,
            "BATCH20" : 200,
        }

def Evaluate(it, evl, Submit, num):
    mrk = "MRK" + it + "_" + str(num)  
    evl.TrainingSampleName = "topsChildren"
    evl.DataCache = True
    evl.EvaluateModel(direc, BasicBaseLineRecursion(), btch["BATCH" + it])
    Submit.AddJob(mrk, evl, "12GB", "1h", ["MRK"+it])
    return [mrk]

evlmod = Analysis()
evlmod.PlotNodeStatistics = True 
evlmod.PlotTrainingStatistics = True
evlmod.PlotTrainSample = True 
evlmod.PlotTestSample = True
evlmod.PlotEntireSample = True
evlmod.TrainingSampleName = "topsChildren"

wait = []
for i in range(len(Opt)):
    it = str(i+1)
    op = Optimization()
    op.RunName = "BasicBaseLineRecursion_MRK" + it 
    op.BatchSize = btch["BATCH" + it]
    op.Optimizer = Opt["Optimizer" + it]
    op.Scheduler = sched["Sched" + it]
    op.Model = BasicBaseLineRecursion()
    op.TrainingSampleName = "topsChildren"
    op.ContinueTraining = True

    Sub.AddJob("MRK" + it, op, "12GB", "1h") #, ["Training"])
  
    direc = "./Results/" + Sub.ProjectName  + "/TrainedModels/BasicBaseLineRecursion_MRK" + it
    evlmod.EvaluateModel(direc, BasicBaseLineRecursion(), btch["BATCH" + it])
 
    evl = Analysis()
    evl.PlotTrainingStatistics = True
    wait += Evaluate(it, evl, Sub, 1)
 
    evl = Analysis()
    evl.PlotTrainSample = True 
    wait += Evaluate(it, evl, Sub, 2)

    evl = Analysis()
    evl.PlotTestSample = True
    wait += Evaluate(it, evl, Sub, 3)

    evl = Analysis()
    evl.PlotEntireSample = True
    wait += Evaluate(it, evl, Sub, 4)

Sub.AddJob("Evaluator" , evlmod, "12GB", "1h", wait)
#Sub.DumpCondorJobs()
Sub.LocalDryRun()
