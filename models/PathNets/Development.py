from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren
from AnalysisTopGNN.Features import ApplyFeatures
from time import time
from PathNets import PathNetsBase

if __name__ == "__main__":
  

    Ana = Analysis()
    Ana.Event = Event 
    Ana.EventGraph = EventGraphTruthTopChildren
    Ana.EventStop = 1000
    
    Ana.EventCache = False
    Ana.DataCache = True 

    Ana.DumpPickle = True
    Ana.InputSample("bsm4top", "/CERN/Samples/SingleLepton/Collections/ttH_tttt_m1000/DAOD_TOPQ1.21955717._000001.root")
    Ana.Threads = 1
    Ana.BatchSize = 10
    Ana.kFolds = 100
    Ana.ContinueTraining = False
    Ana.Device = "cuda"
    ApplyFeatures(Ana, "TruthChildren")
    Ana.Model = PathNetsBase()
    Ana.DebugMode = "accuracy-loss"
    Ana.Optimizer = {"ADAM" : { "lr" : 0.01, "weight_decay" : 0.001}}
    Ana.Launch()
