from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren
from AnalysisTopGNN.Features import ApplyFeatures
from time import time
from PathNets import PathNetsBase

def TestCombinatorial():
    import torch
    from PathNetOptimizer import CombinatorialCPU
    
    t1 = time()
    adj = CombinatorialCPU(32, 4, "cuda")
    print(time() - t1) 


if __name__ == "__main__":
  
    #TestCombinatorial()
    direc = "/CERN/Samples/Processed/bsm4tops"

    Ana = Analysis()
    Ana.Event = Event 
    Ana.EventGraph = EventGraphTruthTopChildren
    Ana.EventStop = 100
    
    Ana.EventCache = False
    Ana.DataCache = True 

    Ana.DumpPickle = True
    Ana.InputSample("bsm4top", direc + "/mc16a/DAOD_TOPQ1.21955713._000001.root")
    Ana.Threads = 1
    Ana.BatchSize = 1
    Ana.Device = "cuda"
    ApplyFeatures(Ana, "TruthChildren")
    Ana.Model = PathNetsBase()
    Ana.DebugMode = "accuracy"
    Ana.Optimizer = {"ADAM" : { "lr" : 0.0001, "weight_decay" : 0.0001}}
    Ana.Launch()
