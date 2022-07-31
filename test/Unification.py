from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthTops, EventGraphTruthTopChildren
from AnalysisTopGNN.Models import *
import FeatureTemplates.Generic.EdgeFeature as ef
import FeatureTemplates.Generic.NodeFeature as nf
import FeatureTemplates.Generic.GraphFeature as gf

def TestUnificationEventGenerator(FileDir, Files):
    U = Analysis()
    U.EventCache = True
    U.NEvent_Stop = 100
    for key, Dir in Files.items():
        U.InputSample(key, Dir)
    U.Launch()
    return True

def TestUnificationDataLoader():
    def Test(a):
        return float(a.Test)

    U = Analysis()
    U.DataCache = True
    U.EventGraph = EventGraphTruthTops
    U.AddGraphFeature("mu", gf.mu)
    U.AddGraphTruth("mu_actual", gf.mu_actual)
    U.AddNodeFeature("Test", Test)
    U.Launch()

    U = Analysis()
    U.Model = ""
    U.Launch()
    
    for i in range(10):
        for n in U.TrainingSample:
            for p in U.RecallFromCache(U.TrainingSample[n], U.ProjectName + "/" + U.CacheDir):
                pass

    return True

def TestUnificationOptimizer():
    U = Analysis()
    U.EventGraph = EventGraphTruthTopChildren
    U.EventCache = False
    U.DataCache = True # <--- recompiles the samples
    U.ONNX_Export = True 
    U.TorchScript_Export = True
    U.Device = "cuda"
    U.Epoch = 10
    
    # Define the Edge Features 
    U.AddEdgeTruth("Topo", ef.Index)
    U.AddNodeFeature("Index", ef.Index)
    
    # Define the Node Features 
    U.AddNodeFeature("eta", nf.Index)
    U.AddNodeFeature("energy", nf.energy)
    U.AddNodeFeature("pT", nf.pT)
    U.AddNodeFeature("phi", nf.phi)
    U.AddNodeTruth("Index", nf.Index)

    # Define Graph Features
    U.AddGraphFeature("mu", gf.mu)
    U.AddGraphFeature("met", gf.met)
    U.AddGraphFeature("met_phi", gf.met_phi)
    U.AddGraphFeature("pileup", gf.pileup)
    U.AddGraphFeature("nTruthJets", gf.nTruthJets)
    U.AddGraphTruth("mu_actual", gf.mu_actual)
    U.AddGraphTruth("nTops", gf.nTops)

    U.Model = BaseLineModel(1, 2)
    U.Launch()
    return True

