from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthTops, EventGraphTruthTopChildren, EventGraphTruthJetLepton
from AnalysisTopGNN.Models import *
from AnalysisTopGNN.Submission import Condor
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

def TestUnificationSubmission():
    nfs = "/CERN/Analysis/"
    out = "/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/FourTopsAnalysis/test"

    # Job for creating samples
    A1 = Analysis()
    A1.ProjectName = "TopEvaluation"
    A1.CompileSingleThread = False
    A1.CPUThreads = 4
    A1.EventCache = True
    A1.OutputDir = out
    A1.EventImplementation = Event
    A1.InputSample("ttbar", nfs + "CustomAnalysisTopOutputTest/ttbar")


    A2 = Analysis()
    A2.ProjectName = "TopEvaluation"
    A2.CompileSingleThread = False
    A2.CPUThreads = 4
    A2.EventCache = True
    A2.OutputDir = out
    A2.EventImplementation = Event
    A2.InputSample("zmumu", nfs + "CustomAnalysisTopOutputTest/Zmumu")

    
    A3 = Analysis()
    A3.ProjectName = "TopEvaluation"
    A3.CompileSingleThread = False
    A3.CPUThreads = 4
    A3.EventCache = True
    A3.OutputDir = out
    A3.EventImplementation = Event
    A3.InputSample("t", nfs + "CustomAnalysisTopOutputTest/t")

    A4 = Analysis()
    A4.ProjectName = "TopEvaluation"
    A4.CompileSingleThread = False
    A4.CPUThreads = 4
    A4.EventCache = True
    A4.OutputDir = out
    A4.EventImplementation = Event
    A4.InputSample("tttt", nfs + "CustomAnalysisTopOutputTest/tttt")


    # Job for creating Dataloader
    D1 = Analysis()
    D1.ProjectName = "TopEvaluation"
    D1.OutputDir = out
    D1.DataCache = True
    D1.EventGraph = EventGraphTruthJetLepton
    D1.AddNodeTruth("from_res", nf.FromTop) 
    D1.DataCacheOnlyCompile = ["t"] 


    # Job for creating Dataloader
    D2 = Analysis()
    D2.ProjectName = "TopEvaluation"
    D2.OutputDir = out
    D2.DataCache = True
    D2.EventGraph = EventGraphTruthJetLepton
    D2.AddNodeTruth("from_res", nf.FromTop) 
    D2.DataCacheOnlyCompile = ["zmumu"] 

    # Job for creating TrainingSample
    T2 = Analysis()
    T2.ProjectName = "TopEvaluation"
    T2.OutputDir = out
    T2.GenerateTrainingSample = True

    # Job for optimization
    Op = Analysis()
    Op.ProjectName = "TopEvaluation"
    Op.Device = "cuda"
    Op.OutputDir = out
    Op.TrainWithoutCache = True
    Op.LearningRate = 0.0001
    Op.WeightDecay = 0.0001
    Op.kFold = 2
    Op.Epochs = 3
    Op.BatchSize = 20
    Op.RunName = "BasicBaseLineTruthJet"
    Op.ONNX_Export = True
    Op.TorchScript_Export = True
    Op.Model = BasicBaseLineTruthJet()

    T = Condor()
    T.DisableEventCache = False
    #T.AddJob("ttbar", A1, "10GB", "1h")
    T.AddJob("Zmumu", A2, "10GB", "1h")
    T.AddJob("t", A3, "10GB", "1h")
    #T.AddJob("tttt", A4, "10GB", "1h")

    T.AddJob("tData", D1, "10GB", "1h", ["t", "Zmumu"])
    T.AddJob("ZmumuData", D2, "10GB", ["t", "Zmumu"])
    T.AddJob("DataTraining", T2, "10GB", ["tData", "ZmumuData"])

    T.AddJob("TruthJet", Op, "10GB", ["DataTraining"])

    T.LocalDryRun() 



    return True





