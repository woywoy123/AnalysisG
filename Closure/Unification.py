from Functions.Unification.Unification import Unification 
from Functions.Event.Implementations.Event import Event 

def TestUnificationEventGenerator(FileDir, Files):
    U = Unification()
    U.EventCache = True
    U.NEvent_Stop = 100
    for key, Dir in Files.items():
        U.InputSample(key, Dir)
    U.Launch()
    return True

def TestUnificationDataLoader():
    import Functions.FeatureTemplates.ParticleGeneric.EdgeFeature as ef
    import Functions.FeatureTemplates.ParticleGeneric.NodeFeature as nf
    import Functions.FeatureTemplates.ParticleGeneric.GraphFeature as gf
    from Functions.Event.Implementations.EventGraphs import EventGraphTruthTops
    
    def Test(a):
        return float(a.Test)

    U = Unification()
    U.DataCache = True
    U.EventGraph = EventGraphTruthTops
    U.AddGraphFeature("mu", gf.mu)
    U.AddGraphTruth("mu_actual", gf.mu_actual)
    U.AddNodeFeature("Test", Test)
    U.Launch()

    U = Unification()
    U.Model = ""
    U.Launch()
    
    for i in range(10):
        for n in U.TrainingSample:
            for p in U.RecallFromCache(U.TrainingSample[n], U.ProjectName + "/" + U.CacheDir):
                pass

    return True

def TestUnificationOptimizer():
    from Functions.GNN.Models.BaseLine import BaseLineModelEvent
    from Functions.Event.Implementations.EventGraphs import EventGraphTruthTopChildren
    import Functions.FeatureTemplates.ParticleGeneric.EdgeFeature as ef
    import Functions.FeatureTemplates.ParticleGeneric.NodeFeature as nf
    import Functions.FeatureTemplates.ParticleGeneric.GraphFeature as gf
    
    U = Unification()
    U.EventGraph = EventGraphTruthTopChildren
    U.EventCache = False
    U.DataCache = False
    U.ONNX_Export = True 
    U.TorchScript_Export = True
    U.Device = "cuda"
    
    # Define the Edge Features 
    U.AddEdgeTruth("Topo", ef.Index)
    
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
    U.AddGraphFeature("nTruthJet", gf.nTruthJet)
    U.AddGraphTruth("mu_actual", gf.mu_actual)
    U.AddGraphTruth("nTops", gf.nTops)

    U.Model = BaseLineModelEvent()
    U.Launch()
    return True

