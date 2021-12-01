from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Optimizer import Optimizer
from Functions.GNN.Metrics import EvaluationMetrics
from Functions.IO.IO import UnpickleObject, PickleObject

def SimpleFourTops():
    def Signal(a):
        return int(a.Signal)

    def Charge(a):
        return float(a.Signal)


    ev = UnpickleObject("SignalSample.pkl")
    Loader = GenerateDataLoader()
    Loader.AddNodeFeature("x", Charge)
    Loader.AddNodeTruth("y", Signal)
    Loader.AddSample(ev, "nominal", "TruthTops")
    Loader.ToDataLoader()

    Sig = GenerateDataLoader()
    Sig.AddNodeFeature("x", Charge)
    Sig.AddSample(ev, "nominal", "TruthTops")

    op = Optimizer(Loader)
    op.DefaultBatchSize = 20
    op.Epochs = 10
    op.kFold = 3
    op.DefineEdgeConv(1, 2)
    op.kFoldTraining()
    op.ApplyToDataSample(Sig, "Sig")    
    
    for i in Sig.DataLoader:
        for n_p in Sig.EventData[i]:
            p_v = n_p.NodeParticleMap
            for t in p_v:
                p = p_v[t] 
                try:
                    assert p.Sig == p.Signal
                except AssertionError:
                    return False

    return True



