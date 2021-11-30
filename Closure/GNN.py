from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Optimizer import Optimizer
from Functions.IO.IO import UnpickleObject, PickleObject



def SimpleFourTops():
    def Signal(a):
        return a.Signal

    ev = UnpickleObject("SignalSample.pkl")

    Loader = GenerateDataLoader()
    Loader.AddNodeFeature("x", Signal)
    Loader.AddNodeTruth("y", Signal)
    Loader.AddSample(ev, "nominal", "TruthTops")
    Loader.ToDataLoader()

    #PickleObject(Loader, "Debug.pkl")
    #Loader = UnpickleObject("Debug.pkl") 
    op = Optimizer(Loader)
    op.DefaultBatchSize = 1
    op.kFold = 2
    op.DefineEdgeConv(1, 1)
    op.kFoldTraining()

    return True



