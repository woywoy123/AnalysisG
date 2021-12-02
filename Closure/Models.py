from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Optimizer import Optimizer
from Functions.GNN.Metrics import EvaluationMetrics
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.GNN.Models import EdgeConv

def Template():
    def eta(a):
        return float(a.eta)
    def energy(a):
        return float(a.e)
    def pt(a):
        return float(a.pt)
    def phi(a):
        return float(a.phi)
    def Signal(a):
        return int(a.Signal)

    ev = UnpickleObject("SignalSample.pkl")
    Loader = GenerateDataLoader()
    Loader.AddNodeFeature("e", energy)
    Loader.AddNodeFeature("eta", eta)
    Loader.AddNodeFeature("pt", pt)
    Loader.AddNodeFeature("phi", phi)
    Loader.AddNodeTruth("y", Signal)

    Loader.AddSample(ev, "nominal", "TruthChildren_init")
    Loader.ToDataLoader()
    
    for i in Loader.EventData:
        ev = Loader.EventData[i]
        for k in ev:
            PickleObject(k, "Nodes_" + str(i) + ".pkl")
            break

def TestEdgeConvModel():
    def eta(a):
        return float(1)
    def energy(a):
        return float(a.e)
    def pt(a):
        return float(a.pt)
    def phi(a):
        return float(a.phi)
    def Signal(a):
        return int(a.Signal)

    #Template()
    event1 = UnpickleObject("Nodes_10.pkl")
    event2 = UnpickleObject("Nodes_12.pkl") 

    event1.SetNodeAttribute("x", energy)
    event1.SetNodeAttribute("x", eta)
    event1.SetNodeAttribute("x", pt)
    event1.SetNodeAttribute("x", phi)
    event1.ConvertToData()

    Data1 = event1.Data
    Data2 = event2.Data

    Map1 = event1.NodeParticleMap
    Map2 = event2.NodeParticleMap

    Op = Optimizer({}, Debug = True)
    Op.Model = EdgeConv(4, 2)
    Op.DefineOptimizer()
    Op.sample = Data1
    Op.TrainClassification()

    return True

