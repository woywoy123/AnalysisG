import AnalysisG
from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.generators.analysis import Analysis
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops
from AnalysisG.graphs.graph_bsm_4tops import GraphTops, GraphChildren, GraphTruthJets, GraphTruthJetsNoNu
from AnalysisG.models.RecursiveGraphNeuralNetwork import *

root1 = "./samples/dilepton/*"

x = BSM4Tops()
#tt = GraphChildren()
#tt = GraphTruthJets()
tt = GraphTruthJetsNoNu()


params = [
    ("MRK-1", "adam", {"lr" : 1e-4}),
#   ("MRK-2", "adam", {"lr" : 1e-6}),
#    ("MRK-3", "adam", {"lr" : 1e-6, "amsgrad" : True}),
#    ("MRK-4", "sgd", {"lr" : 1e-3}),
#    ("MRK-5", "sgd", {"lr" : 1e-6}),
#    ("MRK-6", "sgd", {"lr" : 1e-6, "momentum" : 0.01}),
#    ("MRK-7", "sgd", {"lr" : 1e-6, "momentum" : 0.01, "dampening" : 0.01})
]

trains = []
optims = []
ana = Analysis()
for k in params:
    m1 = RecursiveGraphNeuralNetwork()
    m1.o_edge  = {
            "top_edge" : "CrossEntropyLoss",
            "res_edge" : "CrossEntropyLoss"
    }
    m1.o_graph = {
            "ntops"  : "CrossEntropyLoss",
            "signal" : "CrossEntropyLoss"
    }
    m1.i_node  = ["pt", "eta", "phi", "energy"]
    m1.i_graph = ["met", "phi"]
    m1.device  = "cuda:0"
    m1.rep = 1024
    m1.NuR = True

    opti = OptimizerConfig()
    opti.Optimizer = k[1]
    for t in k[2]: setattr(opti, t, k[2][t])

    trains.append(m1)
    optims.append(opti)

ana.AddSamples(root1, "tmp")
ana.AddEvent(x, "tmp")
ana.AddGraph(tt, "tmp")

for i in range(len(optims)): ana.AddModel(trains[i], optims[i], params[i][0])

ana.kFolds = 1
ana.Epochs = 100
ana.Targets = ["res_edge", "top_edge"]
ana.MaxRange = 1500
ana.TrainSize = 95
ana.DebugMode = False
ana.Validation = False
ana.Evaluation = False
ana.Start()
