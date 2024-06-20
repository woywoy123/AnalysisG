from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.generators.analysis import Analysis
from AnalysisG.events.event_bsm_4tops import BSM4Tops
from AnalysisG.graphs.graph_bsm_4tops import GraphTops, GraphChildren
from AnalysisG.models.RecursiveGraphNeuralNetwork import *

root1 = "./samples/dilepton/*"
#root1 = "/home/tnom6927/Downloads/tmp/*"

x = BSM4Tops()
tt = GraphChildren()

m = RecursiveGraphNeuralNetwork()
m.o_edge = {"top_edge" : "CrossEntropyLoss"}
m.i_node = ["pt", "eta", "phi", "energy"]
m.i_graph = ["met", "phi"]
#m.kinematic_map = {"top_edge" : "polar -> pt, eta, phi, energy"}
m.device = "cuda:0"

op = OptimizerConfig()
op.Optimizer = "adam"
op.lr = 1e-4

ana = Analysis()
ana.AddSamples(root1, "tmp")
ana.AddEvent(x, "tmp")
ana.AddGraph(tt, "tmp")
ana.AddModel(m, op, "test")
ana.kFolds = 2
ana.Start()
