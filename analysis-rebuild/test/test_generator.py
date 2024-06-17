from AnalysisG.generators.eventgenerator import EventGenerator
from AnalysisG.generators.graphgenerator import GraphGenerator
from AnalysisG.generators.optimizer import Optimizer
from AnalysisG.events.event_bsm_4tops import BSM4Tops
from AnalysisG.graphs.graph_bsm_4tops import GraphTops, GraphChildren
from AnalysisG.models.RecursiveGraphNeuralNetwork import *

root1 = "./samples/dilepton/*"
#root1 = "/home/tnom6927/Downloads/mc16_13_ttZ_m500/*"

x = BSM4Tops()
evg = EventGenerator()
evg.Files = root1
evg.ImportEvent(x)

tt = GraphChildren()
egr = GraphGenerator()
egr.ImportGraph(tt)

m = RecursiveGraphNeuralNetwork()
m.o_edge = {"top_edge" : "CrossEntropyLoss"}
m.i_node = ["pt", "eta", "phi", "energy"]
m.i_graph = ["met", "phi"]
m.device = "cuda:0"

op = Optimizer()
op.AddGeneratorPairs(evg, egr)

op.DefineModel(m)
op.DefineOptimizer("Adam")
op.Start()
