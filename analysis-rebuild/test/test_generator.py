from AnalysisG.generators.eventgenerator import EventGenerator
from AnalysisG.generators.graphgenerator import GraphGenerator
from AnalysisG.generators.optimizer import Optimizer
from AnalysisG.events.event_bsm_4tops import BSM4Tops
from AnalysisG.graphs.graph_bsm_4tops import TruthTops
from AnalysisG.models.RecursiveGraphNeuralNetwork import *

root1 = "./samples/dilepton/*"
#root1 = "/home/tnom6927/Downloads/mc16_13_ttZ_m500/*"

x = BSM4Tops()
evg = EventGenerator()
evg.Files = root1
evg.ImportEvent(x)
#evg.CompileEvents()

tt = TruthTops()
egr = GraphGenerator()
egr.ImportGraph(tt)
#egr.AddEvents(evg)
#egr.CompileEvents()

m = RecursiveGraphNeuralNetwork()
m.o_graph = {"signal" : "CrossEntropyLoss"}
m.i_node = ["pt", "eta", "phi", "energy"]

op = Optimizer()
op.AddGeneratorPairs(evg, egr)

op.DefineModel(m)
op.DefineOptimizer("Adam")
op.Start()
