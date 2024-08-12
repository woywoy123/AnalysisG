from AnalysisG.generators import Analysis

from AnalysisG.events.gnn import *
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops

from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.models import *

#root1 = "./samples/dilepton/*"
#root1 = "../test/ProjectName/results/*"

#ev = EventGNN()
ev = BSM4Tops()
gr = GraphDetector()

gn = RecursiveGraphNeuralNetwork()
gn.o_edge = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
gn.o_graph = {"ntops" : "CrossEntropyLoss", "signal" : "CrossEntropyLoss"}
gn.i_node = ["pt", "eta", "phi", "energy"]
gn.device = "cuda:0"
gn.rep = 1024
gn.checkpoint_path = "GraphDetector/RecursiveGraphNeuralNetwork/MRK-1/state/epoch-1/kfold-1_model.pt"

ana = Analysis()
ana.OutputPath = "GraphDetector"
#ana.AddSamples(root1, "tmp")
ana.GraphCache = "GraphDetector/GraphCache/GraphDetector"
#ana.AddEvent(ev, "tmp")
#ana.AddGraph(gr, "tmp")
ana.Threads = 1
ana.AddModelInference(gn, "ROOT/MRK-1/epoch-1/kfold-1")
ana.Start()

