from AnalysisG.generators import Analysis

from AnalysisG.events.gnn import *
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops

from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.models.RecursiveGraphNeuralNetwork import *

root1 = "./samples/dilepton/*"
root1 = "../test/ProjectName/results/*"

ev = EventGNN()
#ev = BSM4Tops()
#gr = GraphDetector()

#gn = RecursiveGraphNeuralNetwork()
#gn.o_edge = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
#gn.o_graph = {"ntops" : "CrossEntropyLoss", "signal" : "CrossEntropyLoss"}
#gn.i_node = ["pt", "eta", "phi", "energy"]
#gn.device = "cuda:0"
#gn.checkpoint_path = "../test/ProjectName/RecursiveGraphNeuralNetwork/test/state/epoch-46/kfold-1_model.pt"


ana = Analysis()
ana.AddSamples(root1, "tmp")
ana.AddEvent(ev, "tmp")
#ana.AddGraph(gr, "tmp")
#ana.AddModelInference(gn, "results")
ana.Start()

