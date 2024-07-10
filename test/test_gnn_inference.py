from AnalysisG.graphs.graph_bsm_4tops import GraphTruthJets
from AnalysisG.events.event_bsm_4tops import BSM4Tops
from AnalysisG.generators.analysis import Analysis
from AnalysisG.models.RecursiveGraphNeuralNetwork import *

root1 = "./samples/dilepton/*"

ev = BSM4Tops()
gr = GraphTruthJets()

gn = RecursiveGraphNeuralNetwork()
gn.o_edge = {"top_edge" : "CrossEntropyLoss"}
gn.i_node = ["pt", "eta", "phi", "energy"]
gn.i_graph = ["met", "phi"]
gn.device = "cuda:0"
gn.checkpoint_path = "./gnn-example-weights/kfold-1_model.pt"

ana = Analysis()
ana.AddSamples(root1, "tmp")
ana.AddEvent(ev, "tmp")
ana.AddGraph(gr, "tmp")
ana.AddModelInference(gn, "test-run")
#ana.Start()

