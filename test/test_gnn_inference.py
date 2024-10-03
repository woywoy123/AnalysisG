from AnalysisG.graphs.bsm_4tops import GraphTruthJets
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops
from AnalysisG.generators.analysis import Analysis
from AnalysisG.models import *

root1 = "./samples/dilepton/*"

ev = BSM4Tops()
gr = GraphTruthJets()

gn = Grift()
gn.o_edge = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
gn.o_graph = {"ntops" : "CrossEntropyLoss", "signal" : "CrossEntropyLoss"}
gn.i_node = ["pt", "eta", "phi", "energy"]
gn.i_graph = ["met", "phi"]
gn.device = "cuda:0"
gn.checkpoint_path = "./ProjectName/Experimental/MRK-1-0/state/epoch-1/kfold-1_model.pt"

gn2 = Grift()
gn2.o_edge = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
gn2.o_graph = {"ntops" : "CrossEntropyLoss", "signal" : "CrossEntropyLoss"}
gn2.i_node = ["pt", "eta", "phi", "energy"]
gn2.i_graph = ["met", "phi"]
gn2.device = "cuda:1"
gn2.checkpoint_path = "./ProjectName/Experimental/MRK-1-1/state/epoch-1/kfold-1_model.pt"

ana = Analysis()
ana.Threads = 40
ana.AddSamples(root1, "tmp")
ana.AddEvent(ev, "tmp")
ana.AddGraph(gr, "tmp")
ana.AddModelInference(gn, "test-run")
ana.AddModelInference(gn2, "test-run-2")
ana.Start()

