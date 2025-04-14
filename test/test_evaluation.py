import AnalysisG
from AnalysisG import Analysis
from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.events.bsm_4tops import *
from AnalysisG.core.lossfx import *
from AnalysisG.metrics import *
from AnalysisG.models import *

ev = BSM4Tops()
gr = GraphJets()

gn = Grift()
gn.name = "Grift-MRK-1"
gn.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
gn.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
gn.i_node  = ["pt", "eta", "phi", "energy", "charge"]
gn.i_graph = ["met", "phi"]
gn.device = "cuda:0"

opti = OptimizerConfig()
opti.Optimizer = "adam"
opti.lr = 1e-4

#root1 = "/home/tnom6927/Downloads/mc16_small/*"
root1 = "./samples/dilepton/*"


base_dir = "./ProjectName/Grift/"
mx = AccuracyMetric()
mx.RunNames = {
        "Grift-MRK-1::epoch-1::k-1" : base_dir + "MRK-1/state/epoch-1/kfold-1_model.pt", 
        "Grift-MRK-1::epoch-1::k-2" : base_dir + "MRK-1/state/epoch-1/kfold-2_model.pt", 
        "Grift-MRK-1::epoch-2::k-1" : base_dir + "MRK-1/state/epoch-2/kfold-1_model.pt"
}

mx.Variables = [
        "Grift-MRK-1::truth::graph::ntops", 
        "Grift-MRK-1::prediction::extra::ntops_score", 

        "Grift-MRK-1::truth::edge::top_edge",
        "Grift-MRK-1::prediction::extra::top_edge_score", 

        "Grift-MRK-1::data::node::index",
        "Grift-MRK-1::data::edge::index", 
]

ana = Analysis()
ana.TrainingDataset = "./ProjectName/sample.h5"
ana.Threads = 2
ana.BatchSize = 2
ana.AddMetric(mx, gn)
#ana.AddModel(gn, opti, "MRK-1")
ana.GraphCache = "./ProjectName/"
#ana.AddSamples(root1, "tmp")
#ana.AddEvent(ev, "tmp")
#ana.AddGraph(gr, "tmp")
#ana.DebugMode = True
ana.Validation = True
ana.Evaluation = True
ana.Training   = True
ana.Start()


